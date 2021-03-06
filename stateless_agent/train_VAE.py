import tensorflow as tf
import numpy as np
import glob, random, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = "saved_models/"
model_name = model_path + 'model'

_EXP_NAME = "continuous"
_EMBEDDING_SIZE = 32  # TODO: Handle this better!

class Network(object):
    # Create model
    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, 96, 96, 3], name='image')
        self.resized_image = tf.image.resize_images(self.image, [64, 64])
        tf.summary.image('resized_image', self.resized_image, 20)

        self.z_mu, self.z_logvar = self.encoder(self.resized_image)
        self.z = self.sample_z(self.z_mu, self.z_logvar)
        self.reconstructions = self.decoder(self.z)
        tf.summary.image('reconstructions', self.reconstructions, 20)

        self.merged = tf.summary.merge_all()

        self.loss = self.compute_loss()

    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps

    def encoder(self, x):
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)

        x = tf.layers.flatten(x)
        z_mu = tf.layers.dense(x, units=_EMBEDDING_SIZE, name='z_mu')
        z_logvar = tf.layers.dense(x, units=_EMBEDDING_SIZE, name='z_logvar')
        return z_mu, z_logvar

    def decoder(self, z):
        x = tf.layers.dense(z, 1024, activation=None)
        x = tf.reshape(x, [-1, 1, 1, 1024])
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=6, strides=2, padding='valid', activation=tf.nn.sigmoid)
        return x

    def compute_loss(self):
        logits_flat = tf.layers.flatten(self.reconstructions)
        labels_flat = tf.layers.flatten(self.resized_image)
        reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis = 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        return vae_loss

    def get_embedding(self, sess, observation):
        return sess.run(self.z, feed_dict={self.image: observation[None, :, :, :]})

    def normalize_observation(self, observation):
        return observation.astype('float32') / 255.

    def predict(self, sess, data):
        print(sess)
        pred, mu, z_logvar, z = sess.run([self.reconstructions, self.z_mu, self.z_logvar, self.z], feed_dict={self.image:data})
        sigma = np.exp(z_logvar/2)
        return pred, mu, sigma, z


def data_iterator(batch_size):
    data_files = glob.glob('../data/obs_data_VAE_*')
    while True:
        data = np.load(random.sample(data_files, 1)[0])
        np.random.shuffle(data)
        np.random.shuffle(data)
        N = data.shape[0]
        start = np.random.randint(0, N-batch_size)
        yield data[start:start+batch_size]

def train_vae():
    sess = tf.InteractiveSession()

    global_step = tf.Variable(0, name='global_step', trainable=False)

    writer = tf.summary.FileWriter('logdir')

    network = Network()
    train_op = tf.train.AdamOptimizer(0.001).minimize(network.loss, global_step=global_step)
    tf.global_variables_initializer().run()

    saver = tf.train.Saver(max_to_keep=1)
    step = global_step.eval()
    training_data = data_iterator(batch_size=128)

    try:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print("Model restored from: {}".format(model_path))
    except:
        print("Could not restore saved model")

    try:
        while True:
            images = next(training_data)
            _, loss_value, summary = sess.run([train_op, network.loss, network.merged],
                                feed_dict={network.image: images})
            writer.add_summary(summary, step)

            if np.isnan(loss_value):
                raise ValueError('Loss value is NaN')
            if step % 10 == 0 and step > 0:
                print ('step {}: training loss {:.6f}'.format(step, loss_value))
                save_path = saver.save(sess, model_name, global_step=global_step)
            if loss_value <= 35:
                print ('step {}: training loss {:.6f}'.format(step, loss_value))
                save_path = saver.save(sess, model_name, global_step=global_step)
                break
            step+=1

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")

    except Exception as e:
        print("Exception: {}".format(e))


def load_vae():

    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=graph)

        network = Network()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=1)
        training_data = data_iterator(batch_size=128)

        try:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        except:
            raise ImportError("Could not restore saved model")

        return sess, network

if __name__ == '__main__':
    train_vae()
