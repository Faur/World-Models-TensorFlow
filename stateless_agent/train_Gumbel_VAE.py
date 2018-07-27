import tensorflow as tf
import numpy as np
import glob, random, os

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from tensorboard_logging import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = "saved_models/"
exp_name = '_temp_anneal_2'
model_name = model_path + 'Gumbel_model' + exp_name

class Network(object):
    # Create model
    def __init__(self, K=32, N=16):
        self.N = N  # number of variables
        self.K = K  # number of values per variable
        self.temperature = tf.placeholder(tf.float32, [], name='temperature')

        self.image = tf.placeholder(tf.float32, [None, 96, 96, 3], name='image')
        self.resized_image = tf.image.resize_images(self.image, [64, 64])
        self.data_dim = 64*64*3
        tf.summary.image('resized_image', self.resized_image, 20)

        self.gumbel_logits = self.encoder(self.resized_image)
        self.z = self.sample_z(self.gumbel_logits, self.temperature)
        self.reconstructions = self.decoder(self.z)
        tf.summary.image('reconstructions', self.reconstructions, 20)

        self.loss, rec_loss, KL_loss = self.compute_loss()
        tf.summary.scalar('train/total_loss', self.loss)
        tf.summary.scalar('train/rec_loss', rec_loss)
        tf.summary.scalar('train/KL_loss', KL_loss)

        self.merged = tf.summary.merge_all()

    def encoder(self, x):
        x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)

        x = tf.layers.flatten(x)
        logits = tf.layers.dense(x, self.N*self.K, activation=None)
        logits = tf.reshape(logits, [-1, self.K])
        return logits

    def sample_z(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""

        def sample_gumbel(shape, eps=1e-20):
            """Sample from Gumbel(0, 1)"""
            U = tf.random_uniform(shape, minval=0, maxval=1)
            return -tf.log(-tf.log(U + eps) + eps)

        z = logits + sample_gumbel(tf.shape(logits))
        z = tf.nn.softmax(z / temperature)
        # z = tf.layers.flatten(z)
        z = tf.reshape(z, [-1, self.N*self.K])
        return z

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

        reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis=1)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)

        q_y = tf.nn.softmax(self.gumbel_logits)
        log_q_y = tf.log(q_y + 1e-20)
        KL = tf.reshape(q_y * (log_q_y - tf.log(1.0 / self.K)), [-1, self.N, self.K])
        KL = tf.reduce_sum(KL, [1, 2])
        KL = tf.reduce_mean(KL)

        # kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.z_mu**2 - 1. - self.z_logvar, 1)
        # elbo = tf.reduce_sum(p_x.log_prob(x), 1) - KL
        # loss = tf.reduce_mean(-elbo)
        # tf.reduce_mean(reconstruction_loss + kl_loss)

        loss = reconstruction_loss - KL
        return loss, reconstruction_loss, KL

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

    writer = tf.summary.FileWriter('logdir/'+exp_name)
    logger = Logger('logdir/'+exp_name)

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

    temp_0 = 5
    temp_min = 0.5
    temp_anneal = 0.0003
    temp_STUPID_CONSTANT = 8000  # TODO

    try:
        while True:
            images = next(training_data)
            temperature = np.max(temp_min, temp_0*np.exp(-temp_anneal*step / temp_STUPID_CONSTANT))

            _, loss_value = sess.run([train_op, network.loss],
                                feed_dict={network.image: images,
                                           network.temperature: temperature})

            if np.isnan(loss_value):
                raise ValueError('Loss value is NaN')
            if step % 10 == 0 and step > 0:
                [summary] = sess.run([network.merged],
                                                  feed_dict={network.image: images,
                                                             network.temperature: temperature})
                writer.add_summary(summary, step)
                logger.log_scalar('train/temperature', temperature, step)
                print('step {}: training loss {:.6f}'.format(step, loss_value))
                save_path = saver.save(sess, model_name, global_step=global_step)
            if loss_value <= 0:
            # if loss_value <= 35:
                print('step {}: training loss {:.6f}'.format(step, loss_value))
                save_path = saver.save(sess, model_name, global_step=global_step)
                break
            step += 1

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
