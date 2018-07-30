
import numpy as np
import glob, random
import time

import matplotlib.pyplot as plt
from skimage.transform import resize

import tensorflow as tf
from keras import backend as K
from keras.models import Model, Sequential
from keras.activations import softmax
from keras.objectives import mean_squared_error as mse
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Conv2DTranspose, Reshape, Activation

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
    def __init__(self, M=8, N=12):  # TODO: Better param
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.epochs = 10
        self.batch_size = 64

        self.N = N  # number of variables
        self.M = M  # number of values per variable
        self.org_dim = (96, 96, 3)
        self.input_dim = (64, 64, 3)
        self.data_dim = np.prod(self.input_dim)
        self.activation = 'relu'

        self.tau = K.variable(5.0, name="temperature")
        self.tau0 = K.variable(5.0, name="start_temperature")
        self.tau_min = 0.5
        half_life = 100
        self.anneal_rate = np.log(2)/half_life

        self.model, self.decoder = self._build_model()

    def _build_model(self):
        self.encoder_input = Input(shape=self.input_dim)
        h = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation=self.activation)(self.encoder_input )
        h = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation=self.activation)(h)
        h = Flatten()(h)
        logits_y = Dense(self.M * self.N, activation=None)(h)

        z_lay = Lambda(self.sampling, output_shape=(self.M * self.N,))
        z = z_lay(logits_y)
        self.gumbel_logits = Activation(None)(z)  # make into layer --> nice properties

        z_1 = Reshape((1, 1, self.N * self.M))
        z_2 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding="same", activation=self.activation)
        z_3 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding="same", activation=self.activation)
        z_4 = Conv2DTranspose(filters=16, kernel_size=4, strides=2, padding="same", activation=self.activation)
        z_5 = Conv2DTranspose(filters=16, kernel_size=4, strides=4, padding="same", activation=self.activation)
        x_hat = Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding="same", activation="sigmoid")

        self.vae_output = x_hat(z_5(z_4(z_3(z_2(z_1(self.gumbel_logits))))))

        self.decode_input = Input(shape=(self.M * self.N,))
        self.decoder_output = x_hat(z_5(z_4(z_3(z_2(z_1(self.decode_input))))))
        generator = Model(self.decode_input, self.decoder_output)

        vae = Model(self.encoder_input, self.vae_output)
        vae.compile(optimizer='adam', loss=self.gumbel_loss)
        vae.summary()

        return vae, generator

    def gumbel_loss(self, x, x_hat):
        q_y = K.reshape(self.gumbel_logits, (-1, self.N, self.M))
        q_y = softmax(q_y)
        log_q_y = K.log(q_y + 1e-20)
        self.KL = q_y * (log_q_y - K.log(1.0 / self.M))
        self.KL = K.sum(self.KL, axis=(1, 2))

        x = K.reshape(x, (1, -1))
        x_hat = K.reshape(x_hat, (1, -1))
        self.rec_loss = self.data_dim * mse(x, x_hat)

        elbo = self.rec_loss - self.KL
        return elbo

    def sampling(self, logits_y):
        U = K.random_uniform(K.shape(logits_y), 0, 1)
        y = logits_y - K.log(-K.log(U + 1e-20) + 1e-20)  # logits + gumbel noise
        y = K.reshape(y, (-1, self.N, self.M)) / self.tau
        y = softmax(y)
        y = K.reshape(y, (-1, self.N * self.M))
        return y

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, data, validation_split=0.2):
        # earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        # tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_images=True)
        # callbacks_list = [earlystop, tensorboard]
        # TODO: ^

        self.model.fit(data, data,
                       shuffle=True,
                       epochs=1,
                       batch_size=self.batch_size)

        self.model.save_weights('./vae/weights.h5')

    def get_embedding(self, sess, observation, hard=False):
        #TODO
        if hard:
            raise Exception
        else:
            return self.encoder.predict(observation)[1]

def data_iterator(batch_size):
    data_files = glob.glob('../data/obs_data_VAE_*')
    while True:
        data = np.load(random.sample(data_files, 1)[0])
        np.random.shuffle(data)
        np.random.shuffle(data)
        N = data.shape[0]
        start = np.random.randint(0, N-batch_size)
        yield data[start:start+batch_size]


def show_pred(title, data, pred):
    plt.figure(figsize=(10,10))
    max = len(pred)
    for i in range(max):
        # print('Plotting vae pred:', i, '/', max)
        plt.subplot(121)
        plt.imshow(data[i])
        plt.title(str(i))

        plt.subplot(122)
        plt.imshow(pred[i])
        plt.title(title)
        plt.savefig('./videos/CarRacing-'+title+'-'+str(i)+'.png', bbox_inches='tight')

def train_vae():

    vae = Network()
    try:
      vae.set_weights('./vae/weights.h5')
    except:
      print("No weights found in ./vae/weights.h5 exists")

    start_batch = 0
    max_batch = 14
    test_batch = 15
    # TODO: ^: Don't hardcode like this

    try:
        for epoch in range(vae.epochs):
            print('EPOCH ' + str(epoch))
            for batch_num in range(start_batch, max_batch + 1):
                batch_to_load = '../data/obs_data_VAE_' + str(batch_num) + '.npy'
                try:
                    data = np.load(batch_to_load)
                    # data = data[:150]  # TODO!! <-- ONLY FOR TESTINg

                    data = resize(data, (len(data), 64, 64, 3), anti_aliasing=True)
                    print('Found batch at {}...current data size = {} episodes'.format(
                        batch_to_load, len(data)))
                except:
                    print('Unable to load:', batch_to_load)
                    continue

                vae.train(data)
                vae.model.save_weights('./vae/weights.h5')

            ## Testing
            print('Testing')
            batch_to_load = '../data/obs_data_VAE_' + str(test_batch) + '.npy'
            data = np.load(batch_to_load)
            data = data[50:50+96]
            data = resize(data, (len(data), 64, 64, 3), anti_aliasing=True)

            print("Validation score", vae.model.evaluate(data, data))

            pred = vae.model.predict(data)
            title = "e{}".format(epoch)
            show_pred(title, data, pred)
            print('')


    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")

    except Exception as e:
        print("Exception: {}".format(e))


def load_vae():

    raise NotImplementedError
    # sess = tf.InteractiveSession()
    #
    # graph = tf.Graph()
    # with graph.as_default():
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     sess = tf.Session(config=config, graph=graph)
    #
    #     network = Network()
    #     init = tf.global_variables_initializer()
    #     sess.run(init)
    #
    #     saver = tf.train.Saver(max_to_keep=1)
    #     training_data = data_iterator(batch_size=128)
    #
    #     try:
    #         saver.restore(sess, tf.train.latest_checkpoint(model_path))
    #     except:
    #         raise ImportError("Could not restore saved model")
    #
    #     return sess, network

if __name__ == '__main__':
    train_vae()
