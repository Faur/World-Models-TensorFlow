
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
from keras.optimizers import Adam
from keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Conv2DTranspose, Reshape, Activation

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from tensorboard_logging import Logger

import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = "saved_models/"
exp_name = '_temp_anneal_2'
model_name = model_path + 'Gumbel_model' + exp_name

_GLOBAL_N = 16
_GLOBAL_M = 8
_EMBEDDING_SIZE = _GLOBAL_N * _GLOBAL_M  # TODO: Handle this better!

class Network(object):
    # Create model
    def __init__(self, N=_GLOBAL_N, M=_GLOBAL_M, lr = 0.00005):  # TODO: Better param handling
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.epochs = 1000
        self.batch_size = 64
        self.learning_rate = lr  # Default: 0.001

        self.N = N  # number of variables
        self.M = M  # number of values per variable
        self.org_dim = (96, 96, 3)
        self.input_dim = (64, 64, 3)
        self.data_dim = np.prod(self.input_dim)
        self.activation = 'relu'

        self.tau = K.variable(5.0, name="temperature")
        # self.tau0 = K.variable(5.0, name="start_temperature")
        self.tau0 = 5.0
        self.tau_min = 0.1
        half_life = 10
        self.anneal_rate = np.log(2)/half_life

        self.model, self.decoder, self.tester = self._build_model()

    def _build_model(self):
        self.encoder_input = Input(shape=self.input_dim)
        # h = tf.image.resize_images(self.image, [64, 64])
        # https://stackoverflow.com/questions/42260265/resizing-an-input-image-in-a-keras-lambda-layer
        h = self.encoder_input
        use_reduced = False
        print("use_reduced", use_reduced)

        if use_reduced:
            h = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation=self.activation)(h)
            h = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation=self.activation)(h)
            h = Flatten()(h)
        else:
            CONV_FILTERS = [32, 64, 128, 256]
            CONV_KERNEL_SIZES = [4, 4, 4, 4]
            CONV_STRIDES = [2, 2, 2, 2]
            CONV_ACTIVATIONS = ['relu', 'relu', 'relu', 'relu']

            vae_c1 = Conv2D(filters=CONV_FILTERS[0], kernel_size=CONV_KERNEL_SIZES[0], strides=CONV_STRIDES[0],
                            activation=CONV_ACTIVATIONS[0])(h)
            vae_c2 = Conv2D(filters=CONV_FILTERS[1], kernel_size=CONV_KERNEL_SIZES[1], strides=CONV_STRIDES[1],
                            activation=CONV_ACTIVATIONS[0])(vae_c1)
            vae_c3 = Conv2D(filters=CONV_FILTERS[2], kernel_size=CONV_KERNEL_SIZES[2], strides=CONV_STRIDES[2],
                            activation=CONV_ACTIVATIONS[0])(vae_c2)
            vae_c4 = Conv2D(filters=CONV_FILTERS[3], kernel_size=CONV_KERNEL_SIZES[3], strides=CONV_STRIDES[3],
                            activation=CONV_ACTIVATIONS[0])(vae_c3)
            h = Flatten()(vae_c4)
            # h = Dense(1024, activation='relu')(h)

        self.encoder_logits = Dense(self.M * self.N, activation=None)(h)
        z = Lambda(self.sampling, output_shape=(self.M * self.N,))(self.encoder_logits)
        self.gumbel_logits = Activation(None)(z)  # make into layer --> nice properties

        if use_reduced:
            z_1 = Reshape((1, 1, self.N * self.M))
            z_2 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding="same", activation=self.activation)
            z_3 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding="same", activation=self.activation)
            z_4 = Conv2DTranspose(filters=16, kernel_size=4, strides=2, padding="same", activation=self.activation)
            z_5 = Conv2DTranspose(filters=16, kernel_size=4, strides=4, padding="same", activation=self.activation)
            x_hat = Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding="same", activation="sigmoid")

            self.decode_input = Input(shape=(self.M * self.N,))
            self.decoder_output = x_hat(z_5(z_4(z_3(z_2(z_1(self.decode_input))))))
            self.vae_output = x_hat(z_5(z_4(z_3(z_2(z_1(self.gumbel_logits))))))
        else:
            # CONV_T_FILTERS = [64,64,32,3]
            CONV_T_FILTERS = [128, 64, 32, 3]
            CONV_T_KERNEL_SIZES = [5, 5, 6, 6]
            CONV_T_STRIDES = [2, 2, 2, 2]
            CONV_T_ACTIVATIONS = ['relu', 'relu', 'relu', 'sigmoid']

            vae_dense = Dense(1024)
            vae_z_out = Reshape((1,1,1024))
            vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0])
            vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1])
            vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2])
            vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3])

            self.decode_input = Input(shape=(self.M * self.N,))
            self.decoder_output = vae_d4(vae_d3(vae_d2(vae_d1(vae_z_out(vae_dense(self.decode_input))))))
            self.vae_output = vae_d4(vae_d3(vae_d2(vae_d1(vae_z_out(vae_dense(self.gumbel_logits))))))
        generator = Model(self.decode_input, self.decoder_output)

        vae = Model(self.encoder_input, self.vae_output)
        optimizer = Adam(self.learning_rate)
        vae.compile(optimizer=optimizer, loss=self.gumbel_loss)
        vae.summary()

        tester = Model(self.encoder_input, [self.encoder_logits, self.gumbel_logits, self.vae_output])

        return vae, generator, tester

    def gumbel_loss(self, x, x_hat):
        q_y = K.reshape(self.encoder_logits, (-1, self.N, self.M))
        q_y = softmax(q_y)
        log_q_y = K.log(q_y + 1e-20)
        KL = q_y * (log_q_y - K.log(1.0 / self.M))
        KL = K.sum(KL, axis=(1, 2))

        x = K.reshape(x, (1, -1))
        x_hat = K.reshape(x_hat, (1, -1))
        rec_loss = self.data_dim * mse(x, x_hat)

        elbo = rec_loss - KL
        return elbo

    def sampling(self, logits):
        # TODO: Check - should this be logits or should we softmax?
        U = K.random_uniform(K.shape(logits), 0, 1)
        y = logits - K.log(-K.log(U + 1e-20) + 1e-20)  # logits + gumbel noise
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
                       batch_size=self.batch_size,
                       verbose=0)

        self.model.save_weights('./vae/weights.h5')

    def get_embedding(self, sess, observation, hard=False):
        if hard:
            # TODO at test time it shuold be hard!
            logits = self.encoder.predict(observation)[0]
            logits = np.reshape(logits, (-1, self.N, self.M))
            raise Exception
            prob = []
            for l in range(logits.shape[0]):
                prob.append(utils.softmax(logits[l], 0))

            prob = np.array(prob)
            print(np.sum(prob, 2))
            return

        else:
            return self.encoder.predict(observation)[1]

    def normalize_observation(self, observation):
        # TODO: Probably reshape
        obs = resize(observation, (len(observation), 64, 64, 3), anti_aliasing=True)
        return obs.astype('float32') / 255.

    def show_pred(self, title, data, logits, gumbel, pred):
        max = len(pred)

        for i in range(max):
            # fig, [[ax00, ax01], [ax10, ax11]] = plt.subplots(2, 2, figsize=(10, 5))
            fig = plt.figure(1, figsize=(10, 5))
            print('\r' + title + ' - Plotting vae pred: {}/{}'.format(i, max), end='')
            n = np.random.randint(len(data))

            plt.subplot(221)
            plt.imshow(data[n])
            plt.title(title + 'i' + str(i))
            plt.subplot(222)
            plt.imshow(pred[n])
            plt.title('pred')
            # ax00.imshow(data[n])
            # ax00.set_title("data")
            # ax01.imshow(pred[n])
            # ax01.set_title("pred")

            plt.subplot(223)
            plt.imshow(np.reshape(logits[n], (self.N, self.M)).T)
            plt.title('logits')
            plt.colorbar()
            plt.subplot(224)
            plt.imshow(np.reshape(gumbel[n], (self.N, self.M)).T)
            plt.title('gumbel')

            # im = ax10.imshow(np.reshape(logits[n], (self.N, self.M)).T)
            # ax00.set_title("logits")
            # cax = fig.add_axes()
            # fig.colorbar(im, cax=cax)
            # ax11.imshow(np.reshape(gumbel[n], (self.N, self.M)).T)
            # ax00.set_title("gumbel")

            plt.tight_layout()
            plt.savefig('./videos/CarRacing-'+title+'-'+str(i)+'.png', bbox_inches='tight')
            plt.close()
        print()

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
    vae = Network()
    try:
      vae.set_weights('./vae/weights.h5')
      print('Weights usccessfully loaded')
    except:
      print("No weights found in ./vae/weights.h5 exists")

    start_batch = 0
    max_batch = 14
    test_batch = 15
    # TODO: ^: Don't hardcode like this

    try:
        print("Loading data ...")
        data_list = []
        for batch_num in range(start_batch, max_batch + 1):
            batch_to_load = '../data/obs_data_VAE_' + str(batch_num) + '.npy'
            try:
                data = np.load(batch_to_load)
                # data = data[100:150]  # <-- ONLY FOR TESTING

                data = resize(data, (len(data), 64, 64, 3), anti_aliasing=True).astype(np.float32)
                print('\tFound batch at {}...current data size = {} episodes'.format(
                    batch_to_load, len(data)))
                data_list.append(data)
            except:
                print('\tUnable to load:', batch_to_load)
        data = np.concatenate(data_list, axis=0)

        batch_to_load = '../data/obs_data_VAE_' + str(test_batch) + '.npy'
        test_data = np.load(batch_to_load)
        test_data = resize(test_data, (len(test_data), 64, 64, 3), anti_aliasing=True)
        print('\tFound batch at {}...current data size = {} episodes'.format(
            batch_to_load, len(test_data)))
        print()


        print("Begin Training ...")
        for epoch in range(vae.epochs):
            # for batch_num in range(len(data_list)):
            K.set_value(vae.tau,np.max([vae.tau_min,
                    vae.tau0 * np.exp(-vae.anneal_rate * K.get_value(vae.global_step))]))
            print('EPOCH ' + str(epoch), '- step', K.get_value(vae.global_step),
                  "- tau {:5.3f}".format(K.get_value(vae.tau)),
                  '- Validationscore', vae.model.evaluate(test_data, test_data, verbose=0))

            vae.train(data)
            vae.model.save_weights('./vae/weights.h5')
            K.set_value(vae.global_step, K.get_value(vae.global_step) + 1)
            print()

            ## Testing
            if epoch % 1 == 0:# and epoch > 0:
                n = np.random.randint(96, len(test_data)-96)
                test_data_reduced = test_data[n:n + 96]
                print('\nTesting')
                print("\tValidation score", vae.model.evaluate(test_data, test_data, verbose=0))

                logits, gumbel, pred = vae.tester.predict(test_data_reduced, verbose=0)
                title = "e{}".format(epoch)
                vae.show_pred(title, test_data_reduced, logits, gumbel, pred)
                print('')

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")

    except Exception as e:
        print("Exception: {}".format(e))


def load_vae():
    # sess = tf.InteractiveSession()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=config, graph=graph)
        network = Network()

        network.set_weights('./vae/weights.h5')
        try:
            network.set_weights('./vae/weights.h5')
        except:
            print(currentdir)
            raise ImportError("Could not restore saved model")

        return sess, network


if __name__ == '__main__':
    train_vae()
