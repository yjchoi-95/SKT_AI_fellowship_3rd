import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model


from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split

import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
import pandas as pd
tf.config.experimental.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class Ganomaly:
    def __init__(self, latent_dim=100, input_shape=(28, 28, 1), batch_size=128, epochs=40000, anomaly_class=2, seed=0):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.anomaly_class = anomaly_class
        self.seed = seed

    def get_data(self):
        print('get train and tst data...')
        data = pd.read_csv('C:\\Users\\JaehyeonJoo\\PycharmProjects\\deeplearning\\SKT\\db2_merged_dataset_BearingTest_2.csv')
        data = data.iloc[:, 0:4]
        X_train = data.iloc[:400]
        #X_test = data.iloc[400:]
        X_test = data
        # test_y = X_test.iloc[:, 0]
        # test_y.loc[0:400] = 0
        # test_y.loc[400:] = 2
        # test_y = list(test_y)
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        self.X_train,  self.X_test = X_train,  X_test
        #self.X_train, self.X_test, self.test_y = X_train, X_test, test_y

        print('train samples: %d, test samples: %d.' % (len(self.X_train), len(self.X_test)))
        print('[OK]')

    def basic_encoder(self):
        modelE = Sequential()
        modelE.add(Dense(3, input_shape=self.input_shape))
        modelE.add(Activation("relu"))
        modelE.add(BatchNormalization(momentum=0.8))
        modelE.add(Dense(self.latent_dim))
        return modelE

    # Encoder 1
    def make_encoder_1(self):
        enc_model_1 = self.basic_encoder()
        img = Input(shape=self.input_shape)
        z = enc_model_1(img)
        encoder1 = Model(img, z)
        return encoder1

    # Generator
    def make_generator(self):
        modelG = Sequential()
        modelG.add(Dense(3, input_dim=self.latent_dim))
        modelG.add(LeakyReLU(alpha=0.2))
        modelG.add(BatchNormalization(momentum=0.8))
        modelG.add(Dense(4, activation='tanh'))

        z = Input(shape=(self.latent_dim,))
        gen_img = modelG(z)
        generator = Model(z, gen_img)
        return generator

    # Encoder 2
    def make_encoder_2(self):
        enc_model_2 = self.basic_encoder()
        img = Input(shape=self.input_shape)
        z = enc_model_2(img)
        encoder2 = Model(img, z)
        return encoder2

    # Discriminator
    def make_discriminator(self):
        modelD = Sequential()
        #modelD.add(Dense(3, input_shape=self.input_shape))
        modelD.add(Dense(2, input_shape=self.input_shape))
        modelD.add(LeakyReLU(alpha=0.2))
        modelD.add(Dropout(0.25))
        #modelD.add(Dense(2))
        #modelD.add(BatchNormalization(momentum=0.8))
        #modelD.add(LeakyReLU(alpha=0.2))
        #modelD.add(Dropout(0.25))
        modelD.add(Dense(1, activation='sigmoid'))

        return modelD

    def make_components(self):
        print('make components...')

        self.optimizer = tf.keras.optimizers.Adam(0.0001, 0.5)

        self.discriminator = self.make_discriminator()
        self.discriminator.trainable = True

        # Build and compile the discriminator
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=self.optimizer, metrics=['accuracy'])

        self.discriminator.trainable = False

        # First image encoding
        self.img = Input(shape=self.input_shape)

        self.encoder1 = self.make_encoder_1()
        self.z = self.encoder1(self.img)

        self.generator = self.make_generator()
        self.img_ = self.generator(self.z)

        self.encoder2 = self.make_encoder_2()
        self.z_ = self.encoder2(self.img_)

        # The discriminator takes generated images as input and determines if real or fake
        self.real_or_fake = self.discriminator(self.img_)

        self.ganomaly_model = Model(self.img, [self.real_or_fake, self.img_, self.z_])
        self.ganomaly_model.compile(loss=['binary_crossentropy', 'mean_absolute_error', 'mean_squared_error'],
                                    optimizer=self.optimizer)

        self.g_loss_list = []
        self.d_loss_list = []

        print('[OK]')

    def train(self):
        self.get_data()
        tf.random.set_seed(self.seed)
        self.make_components()
        # Adversarial ground truths
        real = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        print('start train...')
        for epoch in range(self.epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images and encode/decode/encode
            idx = np.random.randint(0, self.X_train.shape[0], self.batch_size)
            imgs = self.X_train[idx]
            z = self.encoder1.predict(imgs)
            imgs_ = self.generator.predict(z)
            z_ = self.encoder2(imgs_)
            # Train the discriminator (imgs are real, imgs_ are fake)
            d_loss_real = self.discriminator.train_on_batch(imgs, real)
            d_loss_fake = self.discriminator.train_on_batch(imgs_, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.ganomaly_model.train_on_batch(imgs, [real, imgs, z])

            # Plot the progress
            print("epoch: %d [D loss: %f, acc: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))
            self.g_loss_list.append(g_loss)
            self.d_loss_list.append(d_loss)
        # save loss img
        plt.plot(np.asarray(self.g_loss_list)[:, 0], label='G loss')
        plt.plot(np.asarray(self.d_loss_list)[:, 0], label='D loss')
        plt.plot(np.asarray(self.d_loss_list)[:, 1], label='D accuracy')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.savefig("loss_%d.png" % self.anomaly_class, bbox_inches='tight', pad_inches=1)
        plt.close()

        print('[OK]')

    # def eval(self):
    #     print('evaluate on test data...')
    #     print('generate z1...')
    #     z1_gen_ema = self.encoder1.predict(self.X_test)
    #     print('generate fake images...')
    #     reconstruct_ema = self.generator.predict(z1_gen_ema)
    #     print('generate z2...')
    #     z2_gen_ema = self.encoder2.predict(reconstruct_ema)
    #
    #     val_list = []
    #     for i in range(0, len(self.X_test)):
    #         val_list.append(np.mean(np.square(z1_gen_ema[i] - z2_gen_ema[i])))
    #
    #     anomaly_labels = np.zeros(len(val_list))
    #     for i, label in enumerate(self.Y_test):
    #         if label == self.anomaly_class:
    #             anomaly_labels[i] = 1
    #
    #     val_arr = np.asarray(val_list)
    #     val_probs = val_arr / max(val_arr)
    #     # print('val_arr:')
    #     # print(val_arr[:50])
    #     # print('val_probs:')
    #     # print(val_probs[:50])
    #     # print('anomaly labels:')
    #     # print(anomaly_labels[:50])
    #
    #     # preview val
    #     idx = np.random.randint(0, len(val_arr), 100)
    #     print('-val_arr- -val_probs- -anomaly_labels-')
    #     print('--------------------------------------')
    #     for i in idx:
    #         print(val_arr[i], val_probs[i], anomaly_labels[i])
    #
    #     fp_rate, tp_rate, thresholds = roc_curve(anomaly_labels, val_probs)
    #     auc_rate = auc(fp_rate, tp_rate)
    #     roc_auc = roc_auc_score(anomaly_labels, val_probs)
    #     prauc = average_precision_score(anomaly_labels, val_probs)
    #     # roc_auc_scores.append(roc_auc)
    #     # prauc_scores.append(prauc)
    #     print("fp_rate:", fp_rate)
    #     print("tp_rate:", tp_rate)
    #     print("auc_rate:", auc_rate)
    #     print("threshold:", thresholds)
    #     print("ROC AUC SCORE FOR [%d](anomaly class): %f" % (self.anomaly_class, roc_auc))
    #     print("PRAUC SCORE FOR [%d](anomaly class): %f" % (self.anomaly_class, prauc))
    #     print('[OK]')
    #     return val_arr



if __name__ == '__main__':

    model = Ganomaly(latent_dim=2, input_shape=(4,),batch_size=400, epochs=2500, anomaly_class=2, seed=4)
    model.train()
    #val_arr = model.eval()
    z1_gen_ema = model.encoder1.predict(model.X_test)
    reconstruct_ema = model.generator.predict(z1_gen_ema)
    z2_gen_ema = model.encoder2.predict(reconstruct_ema)

    val_list = []
    for i in range(0, len(model.X_test)):
        val_list.append(np.mean(np.square(z1_gen_ema[i] - z2_gen_ema[i])))
    val_arr = np.asarray(val_list)

    b = val_arr[0:400]
    plt.plot([x for x in range(984)], val_arr)
    plt.axhline(y=b.mean() + 5 * b.std(), color='r', linewidth=1)
    plt.title('Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Anomaly Score')
    plt.show()

    data = pd.DataFrame(val_arr)
    result = data[val_arr >b.mean() + 5 * b.std()]