import tensorflow.compat.v1 as tf
tf.config.experimental.set_visible_devices([], 'GPU')
tf.disable_v2_behavior()
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, 'C:\\Users\JaehyeonJoo\PycharmProjects\deeplearning\SKT')
from compression_net import CompressionNet
from estimation_net import EstimationNet
from gmm import GMM

class DAGMM:
    """ Deep Autoencoding Gaussian Mixture Model.
    This implementation is based on the paper:
    Bo Zong+ (2018) Deep Autoencoding Gaussian Mixture Model
    for Unsupervised Anomaly Detection, ICLR 2018
    (this is UNOFFICIAL implementation)
    """

    MODEL_FILENAME = "DAGMM_model"
    SCALER_FILENAME = "DAGMM_scaler"

    def __init__(self, comp_hiddens, comp_activation,
            est_hiddens, est_activation, est_dropout_ratio=0.5,
            minibatch_size=1024, epoch_size=100,
            learning_rate=0.0001, lambda1=0.1, lambda2=0.0001,
            normalize=True, random_seed=123):
        """
        Parameters
        ----------
        comp_hiddens : list of int
            sizes of hidden layers of compression network
            For example, if the sizes are [n1, n2],
            structure of compression network is:
            input_size -> n1 -> n2 -> n1 -> input_sizes
        comp_activation : function
            activation function of compression network
        est_hiddens : list of int
            sizes of hidden layers of estimation network.
            The last element of this list is assigned as n_comp.
            For example, if the sizes are [n1, n2],
            structure of estimation network is:
            input_size -> n1 -> n2 (= n_comp)
        est_activation : function
            activation function of estimation network
        est_dropout_ratio : float (optional)
            dropout ratio of estimation network applied during training
            if 0 or None, dropout is not applied.
        minibatch_size: int (optional)
            mini batch size during training
        epoch_size : int (optional)
            epoch size during training
        learning_rate : float (optional)
            learning rate during training
        lambda1 : float (optional)
            a parameter of loss function (for energy term)
        lambda2 : float (optional)
            a parameter of loss function
            (for sum of diagonal elements of covariance)
        normalize : bool (optional)
            specify whether input data need to be normalized.
            by default, input data is normalized.
        random_seed : int (optional)
            random seed used when fit() is called.
        """
        self.comp_net = CompressionNet(comp_hiddens, comp_activation)
        self.est_net = EstimationNet(est_hiddens, est_activation)
        self.est_dropout_ratio = est_dropout_ratio

        n_comp = est_hiddens[-1]
        self.gmm = GMM(n_comp)

        self.minibatch_size = minibatch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.normalize = normalize
        self.scaler = None
        self.seed = random_seed

        self.graph = None
        self.sess = None

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def fit(self, x):
        """ Fit the DAGMM model according to the given data.
        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data.
        """
        n_samples, n_features = x.shape

        if self.normalize:
            self.scaler = scaler = StandardScaler()
            x = scaler.fit_transform(x)

        with tf.Graph().as_default() as graph:
            self.graph = graph
            tf.set_random_seed(self.seed)
            np.random.seed(seed=self.seed)

            # Create Placeholder
            self.input = input = tf.placeholder(
                dtype=tf.float32, shape=[None, n_features])
            self.drop = drop = tf.placeholder(dtype=tf.float32, shape=[])

            # Build graph
            z, x_dash  = self.comp_net.inference(input)
            gamma = self.est_net.inference(z, drop)
            self.gmm.fit(z, gamma)
            energy = self.gmm.energy(z)

            self.x_dash = x_dash

            # Loss function
            loss = (self.comp_net.reconstruction_error(input, x_dash) +
                self.lambda1 * tf.reduce_mean(energy) +
                self.lambda2 * self.gmm.cov_diag_loss())

            # Minimizer
            minimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            # Number of batch
            n_batch = (n_samples - 1) // self.minibatch_size + 1

            # Create tensorflow session and initilize
            init = tf.global_variables_initializer()

            self.sess = tf.Session(graph=graph)
            self.sess.run(init)

            # Training
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)

            for epoch in range(self.epoch_size):
                for batch in range(n_batch):
                    i_start = batch * self.minibatch_size
                    i_end = (batch + 1) * self.minibatch_size
                    x_batch = x[idx[i_start:i_end]]

                    self.sess.run(minimizer, feed_dict={
                        input:x_batch, drop:self.est_dropout_ratio})

                if (epoch + 1) % 100 == 0:
                    loss_val = self.sess.run(loss, feed_dict={input:x, drop:0})
                    print(" epoch {}/{} : loss = {:.3f}".format(epoch + 1, self.epoch_size, loss_val))

            # Fix GMM parameter
            fix = self.gmm.fix_op()
            self.sess.run(fix, feed_dict={input:x, drop:0})
            self.energy = self.gmm.energy(z)

            tf.add_to_collection("save", self.input)
            tf.add_to_collection("save", self.energy)

            self.saver = tf.train.Saver()

    def predict(self, x):
        """ Calculate anormaly scores (sample energy) on samples in X.
        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Data for which anomaly scores are calculated.
            n_features must be equal to n_features of the fitted data.
        Returns
        -------
        energies : array-like, shape (n_samples)
            Calculated sample energies.
        """
        if self.sess is None:
            raise Exception("Trained model does not exist.")

        if self.normalize:
            x = self.scaler.transform(x)

        energies = self.sess.run(self.energy, feed_dict={self.input:x})
        return energies



import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
'''
y=np.zeros((984,1))

dataset_path_2nd = '2nd_test'

dataset_path = dataset_path_2nd #1st # 2nd 3rd

for dirname, _, filenames in os.walk('2nd_test'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        print(filename)


merged_data = pd.DataFrame()

for filename in os.listdir(dataset_path):
    dataset=pd.read_csv(os.path.join(dataset_path, filename), sep='\t', header=None)
    dataset_mean_abs = np.array(dataset.abs().mean())
    ncols = dataset.shape[1] # Number of channels (one column per channel)
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,ncols))
    dataset_mean_abs.index = [filename]
    merged_data = merged_data.append(dataset_mean_abs)
'''
merged_data = pd.read_csv('C:\\Users\\JaehyeonJoo\\PycharmProjects\\deeplearning\\SKT\\db2_merged_dataset_BearingTest_2.csv')
merged_data = merged_data.iloc[:,0:4]



model_dagmm = DAGMM(
    comp_hiddens=[3,2], comp_activation=tf.nn.tanh,
    est_hiddens=[3,2], est_activation=tf.nn.tanh, est_dropout_ratio=0.25,
    epoch_size=5000, minibatch_size=24, random_seed=0
)
training = merged_data[0:400]
model_dagmm.fit(training)
a = model_dagmm.predict(merged_data)
b = model_dagmm.predict(training)
from matplotlib import pyplot as plt


plt.plot([x for x in range(984)], a)
plt.axhline(y=b.mean() + 5 * b.std(), color='r', linewidth=1)
plt.title('Anomaly Detection')
plt.xlabel('Time')
plt.ylabel('Anomaly Score')
plt.show()

result = merged_data[a >b.mean() + 5 * b.std()]