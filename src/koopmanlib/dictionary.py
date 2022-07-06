from scipy.cluster.vq import kmeans
import scipy
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.layers import Input, Add, Multiply, Lambda, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


class AbstractDictionary(object):
    def generate_B(self, inputs):
        target_dim = inputs.shape[-1]
        self.basis_func_number = self.n_dic_customized + target_dim + 1
        # Form B matrix
        self.B = np.zeros((self.basis_func_number, target_dim))
        for i in range(0, target_dim):
            self.B[i + 1][i] = 1
        return self.B


class DicNN(Layer):
    """Trainable dictionaries

    """

    def __init__(self, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
        super(DicNN, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.input_layer = Dense(self.layer_sizes[0], use_bias=False)
        self.hidden_layers = [Dense(s, activation='tanh') for s in layer_sizes]
        self.output_layer = Dense(n_psi_train)
        self.n_psi_train = n_psi_train

    def call(self, inputs):
        psi_x_train = self.input_layer(inputs)
        for layer in self.hidden_layers:
            psi_x_train = psi_x_train + layer(psi_x_train)
        outputs = self.output_layer(psi_x_train)
        return outputs

    def get_config(self):
        config = super(DicNN, self).get_config()
        config.update({
            'layer_sizes': self.layer_sizes,
            'n_psi_train': self.n_psi_train
        })
        return config


class PsiNN(Layer, AbstractDictionary):
    """Concatenate constant, data and trainable dictionaries together as [1, data, DicNN]

    """

    def __init__(
            self,
            dic_trainable=DicNN,
            layer_sizes=[
                64,
                64],
            n_psi_train=22,
            **kwargs):
        super(PsiNN, self).__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.dic_trainable = dic_trainable
        self.n_dic_customized = n_psi_train
        self.dicNN = self.dic_trainable(
            layer_sizes=self.layer_sizes,
            n_psi_train=self.n_dic_customized)

    def call(self, inputs):
        constant = tf.ones_like(tf.slice(inputs, [0, 0], [-1, 1]))
        psi_x_train = self.dicNN(inputs)
        outputs = Concatenate()([constant, inputs, psi_x_train])
        return outputs

    def get_config(self):
        config = super(PsiNN, self).get_config()
        config.update({
            'dic_trainable': self.dic_trainable,
            'layer_sizes': self.layer_sizes,
            'n_psi_train': self.n_dic_customized
        })
        return config


class DicRBF(AbstractDictionary):
    """
    RBF based on notations in
    (https://en.wikipedia.org/wiki/Radial_basis_function)
    """

    def __init__(self, rbf_number=100, regularizer=1e-4):
        self.n_dic_customized = rbf_number
        self.regularizer = regularizer

    def build(self, data):
        self.centers, residual = kmeans(data, self.n_dic_customized)

    def call(self, data):
        rbfs = []
        for n in range(self.centers.shape[0]):
            r = scipy.spatial.distance.cdist(
                data, np.matrix(self.centers[n, :]))
            rbf = scipy.special.xlogy(r**2, r + self.regularizer)
            rbfs.append(rbf)

        rbfs = tf.transpose(tf.squeeze(rbfs))
        rbfs = tf.reshape(rbfs, shape=(data.shape[0], -1))

        ones = tf.ones(shape=(rbfs.shape[0], 1), dtype='float64')
        results = tf.concat([ones, data, rbfs], axis=-1)
        return results
