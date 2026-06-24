from scipy.cluster.vq import kmeans
import scipy
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.layers import Input, Add, Multiply, Lambda, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


class AbstractDictionary(object):
    n_dic_customized: int  # set by subclasses

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


class DicHermite(AbstractDictionary):
    """Fixed dictionary of 2D (physicists') Hermite polynomials.

    This is the EDMD baseline used in Section IV.A. For order ``N`` it returns
    the full tensor-product ("square") truncation
    ``{H_m(x1) * H_n(x2) : 0 <= m, n <= N}``, i.e. ``(N + 1) ** 2`` observables
    (25 for the default ``N = 4``). Note this is *not* a triangular truncation
    by total degree, which would give only 15 terms in 2D.

    The polynomial definitions and the ordering of the basis match the original
    ``dict_hermite`` implementation exactly.
    """

    def __init__(self, N=4):
        assert N <= 5, 'Hermite polynomials are only defined up to order 5 here.'
        self.N = N

    def _hermite_1d(self, x):
        # Physicists' Hermite polynomials H0..H5
        H = [np.ones_like(x),
             x,
             x**2 - 1,
             x**3 - 3 * x,
             x**4 - 6 * x**2 + 3,
             x**5 - 10 * x**3 + 15 * x]
        return np.stack(H[:self.N + 1], axis=-1)  # (n_samples, N + 1)

    def call(self, data):
        data = np.asarray(data, dtype=np.float64)
        Hx = self._hermite_1d(data[:, 0])  # (n_samples, N + 1)
        Hy = self._hermite_1d(data[:, 1])  # (n_samples, N + 1)
        # Tensor product, flattened with ordering index = j * (N + 1) + i,
        # i.e. H_i(x1) * H_j(x2), matching the original dict_hermite.
        psi = (Hy[:, :, None] * Hx[:, None, :]).reshape(data.shape[0], -1)
        return psi

    def generate_B(self, inputs):
        # The state observables sit at fixed indices of the tensor-product
        # ordering: x1 = H1(x1)H0(x2) at index 1, x2 = H0(x1)H1(x2) at index N+1.
        target_dim = inputs.shape[-1]
        self.basis_func_number = (self.N + 1) ** 2
        self.B = np.zeros((self.basis_func_number, target_dim))
        self.B[1, 0] = 1
        self.B[self.N + 1, 1] = 1
        return self.B


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
        # kmeans can return fewer centers than requested (e.g. empty clusters),
        # so sync the basis-function count to the actual number of centers;
        # otherwise call() and generate_B() disagree on the dimension.
        self.n_dic_customized = self.centers.shape[0]

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
