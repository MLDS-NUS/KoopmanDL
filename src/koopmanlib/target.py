import tensorflow as tf
import numpy as np


class AbstractODETarget(object):
    def __init__(
            self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.25,
            dim=2,
            seed=None):
        self.n_init = n_init
        self.traj_len = traj_len
        self.n_data = n_init * traj_len
        self.dim = dim
        self.dt = dt
        self.t_step = t_step
        self.n_step = int(t_step / dt)
        self.seed = seed

    def generate_init_data(self):
        data_x = []
        if self.seed is not None:
            np.random.seed(self.seed)
            x0 = np.random.uniform(
                size=(
                    self.n_init,
                    self.dim),
                low=self.x_min,
                high=self.x_max)
        else:
            x0 = np.random.uniform(
                size=(
                    self.n_init,
                    self.dim),
                low=self.x_min,
                high=self.x_max)

        data_x.append(x0)
        for t in range(self.traj_len - 1):
            data_x.append(self.euler(data_x[t]))
        data_x = tf.reshape(
            tf.transpose(
                tf.convert_to_tensor(data_x), [
                    1, 0, 2]), shape=(
                self.n_init * self.traj_len, self.dim))
        return np.asarray(data_x)

    def generate_next_data(self, data_x):
        data_y = self.euler(data_x)
        return data_y

    def rhs(self):
        """RHS Function
        :return: The rhs of one specific ODE
        """
        return NotImplementedError

    def euler(self, x):
        """ODE Solver

        :param x: variable
        :type x: vector (float)
        :return: ODE Solution at t_step after iterating the Euler method n_step times
        :rtype: vector with the same shape as the variable x (float)
        """
        for _ in range(self.n_step):
            x = x + self.dt * self.rhs(x)
        return x


class DuffingOscillator(AbstractODETarget):
    """Duffing equation based on the notation in

    (https://en.wikipedia.org/wiki/Duffing_equation)
    """

    def __init__(
            self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.25,
            dim=2,
            seed=None,
            delta=0.5,
            beta=-1.0,
            alpha=1.0):
        super(
            DuffingOscillator,
            self).__init__(
            n_init,
            traj_len,
            dt,
            t_step,
            dim,
            seed)
        self.delta = delta
        self.beta = beta
        self.alpha = alpha
        self.x_min = -2
        self.x_max = 2

    def rhs(self, x):
        x1 = tf.reshape(x[:, 0], shape=(x.shape[0], 1))
        x2 = tf.reshape(x[:, 1], shape=(x.shape[0], 1))
        f1 = x2
        f2 = -self.delta * x2 - x1 * (self.beta + self.alpha * x1**2)
        return tf.concat([f1, f2], axis=-1)


class VanderPolOscillator(AbstractODETarget):
    """Van der Pol Oscillator based on the notation in

    (https://en.wikipedia.org/wiki/Van_der_Pol_oscillator)
    """

    def __init__(
            self,
            n_init,
            traj_len,
            dt=1e-3,
            t_step=0.25,
            dim=2,
            seed=None,
            alpha=2.0):
        super(
            VanderPolOscillator,
            self).__init__(
            n_init,
            traj_len,
            dt,
            t_step,
            dim,
            seed)
        self.alpha = alpha
        self.x_min = -5
        self.x_max = 5

    def rhs(self, x):
        x1 = tf.reshape(x[:, 0], shape=(x.shape[0], 1))
        x2 = tf.reshape(x[:, 1], shape=(x.shape[0], 1))
        f1 = x2
        f2 = self.alpha * (1.0 - x1**2) * x2 - x1
        return tf.concat([f1, f2], axis=-1)
