import unittest
import koopmanlib.solver as solver
from koopmanlib.dictionary import PsiNN, DicRBF
import numpy as np


class TestSolver(unittest.TestCase):
    """Test Suite for ``solver.py``
    """

    def test_general_solver(self):
        """Test Koopman general solver
        """
        # Generate data
        data_x_train = np.random.uniform(size=(15, 2))
        data_y_train = np.random.uniform(size=(15, 2))
        data_train = [data_x_train, data_y_train]

        # Build model

        # If we use rbf as a testing case,
        # the number of samples needs to be larger than the number of basis
        # functions.
        rbf_basis_func = DicRBF(rbf_number=10, regularizer=1e-4)
        rbf_basis_func.build(data_x_train)
        rbf_solver = solver.KoopmanGeneralSolver(dic=rbf_basis_func,
                                                 target_dim=2,
                                                 reg=0.0)
        rbf_solver.build(data_train)

        # Test the shape of modes
        self.assertEqual(rbf_solver.modes.shape, (2, 13), 'Should be (2, 13)')

    def test_dl_solver(self):
        """Test Koopman dictionary learning solver
        """
        # Generate data
        data_x_train = np.random.uniform(size=(15, 2))
        data_y_train = np.random.uniform(size=(15, 2))
        data_train = [data_x_train, data_y_train]

        data_x_valid = np.random.uniform(size=(5, 2))
        data_y_valid = np.random.uniform(size=(5, 2))
        data_valid = [data_x_valid, data_y_valid]

        # Build model
        basis_function = PsiNN(layer_sizes=[100, 100, 100], n_psi_train=22)
        dl_solver = solver.KoopmanDLSolver(dic=basis_function,
                                           target_dim=2,
                                           reg=0.1)

        dl_solver.build(data_train=data_train,
                        data_valid=data_valid,
                        epochs=3,
                        batch_size=5,
                        lr=1e-4,
                        log_interval=20,
                        lr_decay_factor=0.8)

        # Test the shape of modes
        self.assertEqual(dl_solver.modes.shape, (2, 25), 'Should be (2, 25)')


if __name__ == '__main__':
    unittest.main()
