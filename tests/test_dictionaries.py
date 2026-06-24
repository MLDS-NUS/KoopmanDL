import unittest
import koopmanlib.dictionary as dictionary
import numpy as np


class TestDictionary(unittest.TestCase):
    """Test Suite for ``dictionary.py``
    """

    def test_base_dictionary(self):
        """Test trainable dictionary
        """
        dic_nn = dictionary.DicNN()
        psi_nn = dictionary.PsiNN()
        dic_config = dic_nn.get_config()
        psi_config = psi_nn.get_config()
        data_x = np.random.uniform(size=(15, 2))
        dic_x = dic_nn(data_x)
        psi_x = psi_nn(data_x)
        
        # Test shapes
        self.assertEqual(dic_x.shape, (15, 22), 'Should be (15, 22)')
        self.assertEqual(psi_x.shape, (15, 25), 'Should be (15, 25)')

    def test_hermite_dictionary(self):
        """Test the fixed 2D Hermite dictionary (square truncation)
        """
        dic = dictionary.DicHermite(N=4)
        data_x = np.random.uniform(size=(15, 2))
        psi_x = dic.call(data_x)

        # Square truncation: (N + 1) ** 2 = 25 observables in 2D
        self.assertEqual(psi_x.shape, (15, 25), 'Should be (15, 25)')

        # Constant term and state observables sit at known indices
        np.testing.assert_allclose(psi_x[:, 0], 1.0)
        np.testing.assert_allclose(psi_x[:, 1], data_x[:, 0])
        np.testing.assert_allclose(psi_x[:, 5], data_x[:, 1])

        B = dic.generate_B(data_x)
        self.assertEqual(B.shape, (25, 2))
        self.assertEqual(B[1, 0], 1)
        self.assertEqual(B[5, 1], 1)
        



if __name__ == '__main__':
    unittest.main()
