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
        



if __name__ == '__main__':
    unittest.main()
