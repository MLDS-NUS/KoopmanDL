import unittest
import koopmanlib.dictionary as dictionary

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
       
if __name__ == '__main__':
    unittest.main()