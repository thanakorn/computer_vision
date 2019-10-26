import unittest
import numpy as np
import sys

sys.path.append('/home/tpanyapiang/git/MSc/computer_vision/hybrid_images/src')

from MyConvolution import convolve
from numpy.testing import assert_array_equal

class TestMyConvolution(unittest.TestCase):
    def test_convolve(self):
        img = np.ndarray(shape=(2,2), buffer=np.array([[1,1], [1,1]]), dtype=int)
        kernel = np.ndarray(shape=(2,2), buffer=np.array([[1,1], [1,1]]), dtype=int)
        result = convolve(img, kernel)
        expected = img
        assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()