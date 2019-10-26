import unittest
import numpy as np
import sys

sys.path.append('/home/tpanyapiang/git/MSc/computer_vision/hybrid_images/src')

from MyConvolution import convolve
from numpy.testing import assert_array_equal

class TestMyConvolution(unittest.TestCase):
    def test_img_eq_kernel(self):
        img = np.ndarray(
            shape = (3,3), 
            dtype = int,
            buffer = np.array([
                [48, 41, 43],
                [42, 47, 44],
                [45, 46, 43]
            ]))
        kernel = np.ndarray(
            shape = (3,3),
            dtype = int,
            buffer = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]))
        expected = np.ndarray(
            shape = (3,3), 
            dtype = int,
            buffer = np.array([
                [0, 0, 0],
                [0, 7, 0],
                [0, 0, 0]
            ]))
        result = convolve(img, kernel)
        assert_array_equal(result, expected)

    def test_img_bigger_than_kernel(self):
        img = np.ndarray(
            shape = (5,5), 
            dtype = int,
            buffer = np.array([
                [3, 3, 2, 1, 0],
                [0, 0, 1, 3, 1],
                [3, 1, 2, 2, 3],
                [2, 0, 0, 2, 2],
                [2, 0, 0, 0, 1],
            ]))
        kernel = np.ndarray(
            shape = (3,3),
            dtype = int,
            buffer = np.array([
                [0, 1, 2],
                [2, 2, 0],
                [0, 1, 2]
            ]))
        expected = np.ndarray(
            shape = (5,5), 
            dtype = int,
            buffer = np.array([
                [0, 0, 0, 0, 0],
                [0, 12, 12, 17, 0],
                [0, 10, 17, 19, 0],
                [0, 9, 6, 14, 0],
                [0, 0, 0, 0, 0],
            ]))
        result = convolve(img, kernel)
        assert_array_equal(result, expected)

    def test_asym_img(self):
        img = np.ndarray(
        shape = (5,4), 
        dtype = int,
        buffer = np.array([
            [3, 3, 2, 1],
            [0, 0, 1, 3],
            [3, 1, 2, 2],
            [2, 0, 0, 2],
            [2, 0, 0, 0],
        ]))
        kernel = np.ndarray(
            shape = (3,3),
            dtype = int,
            buffer = np.array([
                [0, 1, 2],
                [2, 2, 0],
                [0, 1, 2]
            ]))
        expected = np.ndarray(
            shape = (5,4), 
            dtype = int,
            buffer = np.array([
                [0, 0, 0, 0],
                [0, 12, 12, 0],
                [0, 10, 17, 0],
                [0, 9, 6, 0],
                [0, 0, 0, 0],
            ]))
        result = convolve(img, kernel)
        assert_array_equal(result, expected)

    def test_asym_kernel(self):
        img = np.ndarray(
            shape = (5,5), 
            dtype = int,
            buffer = np.array([
                [3, 3, 2, 1, 0],
                [0, 0, 1, 3, 1],
                [3, 1, 2, 2, 3],
                [2, 0, 0, 2, 2],
                [2, 0, 0, 0, 1],
            ]))
        kernel = np.ndarray(
            shape = (3,5),
            dtype = int,
            buffer = np.array([
                [0, 1, 2, 2, 1],
                [2, 2, 0, 1, 1],
                [0, 1, 2, 1, 1]
            ]))
        expected = np.ndarray(
            shape = (5,5), 
            dtype = int,
            buffer = np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 23, 0, 0],
                [0, 0, 26, 0, 0],
                [0, 0, 21, 0, 0],
                [0, 0, 0, 0, 0],
            ]))
        result = convolve(img, kernel)
        assert_array_equal(result, expected)
    
    def test_asym_img_and_kernel(self):
        img = np.ndarray(
            shape = (5,6), 
            dtype = int,
            buffer = np.array([
                [3, 3, 2, 1, 0, 1],
                [0, 0, 1, 3, 1, 1],
                [3, 1, 2, 2, 3, 2],
                [2, 0, 0, 2, 2, 2],
                [2, 0, 0, 0, 1, 2],
            ]))
        kernel = np.ndarray(
            shape = (3,5),
            dtype = int,
            buffer = np.array([
                [0, 1, 2, 2, 1],
                [2, 2, 0, 1, 1],
                [0, 1, 2, 1, 1]
            ]))
        expected = np.ndarray(
            shape = (5,6), 
            dtype = int,
            buffer = np.array([
                [0, 0, 0, 0, 0,0],
                [0, 0, 23, 20, 0, 0],
                [0, 0, 26, 29, 0, 0],
                [0, 0, 21, 21, 0, 0],
                [0, 0, 0, 0, 0,0],
            ]))
        result = convolve(img, kernel)
        assert_array_equal(result, expected)    

if __name__ == '__main__':
    unittest.main()