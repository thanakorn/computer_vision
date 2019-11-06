import unittest
import numpy as np
import scipy
from scipy import ndimage
import sys

sys.path.append('/home/tpanyapiang/git/MSc/computer_vision/hybrid_images/src')

from MyConvolution import convolve, zero_pad, remove_pad
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
        expected = ndimage.convolve(img, kernel, mode='constant', cval=0)
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
        expected = ndimage.convolve(img, kernel, mode='constant', cval=0)
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
        expected = ndimage.convolve(img, kernel, mode='constant', cval=0)
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
        expected = ndimage.convolve(img, kernel, mode='constant', cval=0)
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
        expected = ndimage.convolve(img, kernel, mode='constant', cval=0)
        result = convolve(img, kernel)
        assert_array_equal(result, expected)

    def test_rgb_img(self):
        img = np.ndarray(
            shape = (3,3,3), 
            dtype = int,
            buffer = np.array([[
                [[48, 48, 48],
                [41, 41, 41],
                [43, 43, 43]],
                [[42, 42, 42],
                [47, 47, 47],
                [44, 44, 44]],
                [[45, 45, 45],
                [46, 46, 46],
                [43, 43, 43]]
            ]]))
        kernel = np.ndarray(
            shape = (3,3),
            dtype = int,
            buffer = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]))
        expected = ndimage.convolve(np.ndarray(
            shape = (3,3), 
            dtype = int,
            buffer = np.array([
                [48, 41, 43],
                [42, 47, 44],
                [45, 46, 43]
            ])), kernel, mode='constant', cval=0)
        result = convolve(img, kernel)
        np.testing.assert_array_equal(result[:,:,0], expected)
        np.testing.assert_array_equal(result[:,:,1], expected)
        np.testing.assert_array_equal(result[:,:,2], expected)


    def test_pad_img(sefl):
        img = np.array([[1,1,1], [1,1,1], [1,1,1]])
        expected_img = np.array([[0,0,0,0,0], [0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0], [0,0,0,0,0]])
        np.testing.assert_array_equal(zero_pad(img, 1, 1), expected_img)

    def test_pad_img_asym(self):
        img = np.array([[1,1,1], [1,1,1], [1,1,1]])
        num_row_pad = 1
        num_col_pad = 2
        expected_img = np.array([[0,0,0,0,0,0,0], [0,0,1,1,1,0,0], [0,0,1,1,1,0,0], [0,0,1,1,1,0,0], [0,0,0,0,0,0,0]])
        np.testing.assert_array_equal(zero_pad(img, num_row_pad, num_col_pad), expected_img)

    def test_pad_3d_img(self):
        img = np.ndarray(shape = (3,3,3))
        img.fill(1)
        expected_img = np.array([[0,0,0,0,0], [0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0], [0,0,0,0,0]])
        padded_img = zero_pad(img, 1, 1)
        np.testing.assert_array_equal(padded_img[:,:,0], expected_img)
        np.testing.assert_array_equal(padded_img[:,:,1], expected_img)
        np.testing.assert_array_equal(padded_img[:,:,2], expected_img)

    def test_remove_pad(self):
        img = np.array([[0,0,0,0,0], [0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0], [0,0,0,0,0]])
        expected_img = np.array([[1,1,1], [1,1,1], [1,1,1]])
        np.testing.assert_array_equal(remove_pad(img, 1, 1), expected_img)

    def test_remove_pad_asym(self):
        img = np.array([[0,0,0,0,0,0,0], [0,0,1,1,1,0,0], [0,0,1,1,1,0,0], [0,0,1,1,1,0,0], [0,0,0,0,0,0,0]])
        num_row_pad = 1
        num_col_pad = 2
        expected_img =  np.array([[1,1,1], [1,1,1], [1,1,1]])
        np.testing.assert_array_equal(remove_pad(img, num_row_pad, num_col_pad), expected_img)

    def test_remove_pad_3d_img(self):
        img = np.ndarray(shape = (5,5,3))
        img.fill(1)
        expected_img = np.array([[1,1,1], [1,1,1], [1,1,1]])
        unpadded_img = remove_pad(img, 1, 1)
        np.testing.assert_array_equal(unpadded_img[:,:,0], expected_img)
        np.testing.assert_array_equal(unpadded_img[:,:,1], expected_img)
        np.testing.assert_array_equal(unpadded_img[:,:,2], expected_img)

if __name__ == '__main__':
    unittest.main()