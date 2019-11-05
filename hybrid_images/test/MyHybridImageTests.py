import unittest
import numpy as np
import sys

sys.path.append('/home/tpanyapiang/git/MSc/computer_vision/hybrid_images/src')

from MyHybridImages import myHybridImages, makeGaussianKernel, zero_pad
from numpy.testing import assert_array_equal

class TestMyHubridImages(unittest.TestCase):

    def test_make_gaussian_kernel_size(self):
        self.assertEqual(makeGaussianKernel(0.5).shape, (5, 5))
        self.assertEqual(makeGaussianKernel(0.125).shape, (3, 3))
        self.assertEqual(makeGaussianKernel(1.0).shape, (9, 9))

    def test_make_gaussian_kernel_sum(self):
        np.testing.assert_almost_equal(np.sum(makeGaussianKernel(0.5)), 1.0)
        np.testing.assert_almost_equal(np.sum(makeGaussianKernel(1.0)), 1.0)
        np.testing.assert_almost_equal(np.sum(makeGaussianKernel(5.0)), 1.0)
        np.testing.assert_almost_equal(np.sum(makeGaussianKernel(100.0)), 1.0)

    # def test_make_gaussian_kernel(self):
    #     kernel = makeGaussianKernel(0.5)
    #     expected = np.ndarray(
    #         shape = (5,5),
    #         dtype = float,
    #         buffer = np.array([
    #             [0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
    #             [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
    #             [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
    #             [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
    #             [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]
    #         ]))
    #     np.testing.assert_array_equal(kernel, expected)

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

    def test_pad_3d_img(sefl):
        img = np.ndarray(shape = (3,3,3))
        img.fill(1)
        expected_img = np.array([[0,0,0,0,0], [0,1,1,1,0], [0,1,1,1,0], [0,1,1,1,0], [0,0,0,0,0]])
        padded_img = zero_pad(img, 1, 1)
        np.testing.assert_array_equal(padded_img[:,:,0], expected_img)
        np.testing.assert_array_equal(padded_img[:,:,1], expected_img)
        np.testing.assert_array_equal(padded_img[:,:,2], expected_img)


if __name__ == '__main__':
    unittest.main()