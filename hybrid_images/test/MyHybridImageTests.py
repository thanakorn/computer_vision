import unittest
import numpy as np
import sys

sys.path.append('/home/tpanyapiang/git/MSc/computer_vision/hybrid_images/src')

from MyHybridImages import myHybridImages, makeGaussianKernel
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

if __name__ == '__main__':
    unittest.main()