#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import scipy
import sys
import cv2

sys.path.append('/home/tpanyapiang/git/MSc/computer_vision/hybrid_images/src')

from hybrid_images.src.MyConvolution import convolve
from hybrid_images.src.MyHybridImages import myHybridImages, makeGaussianKernel, zero_pad
%matplotlib inline

#%% RGB to Gray
def to_gray_img(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.144])

# %% Load image
img_file = '/home/tpanyapiang/git/MSc/computer_vision/hybrid_images/sample_images/bicycle.bmp'
raw_img = mimg.imread(img_file)
gray_img = to_gray_img(raw_img)
plt.figure()
plt.imshow(raw_img)
plt.figure()
plt.imshow(gray_img, cmap = 'gray')
# %% Average operator
avg_operator = float(1/9) * np.ndarray(
                shape = (3,3),
                buffer = np.array([
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                ]),
                dtype = int)
avg_img = convolve(raw_img, avg_operator)
plt.figure()
plt.imshow(np.trunc(avg_img).astype(int))

gray_avg_img = convolve(gray_img, avg_operator)
plt.figure()
plt.imshow(np.trunc(gray_avg_img).astype(int), cmap = 'gray')

# %% Gaussian operator
gaussian_operator = makeGaussianKernel(0.5)
gaussian_img = convolve(raw_img, gaussian_operator)
gaussian_gray_img = convolve(gray_img, gaussian_operator)

# over = np.where(gaussian_img > 255)
# print(over)
# print(under)

plt.figure()
plt.imshow(np.trunc(gaussian_img).astype(int))
plt.figure()
plt.imshow(gaussian_gray_img, cmap = 'gray')

# %% Sobel operator
sobel_horizontal_operator = np.ndarray(
                            shape = (3,3),
                            dtype = int,
                            buffer = np.array([
                                [1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]
                            ]))
sobel_vertical_operator = np.ndarray(
                shape = (3,3),
                dtype = int,
                buffer = np.array([
                    [1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]
                ]))
sobel_horizontal_img = convolve(gray_img, sobel_horizontal_operator)
sobel_vertical_img = convolve(gray_img, sobel_vertical_operator)
plt.imshow(sobel_horizontal_img + sobel_vertical_img, cmap = 'gray')

# %% Hybrid image
low_img_file = '/home/tpanyapiang/git/MSc/computer_vision/hybrid_images/sample_images/marilyn.bmp'
high_img_file = '/home/tpanyapiang/git/MSc/computer_vision/hybrid_images/sample_images/einstein.bmp'
low_sigma = 5.0
high_sigma = 5.0

low_img = zero_pad(plt.imread(low_img_file), 20, 20)
high_img = zero_pad(plt.imread(high_img_file), 20, 20)

high_img_filtered = high_img - convolve(high_img, makeGaussianKernel(high_sigma))
plt.figure()
plt.imshow(np.trunc(0.5 + high_img_filtered).astype(int))

hybrid_img = myHybridImages(low_img, low_sigma, high_img, high_sigma)
plt.figure()
plt.imshow(np.trunc(hybrid_img).astype(int))
# %%
