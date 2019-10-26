#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from hybrid_images.src.MyConvolution import convolve

#%% RGB to Gray
def to_gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.144])

# %% Load image
img_file = '/home/tpanyapiang/git/MSc/computer_vision/hybrid_images/sample_images/bicycle.bmp'
raw_img = plt.imread(img_file)
gray_img = to_gray(raw_img)
plt.imshow(gray_img, cmap = 'gray')
# %% Average operator
avg_operator = float(1/9) * np.ndarray(
                shape = (3,3),
                buffer = np.array([
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]
                ]))
avg_img = convolve(gray_img, avg_operator)
plt.imshow(avg_img, cmap = 'gray')

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
# %% Gaussian operator
gaussian_operator = float(1/16) * np.ndarray(
                     shape = (3,3),
                     buffer = np.array([
                        [1.0, 2.0, 1.0],
                        [2.0, 4.0, 2.0],
                        [1.0, 2.0, 1.0]
                    ]))
gaussian_img = convolve(gray_img, gaussian_operator)
plt.imshow(gaussian_img, cmap = 'gray')


# %%
