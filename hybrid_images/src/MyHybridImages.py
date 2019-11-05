import math
import numpy as np

from MyConvolution import convolve

def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
    
    :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    
    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage
    :type float
    
    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    
    :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage before subtraction to create the high-pass filtered image
    :type float 
    
    :returns returns the hybrid image created
           by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with 
           a high-pass image created by subtracting highImage from highImage convolved with
           a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
    :rtype numpy.ndarray
    """
    low_img_filtered = convolve(lowImage, makeGaussianKernel(lowSigma))
    high_img_filtered = highImage - convolve(highImage,makeGaussianKernel(highSigma))
    return (low_img_filtered + high_img_filtered)

def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or 
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    size = int(8.0 * sigma + 1.0)
    if(size % 2 ==0): size += 1
    two_sigma_sqr = 2 * sigma * sigma
    kernel = np.ndarray(shape=(size, size), buffer = np.zeros((size, size)), dtype=float)
    center = int(size / 2)
    sum = 0
    for x in range(size):
        for y in range(size):
            kernel[y,x] = math.exp(-1 * ((x - center)**2 + (y - center)**2) / two_sigma_sqr)
            sum += kernel[y,x]

    return kernel / sum

def zero_pad(image: np.ndarray, num_row_pad, num_col_pad) -> np.ndarray:
    if(image.ndim > 2):
        height, width, channel = image.shape
        padded_img = np.zeros((height + (num_row_pad * 2), width  + (num_col_pad * 2), channel))
        for c in range(channel):
            padded_img[:,:,c] = pad(0, image[:,:,c], num_row_pad, num_col_pad)
        return padded_img
    else: 
        return pad(0, image, num_row_pad, num_col_pad)
        
def pad(pad_value, image: np.ndarray, num_row_pad, num_col_pad):
    height, width = image.shape
    padded_img = np.full((height + (num_row_pad * 2), width + (num_col_pad * 2)), 0)
    padded_img[num_row_pad: height + num_row_pad, num_col_pad: width + num_col_pad] = image
    return padded_img