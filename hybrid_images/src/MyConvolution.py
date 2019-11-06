import numpy as np

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	"""
	Convolve an image with a kernel assuming zero-padding of the image to handle the borders
	
	:param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
	:type numpy.ndarray
	
	:param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
	:type numpy.ndarray 
	
	:returns the convolved image (of the same shape as the input image)
	:rtype numpy.ndarray
	"""
	kheight,kwidth = kernel.shape
	padded_image = zero_pad(image, int(kheight/2), int(kwidth/2))
	if (image.ndim > 2):
		height, width, channel = padded_image.shape
		result_image = np.zeros((height, width, channel))
		for c in range(channel):
			result_image[:,:,c] = convolve_img(padded_image[:,:,c], kernel)
		return remove_pad(result_image, int(kheight/2), int(kwidth/2))
	else:
		return remove_pad(convolve_img(padded_image, kernel), int(kheight/2), int(kwidth/2))

def convolve_img(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	flipped_kernel = np.flip(kernel)
	height, width = image.shape
	kheight,kwidth = kernel.shape
	result_image = np.zeros((height, width))
	height_start = int(kheight / 2)
	height_end = height - int(kheight / 2)
	width_start = int(kwidth / 2)
	width_end = width - int(kwidth / 2)

	for i in range(height):
		for j in range(width):
			if(i >= height_start and i < height_end and j >= width_start and j < width_end):
				row_start = i - int(kheight / 2)
				row_end = i + int(kheight / 2) + 1
				col_start = j - int(kwidth / 2)
				col_end = j + int(kwidth / 2) + 1
				region = image[row_start:row_end, col_start:col_end]
				result_image[i,j] = np.sum(region * flipped_kernel)
			else:
				result_image[i,j] = image[i, j]
	return result_image

def zero_pad(image: np.ndarray, num_row_pad, num_col_pad) -> np.ndarray:
	if(image.ndim > 2):
		height, width, channel = image.shape
		padded_img = np.zeros((height + (num_row_pad * 2), width  + (num_col_pad * 2), channel))
		for c in range(channel):
			padded_img[:,:,c] = pad(0, image[:,:,c], num_row_pad, num_col_pad)
		return padded_img
	else:
		return pad(0, image, num_row_pad, num_col_pad)

def remove_pad(image: np.ndarray, num_row_pad, num_col_pad) -> np.ndarray:
	if(image.ndim > 2):
		height, width, channel = image.shape
		unpadded_img = np.zeros((height - (num_row_pad * 2), width  - (num_col_pad * 2), channel))
		for c in range(channel):
			unpadded_img[:,:,c] = unpad(image[:,:,c], num_row_pad, num_col_pad)
		return unpadded_img
	else:
		return unpad(image, num_row_pad, num_col_pad)

def pad(pad_value, image: np.ndarray, num_row_pad, num_col_pad):
    height, width = image.shape
    padded_img = np.zeros((height + (num_row_pad * 2), width + (num_col_pad * 2)))
    padded_img[num_row_pad: height + num_row_pad, num_col_pad: width + num_col_pad] = image
    return padded_img

def unpad(image: np.ndarray, num_row_pad, num_col_pad):
	height, width = image.shape
	unpadded_img = image[num_row_pad: height - num_row_pad, num_col_pad: width - num_col_pad]
	return unpadded_img