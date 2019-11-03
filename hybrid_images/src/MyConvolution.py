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
	if (image.ndim > 2):
		height, width, channel = image.shape
		result_image = np.copy(image)
		for c in range(channel):
			r = convolve_img(image[:,:,c], kernel)
			result_image[:,:,c] = convolve_img(image[:,:,c], kernel)
		return result_image
	else:
		return convolve_img(image, kernel)

def convolve_img(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	height, width = image.shape
	kheight,kwidth = kernel.shape
	result_image = np.copy(image)

	height_start = int(kheight / 2)
	height_end = height - int(kheight / 2)
	width_start = int(kwidth / 2)
	width_end = width - int(kwidth / 2)

	for i in range(height_start, height_end):
		for j in range(width_start, width_end):
			row_start = i - int(kheight / 2)
			row_end = i + int(kheight / 2) + 1
			col_start = j - int(kwidth / 2)
			col_end = j + int(kwidth / 2) + 1
			region = image[row_start:row_end, col_start:col_end]
			result_image[i,j] = np.sum(region * kernel)
	return result_image