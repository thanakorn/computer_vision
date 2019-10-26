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
	height, width = (image.shape[0], image.shape[1])
	channel = image.shape[2] if(image.ndim) > 2 else 1
	kheight,kwidth = kernel.shape

	print (height, width, channel)

	result_image = np.ndarray(shape=(height, width, channel), buffer = np.zeros((height, width, channel)), dtype=int)
	return result_image