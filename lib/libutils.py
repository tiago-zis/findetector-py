import numpy as np
import cv2
from PIL import Image

class LibUtils:

	def load_image_into_numpy_array(image):
		(im_width, im_height) = image.size
    	
		if image.format == 'PNG':
			image = image.convert('RGB')
    	
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
	
	def auto_canny(image, sigma=0.33):
	    # compute the median of the single channel pixel intensities
	    v = np.median(image)
	 
	    # apply automatic Canny edge detection using the computed median
	    lower = int(max(0, (1.0 - sigma) * v))
	    upper = int(min(255, (1.0 + sigma) * v))
	    edged = cv2.Canny(image, lower, upper)
	 
	    # return the edged image
	    return edged
	   
	def dilation(image, iterations, kernel_size):
		kernel = np.ones((kernel_size,kernel_size),np.uint8)
		res = cv2.dilate(image , kernel, iterations = iterations)
		return res
