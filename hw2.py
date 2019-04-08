#----------------------------------------------
# Julianne Delia
# HW 2
# Computer Vision
#----------------------------------------------

import numpy
import numpy as np
import cv2

import matplotlib.pyplot as plt
from PIL import Image

import scipy
from scipy import misc
from scipy import ndimage
from scipy.ndimage import filters

from fractions import Fraction
import math


#Problem 1--------------------------------------------------------------------------------------------------------------------------------------

# Load the images

image1 = np.float64(misc.imread('cheetah.png', flatten=1, mode='F'))
image2 = np.float64(misc.imread('peppers.png', flatten=1, mode='F'))

# apply the gaussian blur to each image

blur_im1 = ndimage.gaussian_filter(image1, 7)
blur_im2 = ndimage.gaussian_filter(image2, 7)

# Display both images
plt.figure(1, figsize=(15, 5))
plt.suptitle('problem 1.3', fontsize=20, fontweight='bold')

plt.subplot(1, 4, 1)
plt.title('image1', fontsize=10)
plt.imshow(image1, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('image1_blurred', fontsize=10)
plt.imshow(blur_im1, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('image2', fontsize=10)
plt.imshow(image2, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('image1_blurred', fontsize=10)
plt.imshow(blur_im2, cmap=plt.cm.gray)
plt.axis('off')
plt.show()

# Compute DFT of both images
dft_im1 = np.fft.fft2(image1)
dft_im1_shift = np.fft.fftshift(dft_im1)
mag_spect_im1 = np.log(np.abs(dft_im1_shift))

dft_im2 = np.fft.fft2(image2)
dft_im2_shift = np.fft.fftshift(dft_im2)
mag_spect_im2 = np.log(np.abs(dft_im2_shift))


# Display magnitude of DFT
plt.figure(1, figsize=(15, 5))
plt.suptitle('problem 1.4', fontsize=20, fontweight='bold')

plt.subplot(1, 2, 1)
plt.title('dft_im1', fontsize=10)
plt.imshow(mag_spect_im1, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('dft_im2', fontsize=10)
plt.imshow(mag_spect_im2, cmap=plt.cm.gray)
plt.axis('off')

plt.show()


#Problem 2--------------------------------------------------------------------------------------------------------------------------------------


# Convert color image to luminance
im3 = np.float64(misc.imread('lowcontrast.jpg', flatten=1))

# Compute histogram
frequencies, bins = numpy.histogram(im3, bins=numpy.arange(-0.5, 255.1, 0.5))
intensities = bins[1:bins.size]

# CDF
cdf = np.cumsum(frequencies)

# transfer
intensities_mapping = cdf/np.float32(cdf[-1]) * 255

# find the number of pure black and white
number_pure_bw = cdf[-1] * 0.025

# find the number of pure black
num_of_blck = 0
i=0
while (num_of_blck < number_pure_bw):
	num_of_blck += frequencies[i]
	intensities_mapping[i] = 0
	i +=1

# find the number of pure white
num_of_white = 0
j = frequencies.size - 1
while (num_of_white < number_pure_bw):
	num_of_white += frequencies[j]
	intensities_mapping[j] = 255
	j -=1

count = im3.size - (num_of_blck + num_of_white)

# equalize the histogram
tol = count * 0.025
trim = 0
while(trim < tol):
	trim = 0
	count = np.sum(frequencies[i:(j+1)])
	for x in range(i, j+1-i):
		ceiling = count/(j+1-i)
		if frequencies[x] > ceiling:
			trim += frequencies[x] - ceiling
			frequencies[x] = ceiling

intensities_mapping[i:(j+1)] = np.cumsum(frequencies[i:(j+1)]).astype(float)/count*255.0 + 255*(float(num_of_blck)/im3.size)
im3_he = np.interp(im3, intensities, intensities_mapping)

# plot the histogram
plt.figure()
plt.title('equalized_histogram', fontsize=10)
plt.imshow(im3_he, cmap=plt.cm.gray)
plt.show()
	

#Problem 3--------------------------------------------------------------------------------------------------------------------------------------


# separable filters
im4 = np.float64(misc.imread('einstein.png'), flatten=1)

# gaussian blur filter
gaus_kern = 1.0/256*numpy.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
gaus_kern_x = numpy.array([1,4,6,4,1])
gaus_kern_y = 1.0/256*numpy.array([1,4,6,4,1])

# gaussian convole
im4_gaus = filters.convolve(im4, gaus_kern, mode="mirror")
im4_gaus_x = filters.convolve1d(im4, gaus_kern_x, axis=1, mode = "mirror")
im4_gaus_y = filters.convolve1d(im4, gaus_kern_y, axis=0, mode = "mirror")

# box filter
box_kern = numpy.ones((5,5), dtype=numpy.float32)/25
box_kern_x = numpy.array([1,1,1,1,1])
box_kern_y = numpy.array([1,1,1,1,1])/25.0

# box convolve
im4_box = filters.convolve(im4, box_kern, mode="wrap") 
im4_box_x = filters.convolve1d(im4, box_kern_x, axis=1, mode="wrap")
im4_box_y = filters.convolve1d(im4, box_kern_y, axis=0, mode="wrap")

# sobel filter
sobel_kern = numpy.array([[1,2,0,-2,-1],[4,8,0,-8,-4],[6,12,0,-12,-6],[4,8,0,-8,-4],[1,2,0,-2,-1]])
sobel_kern_x = numpy.array([1,2,0,-2,-1])
sobel_kern_y = numpy.array([1,4,6,4,1])

# sobel convolve
im4_sobel = filters.convolve(im4, sobel_kern, mode="nearest")
im4_sobel_x = filters.convolve1d(im4, sobel_kern_x, axis=1, mode="nearest")
im4_sobel_y = filters.convolve1d(im4, sobel_kern_y, axis=0, mode="nearest")


# plot gaussian
fig4_gaussian = plt.figure()
fig4_gaussian.suptitle("Gaussian_Convolve")

im4_plot = fig4_gaussian.add_subplot(2,2,1)
im4_plot.set_title("Original")
im4_plot.imshow(im4, cmap=plt.cm.gray)

# im4_gaus
im4_gaussian_plot = fig4_gaussian.add_subplot(2,2,2)
im4_gaussian_plot.set_title("Gaussian_blur")
im4_gaussian_plot.imshow(im4_gaus, cmap=plt.cm.gray)

# im4_gaus_x
im4_gaussian_x_plot = fig4_gaussian.add_subplot(2,2,3)
im4_gaussian_x_plot.set_title("Horizontal_Gaussian")
im4_gaussian_x_plot.imshow(im4_gaus_x, cmap=plt.cm.gray)

# im4_gaus_y
im4_gaussian_y_plot = fig4_gaussian.add_subplot(2,2,4)
im4_gaussian_y_plot.set_title("Vertical_Gaussian")
im4_gaussian_y_plot.imshow(im4_gaus_y, cmap=plt.cm.gray)


# plot box
fig4_box = plt.figure()
fig4_box.suptitle("Box Convolution")

im4_plot = fig4_box.add_subplot(2,2,1)
im4_plot.set_title("Original")
im4_plot.imshow(im4, cmap=plt.cm.gray)

# im4_box
im4_box_plot = fig4_box.add_subplot(2,2,2)
im4_box_plot.set_title("Box-blur")
im4_box_plot.imshow(im4_box, cmap=plt.cm.gray)

# im4_box_x
im4_box_x_plot = fig4_box.add_subplot(2,2,3)
im4_box_x_plot.set_title("Horizontal box")
im4_box_x_plot.imshow(im4_box_x, cmap=plt.cm.gray)

# im4_box_y
im4_box_y_plot = fig4_box.add_subplot(2,2,4)
im4_box_y_plot.set_title("Vertical box")
im4_box_y_plot.imshow(im4_box_y, cmap=plt.cm.gray)


# plot sobel
fig4_sobel = plt.figure()
fig4_sobel.suptitle("Sobel convolution")

im4_plot = fig4_sobel.add_subplot(2,2,1)
im4_plot.set_title("Original")
im4_plot.imshow(im4, cmap=plt.cm.gray)

# im4_sobel
im4_sobel_plot = fig4_sobel.add_subplot(2,2,2)
im4_sobel_plot.set_title("Sobel")
im4_sobel_plot.imshow(im4_sobel, cmap=plt.cm.gray)

# im4_sobel_x
im4_sobel_x_plot = fig4_sobel.add_subplot(2,2,3)
im4_sobel_x_plot.set_title("Horizontal sobel")
im4_sobel_x_plot.imshow(im4_sobel_x, cmap=plt.cm.gray)

# im4_sobel_y
im4_sobel_y_plot = fig4_sobel.add_subplot(2,2,4)
im4_sobel_y_plot.set_title("Vertical sobel")
im4_sobel_y_plot.imshow(im4_sobel_y, cmap=plt.cm.gray)

plt.show()
	

#Problem 4--------------------------------------------------------------------------------------------------------------------------------------

# import the image
gray_scale = np.float64(misc.imread('zebra.png', flatten=1, mode='F'))

# compute the edges of the images
hori_edge = ndimage.sobel(gray_scale, 0)
vert_edge = ndimage.sobel(gray_scale, 1)
mag = np.hypot(hori_edge, vert_edge)

# plot
plt.figure(figsize=(10, 5))
plt.suptitle('#4: Edge_detection', fontsize=20, fontweight='bold')

# plot original image
plt.subplot(1, 4, 1)
plt.imshow(gray_scale, cmap=plt.cm.gray)
plt.axis('off')
plt.title('original')

# plot x-axis
plt.subplot(1, 4, 2)
plt.imshow(hori_edge, cmap=plt.cm.gray)
plt.title('x-axis edges')
plt.axis('off')

# plot y-axis
plt.subplot(1, 4, 3)
plt.imshow(vert_edge, cmap=plt.cm.gray)
plt.title('y-axis edges')
plt.axis('off')

# plot all edges
plt.subplot(1, 4, 4)
plt.imshow(mag, cmap=plt.cm.gray)
plt.title('all edges')
plt.axis('off')

plt.show()


























