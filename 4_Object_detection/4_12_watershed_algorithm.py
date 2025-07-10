# Watershed Algorithm
# Watershed algorithm is a powerful technique for image segmentation that treats the grayscale image as a topographic surface.
# It is particularly useful for separating touching objects in an image.
import cv2
import numpy as np 
import matplotlib.pyplot as plt
def display_img(img, title=''):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)
    plt.show()
# Load the image
img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample16.jpg')
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Apply thresholding to create a binary image
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Create a kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)
# Perform morphological operations to remove small noise
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# Dilate the image to get the sure background
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Find the sure foreground area using distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
# Convert sure foreground to uint8  
sure_fg = np.uint8(sure_fg)
# Subtract sure foreground from sure background to get unknown region
unknown = cv2.subtract(sure_bg, sure_fg)
# Label markers for watershed algorithm
_, markers = cv2.connectedComponents(sure_fg)
# Add 1 to all the labels so that sure regions are marked with positive integers
markers = markers + 1
# Mark the unknown region with zero
markers[unknown == 255] = 0
# Apply the watershed algorithm
markers = cv2.watershed(img, markers)
# Mark the boundaries of the regions
img[markers == -1] = [255, 0, 0]  # Mark boundaries in red
# Display the result
title = 'Watershed Algorithm Result'
display_img(img, title)