# Watershed Algorithm
import cv2
import numpy as np
import matplotlib.pyplot as plt
def display_img(img, title=''):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(title)
    plt.show()

# Load the image
img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample16.jpg')

# Median blur to reduce noise
med_blur = cv2.medianBlur(img, 35)

# Convert to grayscale
gray_med = cv2.cvtColor(med_blur, cv2.COLOR_BGR2GRAY)

# Binary Thresholding
# ret, thresh_med = cv2.threshold(gray_med, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ret, thresh_med = cv2.threshold(gray_med, 230, 255, cv2.THRESH_BINARY_INV)
# Display the thresholded image
# title = 'Thresholded Image'  
# display_img(thresh_med, title)

# Noise removal using morphological operations
kernel = np.ones((3, 3), np.uint8)
opening_med = cv2.morphologyEx(thresh_med, cv2.MORPH_OPEN, kernel, iterations=2)
# display_img(opening_med, 'Opening Operation Result')

# Sure background
sure_bg_med = cv2.dilate(opening_med, kernel, iterations=3)
# display_img(sure_bg_med, "sure background")

# Dinstance transform to find sure foreground
dist_transform_med = cv2.distanceTransform(opening_med, cv2.DIST_L2, 5)
_, sure_fg_med = cv2.threshold(dist_transform_med, 0.7 * dist_transform_med.max(), 255, 0)

sure_fg_med = np.uint8(sure_fg_med)
# display_img(sure_fg_med, "sure foreground")

unknown_med = cv2.subtract(sure_bg_med, sure_fg_med)
display_img(unknown_med, "Unknown")

ret, markers = cv2.connectedComponents(sure_fg_med)

markers = markers + 1

markers [unknown_med==255] = 0

markers = cv2.watershed(img, markers)

# Find contours
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:  # Only draw external contours
        cv2.drawContours(img, contours, i, (0, 255, 0), 2)  # Draw contours in green
# Display the contours on the original image
title = 'Contours on Original Image'
display_img(img, title)