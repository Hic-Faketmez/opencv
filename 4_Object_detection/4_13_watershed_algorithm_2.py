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
sep_blur = cv2.medianBlur(img, 5)
# Grayscale
gray_sep = cv2.cvtColor(sep_blur, cv2.COLOR_BGR2GRAY)
# Binary Thresholding
ret, thresh_sep = cv2.threshold(gray_sep, 230, 255, cv2.THRESH_BINARY_INV)
# Display the thresholded image
# title = 'Thresholded Image'
# display_img(thresh_sep, title)
# Find contours
contours, hierarchy = cv2.findContours(thresh_sep.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:  # Only draw external contours
        cv2.drawContours(img, contours, i, (0, 255, 0), 2)  # Draw contours in green
# Display the contours on the original image
title = 'Contours on Original Image'
display_img(img, title)