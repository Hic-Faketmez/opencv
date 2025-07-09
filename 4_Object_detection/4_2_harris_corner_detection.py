# Harris Corner Detection
# https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
import cv2 
import numpy as np
import matplotlib.pyplot as plt
# Load the image
# img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample7.jpg')
img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample8.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Convert to float32
gray = np.float32(gray)
# Apply Harris corner detection
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# Dilate the result to mark the corners
dst = cv2.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image
threshold = 0.01 * dst.max()
img[dst > threshold] = [255, 0, 0]  # Mark corners in red
# Display the result
plt.imshow(img)
plt.title('Harris Corner Detection')
plt.axis('off')
plt.show()  