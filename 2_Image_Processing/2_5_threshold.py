import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('opencv/0_Data/sample3.jpg', 0)

# print(img.shape)
img = cv2.resize(img, (1500, 2000))

# plt.imshow(img, cmap='gray')
# plt.title("Original Image")
# plt.axis('off')
# plt.show()
print(img.min())
print(img.max())

# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY, dst=img)
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC, dst=img)
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO, dst=img)

# plt.imshow(th1, cmap='gray')
# plt.title("Thresholded Image")
# plt.axis('off')
# plt.show()

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# plt.imshow(th2, cmap='gray')
# plt.title("Adaptive Thresholded Image")
# plt.axis('off')
# plt.show()

blended = cv2.addWeighted(th1, 0.6, th2, 0.4, 0)
plt.imshow(blended, cmap='gray')
plt.title("Blended Image")
plt.axis('off')
plt.show()
