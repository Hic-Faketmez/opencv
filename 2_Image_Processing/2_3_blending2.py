import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('opencv/0_Data/sample.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (3400, 5000)) 
img2 = cv2.imread('opencv/0_Data/sample2.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2, (1000, 1500))

large_img = img1
small_img = img2

x_offset = 0
y_offset = 0

x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

large_img[y_offset:y_end, x_offset:x_end] = small_img

plt.imshow(large_img)
plt.title("Image")
plt.axis('on')
plt.show()