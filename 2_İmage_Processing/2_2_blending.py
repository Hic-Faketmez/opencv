import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('opencv/0_Data/sample.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (3400, 5000)) 
img2 = cv2.imread('opencv/0_Data/sample2.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2, (3400, 5000))

# print(img1.shape)
# print(img2.shape)

blended = cv2.addWeighted(img1, 0.5, img2, 0.7, 0)

plt.imshow(blended)
plt.title("Image")
plt.axis('off')
plt.show()