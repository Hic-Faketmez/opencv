import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('opencv/0_Data/sample.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  


plt.imshow(img)
plt.title("Image")
plt.axis('off')
plt.show()