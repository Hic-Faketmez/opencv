#Shi Tomassi Corner Detection
# https://docs.opencv.org/4.x/d9/d0c/group__imgproc__feature.html#ga0c1f8b2d3a4e5f6b7c9d8e2f3c4a5b6
import cv2 
import numpy as np
import matplotlib.pyplot as plt
# Load the image
img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample8.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Convert to float32
gray = np.float32(gray)
# Apply Shi-Tomasi corner detection
dst = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3) 
# dst array of corners uses float32 so uses int(x) and int(y) to convert to integer for drawing otherwise use dst = np.int0(dst)
# Draw the corners on the image
for i in dst:
    x, y = i.ravel()
    cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), -1)  # Mark corners in red
# Display the result
plt.imshow(img)
plt.title('Shi-Tomasi Corner Detection')
plt.axis('off')
plt.show()