import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('opencv/sample.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

plt.imshow(img_rgb)
plt.show()

# while True:
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(20) & 0xFF == 27  # Escape key to exit
#     if key:
#         break
# cv2.destroyAllWindows()