# External and internal counter detection
import cv2
import numpy as np 
import matplotlib.pyplot as plt
# Load the image
img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample8.jpg', 0)

counters, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print (f"Number of contours found: {len(counters)}")
print(f"Hierarchy: {hierarchy}")

# external contours
external_contours = np.zeros(img.shape)

for i, c in enumerate(counters):
    if hierarchy[0][i][3] == -1:  # Check if the contour is external
        cv2.drawContours(external_contours, counters, i, (255, 255, 255), -1)  # Fill external contours
#show external contours
plt.imshow(external_contours, cmap='gray')
plt.title('External Contours')
plt.axis('off')
plt.show()

# internal contours
internal_contours = np.zeros(img.shape)
for i, c in enumerate(counters):
    if hierarchy[0][i][3] != -1:  # Check if the contour is internal
        cv2.drawContours(internal_contours, counters, i, (255, 255, 255), -1)  # Fill internal contours
# show internal contours
plt.imshow(internal_contours, cmap='gray')
plt.title('Internal Contours')
plt.axis('off')
plt.show()

# Draw contours on the original image
img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB for display
cv2.drawContours(img_contours, counters, -1, (0, 255, 0), 1)  # Draw all contours in green
# Display the result
plt.figure(figsize=(12, 6))
plt.imshow(img_contours)
plt.title('Contours Detected')
plt.axis('off')
plt.show()
