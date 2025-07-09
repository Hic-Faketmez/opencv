# Edge Detection using Canny Edge Detector
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_canny.html
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample8.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Apply Gaussian blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Threshold1 and Threshold2 are the lower and upper thresholds for the Canny edge detector
#The lower and upper thresholds;
med_val = np.median(blurred)
lower_threshold = int(max(0, (1.0 - 0.33) * med_val))
upper_threshold = int(min(255, (1.0 + 0.33) * med_val))
print(f"Lower Threshold: {lower_threshold}, Upper Threshold: {upper_threshold}")
# Apply Canny edge detection
edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
# edges = cv2.Canny(blurred, 100, 200)  # Adjust thresholds as needed
# Draw the edges on the original image
img_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # Convert edges to RGB for display
# Display the result
plt.imshow(img_edges)
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()
