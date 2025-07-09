# Grid detection using findChessboardCorners and drawChessboardCorners
import cv2
import numpy as np 
import matplotlib.pyplot as plt
# Load the image
img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample7.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Apply findChessboardCorners to detect the grid pattern
pattern_size = (14, 9)  # Adjust based on the grid size in the image (7, 7) for a 7x7 grid
# Find the chessboard corners
found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
# If corners are found, draw them on the image
if found:
    cv2.drawChessboardCorners(img, pattern_size, corners, found)
# Display the result
plt.imshow(img)
plt.title('Grid Detection using findChessboardCorners')
plt.axis('off')
plt.show()

# Find the circles grid
pattern_size = (7, 7)  # Adjust based on the grid size in the image (7, 7) for a 7x7 grid
found, centers = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
# If centers are found, draw them on the image 
if found:
   cv2.drawChessboardCorners(img, pattern_size, centers, found)
# Display the result
plt.imshow(img)
plt.title('Grid Detection using findCirclesGrid')
plt.axis('off')
plt.show()
