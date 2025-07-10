# Watershed Algorithm custom seeds
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def display_img(img, title=''):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(title)
    plt.show()

# Load the image
img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample16.jpg')

# Copy the image
img_copy = np.copy(img)

# Setting marker image and segments
marker_image = np.zeros(img.shape[:2], dtype = np.int32)

segments = np.zeros(img.shape, dtype = np.uint8)

# Setting Colors
# cm is one of the color palette of matplotlib
def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)

colors = []
for i in range (10):
    colors.append(create_rgb(i))

# Global Variables
current_marker = 1
marks_updated = False
n_markers = 10

# Callback Function
def mouse_callback (event, x, y, flags, param):
    global marks_updated

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(marker_image, (x, y), 10, (current_marker), -1)

        cv2.circle(img_copy, (x, y), 10, (colors[current_marker]), -1)

        marks_updated = True

# While True statement
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

while True:
    cv2.imshow("Watershed Segments", segments)
    cv2.imshow("Image", img_copy)
    
    # Close All Windows
    k = cv2.waitKey(1)
    if k==27:
        break
    # Clear All Colors
    elif k ==ord("c"):
        img_copy = img.copy()
        marker_image = np.zeros(img.shape[:2], dtype = np.int32)
        segments = np.zeros(img.shape, dtype = np.uint8)

    # Update Color Choice
    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))

    # Update the Markings
    if marks_updated:
        marker_image_copy = marker_image.copy()
        cv2.watershed(img, marker_image_copy)

        segments = np.zeros(img.shape, dtype = np.uint8)
        for color_ind in range(n_markers):
            segments[marker_image_copy == (color_ind)] = colors[color_ind]

cv2.destroyAllWindows()