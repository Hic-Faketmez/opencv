import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('opencv/sample.jpg')

# Convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
# Flip the image vertically
new_img = img_rgb.copy()
# new_img = cv2.flip(img_rgb, 0)  


pt1 = (1000, 1000) # Top-left corner
pt2 = (2000, 2000)  # Bottom-right corner
cv2.rectangle(new_img, pt1, pt2, (255, 0, 0), 10)  # Draw rectangle in blue
vertices = np.array([[1000,4000], pt2, [3000,4000]], dtype=np.int32)
pts = vertices.reshape((-1, 1, 2))
# cv2.polylines(new_img, [pts], isClosed=True, color=(0, 255, 0), thickness=10)  # Draw polygon in green
cv2.fillPoly(new_img, [pts], color=(0, 0, 255))  # Fill polygon in red
# plt.imshow(new_img)
# plt.show()


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(new_img, (x, y), 500, (0, 255, 0), thickness=-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(new_img, (x, y), 500, (0, 0, 255), thickness=5)

cv2.namedWindow("Image")

cv2.setMouseCallback("Image", draw_circle)


while True:
    cv2.resizeWindow("Image", 800, 600)  # Resize the window
    cv2.imshow("Image", new_img)
    key = cv2.waitKey(20) & 0xFF == 27  # Escape key to exit
    if key:
        break
cv2.destroyAllWindows()