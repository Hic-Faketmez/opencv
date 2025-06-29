import cv2
import numpy as np
# import matplotlib.pyplot as plt

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 50, (0, 255, 0), thickness=-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(image, (x, y), 50, (0, 0, 255), thickness=5)

cv2.namedWindow("Image")

cv2.setMouseCallback("Image", draw_circle)


image = np.zeros((512, 512, 3), dtype=np.int8)
while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(20) & 0xFF == 27  # Escape key to exit
    if key:
        break
cv2.destroyAllWindows()

