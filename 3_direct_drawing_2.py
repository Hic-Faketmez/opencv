import cv2
import numpy as np
# import matplotlib.pyplot as plt


drawing = False
ix, iy = -1, -1


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            image[:] = 0  # Clear the image
            cv2.rectangle(image, (ix, iy), (x, y), (255, 0, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image, (ix, iy), (x, y), (255, 0, 0), -1)


image = np.zeros((512, 512, 3))
cv2.namedWindow("Image")

cv2.setMouseCallback("Image", draw_rectangle)

while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(20) & 0xFF == 27  # Escape key to exit
    if key:
        break
cv2.destroyAllWindows()

