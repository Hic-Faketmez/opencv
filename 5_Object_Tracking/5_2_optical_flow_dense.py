# Dense Optical Flow
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
prvsImg = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv_mask = np.zeros_like(frame1)
hsv_mask[:,:,1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    nextImg = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvsImg, nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)

    hsv_mask[:,:,0] = ang/2
    hsv_mask[:,:,2] = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    cv2.imshow("Tracking Frame", bgr)

    k = cv2.waitKey(30) & 0xFF
    if k ==27:
        break

    prvsImg = nextImg

cv2.destroyAllWindows()
cap.release()


