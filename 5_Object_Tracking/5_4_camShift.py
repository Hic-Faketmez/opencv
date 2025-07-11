# CamShift
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_rects = face_cascade.detectMultiScale(frame)

(face_x, face_y, w,h) = tuple(face_rects[0])
track_window = (face_x, face_y, w,h)

roi = frame[face_y:face_y+h,face_x:face_x+w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0,180])

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    pts = cv2.boxPoints(ret)
    pts = np.int32(pts)
    tracking_frame = cv2.polylines(frame, [pts], True, (255,0,0), 2)
   
    cv2.imshow("Tracking Frame", tracking_frame)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
