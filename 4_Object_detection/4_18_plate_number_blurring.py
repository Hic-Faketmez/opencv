# Plate Number Blurring
import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_img(img, title=''):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)  # cmap='gray'
    ax.axis('off')
    ax.set_title(title)
    plt.show()

# Load the image
img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample17.jpg', 0)

plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

def detect_plate(img):
    plate_img = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img, 1.2, 5)

    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (0,0,255), 5)
    return plate_img

result = detect_plate(img)
display_img(result, "detected plate")

def detect_and_blur_plate(img):
    plate_img = img.copy()

    plate_rects = plate_cascade.detectMultiScale(plate_img, 1.2, 5)

    for (x,y,w,h) in plate_rects:
        roi = plate_img[y:y+h, x:x+w]
        # blurred_roi = cv2.medianBlur(roi, 15) # Median Blur
        blurred_roi = cv2.GaussianBlur(roi, (15, 15), 10) # Gaussian Blur
        plate_img[y:y+h, x:x+w] = blurred_roi
    
    return plate_img
    
result1 = detect_and_blur_plate(img)
display_img(result1, "Blured plate")