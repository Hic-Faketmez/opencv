# Face Detection
import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_img(img, title=''):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(title)
    plt.show()

# Load the image
person1 = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample_person_1.jpg', 0)
person2 = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample_person_2.jpg', 0)
people = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample_people_2.jpg', 0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, 1.2, 5)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (0,0,255), 5)
    return face_img

# result = detect_face(people)
# display_img(result, "result")

def detect_eyes(img):
    eyes_img = img.copy()
    eyes_rects = eye_cascade.detectMultiScale(eyes_img, 1.2, 5)

    for (x,y,w,h) in eyes_rects:
        cv2.rectangle(eyes_img, (x,y), (x+w, y+h), (0,0,255), 5)
    return eyes_img

result = detect_eyes(person2)
display_img(result, "result")

