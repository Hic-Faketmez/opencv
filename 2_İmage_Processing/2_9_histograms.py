import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample6.jpg'

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
    return img
def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()

real_img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample6.jpg')
# display_img(real_img)

show_img = load_img(path)
# display_img(show_img)

# hist = cv2.calcHist([real_img], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.title('Histogram')
# plt.xlabel('Pixel Value') 
# plt.ylabel('Frequency')
# plt.xlim([0, 256])
# plt.show()

img = real_img
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.title('Color Histogram')
plt.xlabel('Pixel Value')  
plt.ylabel('Frequency')
plt.show()