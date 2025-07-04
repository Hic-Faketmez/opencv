import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_img(img):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample6.jpg', 0)
# display_img(img)

# Sobel Gradient
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# display_img(sobel_x)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# display_img(sobel_y)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)  
# display_img(sobel_magnitude)
blended_sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
# display_img(blended_sobel)    

ret,th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# display_img(th1)

kernel = np.ones((4, 4), np.uint8)
gradient = cv2.morphologyEx(blended_sobel, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
# display_img(laplacian)

