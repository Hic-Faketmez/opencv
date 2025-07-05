import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('opencv/0_Data/sample4.jpg', 0)
path = 'C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample4.jpg'

def load_img(path):
    img = cv2.imread(path).astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
    return img
def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()
i = load_img(path)
# display_img(i)

################
gamma = 1/4
result = np.power(i,gamma)
# display_img(result)

#################
img = load_img(path)
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, 'bricks', (10, 400), fontFace=font, fontScale=9, color=(255, 0, 0), thickness=4)
# display_img(img)

#################
kernel = np.ones((5, 5), np.float32) / 25
kernel_blurred = cv2.filter2D(img, -1, kernel)
# display_img(kernel_blurred)

#################
blurred = cv2.blur(img, (10, 10))
# display_img(blurred)

#################
gaussian_blurred = cv2.GaussianBlur(img, (5, 5), 10)
# display_img(gaussian_blurred)

#################
median_blurred = cv2.medianBlur(img, 5) # used to remove salt and pepper noise
# display_img(median_blurred)

#################
bilateral_blurred = cv2.bilateralFilter(img, 9, 75, 75) # used to remove noise while keeping edges sharp
display_img(bilateral_blurred)
