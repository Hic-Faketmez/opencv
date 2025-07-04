import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img():
    img = np.zeros((600, 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'ABCDE', (50, 300), fontFace=font, fontScale=5, color=(255, 255, 255), thickness=25)
    return img

def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()
img = load_img()
# display_img(img)

kernel = np.ones((5, 5), np.uint8)
# Erosion   
erosion = cv2.erode(img, kernel, iterations=1) # iterations=1 means one time erosion
# display_img(erosion)

white_noise = np.random.randint(0, 2, (600, 600)) * 255
img_with_noise = white_noise + img
# display_img(img_with_noise)
opening = cv2.morphologyEx(img_with_noise, cv2.MORPH_OPEN, kernel)
# display_img(opening)

black_noise = np.random.randint(0, 2, (600, 600)) * -255
img_with_black_noise = black_noise + img
img_with_black_noise[img_with_black_noise == -255] = 0  # Set negative values to zero
# display_img(img_with_black_noise)
closing = cv2.morphologyEx(img_with_black_noise, cv2.MORPH_CLOSE, kernel)
# display_img(closing)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# display_img(gradient)

dilation = cv2.dilate(img, kernel, iterations=3) # iterations=1
display_img(dilation)