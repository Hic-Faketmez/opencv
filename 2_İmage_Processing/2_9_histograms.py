import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample6.jpg'

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
    return img
def display_img(img, cmap=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap)
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
print("img shape:", img.shape)

# Color Histogram
# Calculate histogram for each color channel
# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     hist = cv2.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])
# plt.title('Color Histogram')
# plt.xlabel('Pixel Value')  
# plt.ylabel('Frequency')
# plt.show()

# Histogram Equalization
# Convert to grayscale and equalize histogram
mask = np.zeros(img.shape[:2], dtype=np.uint8)
# plt.imshow(mask, cmap='gray')
# plt.show()
mask[300:400, 400:800] = 255
# plt.imshow(mask, cmap='gray')
# plt.show()
masked_img = cv2.bitwise_and(img, img, mask=mask)
# display_img(masked_img)
show_masked_img = cv2.bitwise_and(show_img, show_img, mask=mask)
# display_img(show_masked_img)

hist_mask_values_red = cv2.calcHist([img], channels=[2], mask=mask, histSize=[256], ranges=[0, 256])
# plt.plot(hist_mask_values_red, color='red', label='Masked Red Channel')
# plt.title('Masked Red Channel Histogram')
# plt.xlabel('Pixel Value') 
# plt.ylabel('Frequency')
# plt.xlim([0, 256])
# plt.legend()
# plt.show()

hist_values_red = cv2.calcHist([img], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
# plt.plot(hist_values_red, color='orange', label='Red Channel')
# plt.title('Red Channel Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.xlim([0, 256])
# plt.legend()
# plt.show()  

image = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample6.jpg', 0)
print("image shape:", image.shape)
# display_img(image, cmap='gray')

hist_values = cv2.calcHist([image], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])
# plt.plot(hist_values, color='blue', label='Masked Grayscale Histogram')
# plt.title('Masked Grayscale Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.xlim([0, 256])
# plt.legend()
# plt.show()

eq_image = cv2.equalizeHist(image)
# display_img(eq_image, cmap='gray')

hist_values = cv2.calcHist([eq_image], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])
# plt.plot(hist_values, color='blue', label='Masked Grayscale Histogram')
# plt.title('Masked Grayscale Histogram after Equalization')
# plt.show()

hsv = cv2.cvtColor(real_img, cv2.COLOR_BGR2HSV)
# print (hsv[:, :, 2].max())
# print (hsv[:, :, 2].min())

hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
eq_color_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
display_img(eq_color_img)

