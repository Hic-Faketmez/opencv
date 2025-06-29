import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('opencv/0_Data/sample.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = cv2.resize(img1, (3400, 5000)) 
img2 = cv2.imread('opencv/0_Data/logo.jpeg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2, (1000, 1000))
# print(img2.shape)

# large_img = img1
# small_img = img2

x_offset = img1.shape[1] - img2.shape[1]
y_offset = img1.shape[0] - img2.shape[0]

x_end = img1.shape[1]
y_end = img1.shape[0]

# x_offset = large_img.shape[1] - small_img.shape[1]
# y_offset = large_img.shape[0] - small_img.shape[0]

# x_end = large_img.shape[1]
# y_end = large_img.shape[0]

rows, cols, channels = img2.shape

roi = img1[y_offset:y_end, x_offset:x_end]

# plt.imshow(roi)
# plt.title("Region of Interest (ROI)")
# plt.axis('on')
# plt.show()

img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
# plt.imshow(img2gray, cmap='gray')
# plt.title("Grayscale Image")
# plt.axis('on')
# plt.show()

mask_inv = cv2.bitwise_not(img2gray)
# plt.imshow(mask_inv, cmap='gray')
# plt.title("Inverse Mask")
# plt.axis('on')
# plt.show()

print (mask_inv.shape)
white_background = np.full(img2.shape, 255, dtype=np.uint8)
print (white_background.shape)
bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
# plt.imshow(bk)
# plt.title("White Background")
# plt.axis('on')
# plt.show()

fg = cv2.bitwise_or(img2, img2, mask=mask_inv)
# plt.imshow(fg)
# plt.title("Foreground")
# plt.axis('on')
# plt.show()

final_roi = cv2.bitwise_or(roi, fg)
# plt.imshow(final_roi)
# plt.title("Final ROI")
# plt.axis('on')
# plt.show()

large_img = img1
small_img = final_roi

large_img[y_offset:y_end, x_offset:x_end] = small_img
plt.imshow(large_img)
plt.title("Blended Image")
plt.axis('on')
plt.show()