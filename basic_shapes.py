import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplot inline

blank_image = np.zeros((512, 512, 3), dtype=np.int16)
# print(blank_image.shape)
# plt.imshow(blank_image)
# plt.title("Blank Image")
# plt.axis('off')
# plt.show()

cv2.rectangle(blank_image, (384, 10), (510, 128), (255, 0, 0), thickness=3)
cv2.circle(blank_image, (256, 256), 100, (0, 255, 0), thickness=-1)
cv2.line(blank_image, (0, 0), (511, 511), (0, 0, 255), thickness=5)
cv2.putText(blank_image, "Supremacy", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=5)

plt.imshow(blank_image)
plt.title("Blank Image")
plt.axis('off')
plt.show()