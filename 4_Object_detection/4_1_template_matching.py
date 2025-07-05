import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

template = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample2_template.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

model = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCOEFF, cv2.TM_CCORR_NORMED, cv2.TM_CCORR, cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF]

# model = [5, 4, 3, 2, 1, 0]  # Using indices to refer to the models for easier iteration

# print("model:", model[0]  if isinstance(model, list) else model)

for i in model:
    print(f"Using model: {i}")
    result = cv2.matchTemplate(img, template, i)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Get the coordinates of the matched region
    if i in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    h, w, _ = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Draw a rectangle around the matched region
    cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 9)
# result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    plt.imshow(img)
    plt.title(f"Template Matching Result using {i}")
    plt.axis('off')
    plt.show()
# Save the result image
# cv2.imwrite('C:/Users/pc/Desktop/Yaz/opencv/4_Object_detection/template_matching_result.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) 

# The template matching methods used are:
# cv2.TM_CCOEFF_NORMED, cv2.TM_CCOEFF, cv2.TM_CCORR_NORMED, cv2.TM_CCORR, cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF

