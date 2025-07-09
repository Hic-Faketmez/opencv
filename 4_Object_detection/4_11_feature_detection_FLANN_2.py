# FLANN based feature matching example using SIFT
# This code demonstrates how to perform feature matching between two images using
# SIFT (Scale-Invariant Feature Transform) descriptors and the FLANN (Fast Library for Approximate Nearest Neighbors) matcher in OpenCV.
import cv2
import numpy as np
import matplotlib.pyplot as plt
def display_img(img, title=''):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)
    plt.show()
# Load the images
img1 = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample12.jpeg', 0)  # Query image
img2 = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample13.jpeg', 0)  # Train image
# Initialize SIFT detector 
sift = cv2.SIFT_create()
# Find keypoints and descriptors with SIFT
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
# Create FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass an empty dictionary
# Create a FLANN matcher object
flann = cv2.FlannBasedMatcher(index_params, search_params)
# Match descriptors using knnMatch
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

matchesMask = [[0,0] for i in range(len(matches))]
for i, (m, n) in enumerate(matches):
    if m.distance < 0.90 * n.distance:
        matchesMask[i]=[1,0]
    # else:
    #     matchesMask[i]=[0,1]
draw_params = dict(matchColor=(0, 255, 0), 
                   singlePointColor=(255, 0, 255), 
                   matchesMask=matchesMask, 
                   flags=0)
# Apply ratio test to filter out weak matches       

# Draw matches
flann_matches = cv2.drawMatchesKnn(
    img1, 
    keypoints1, 
    img2, 
    keypoints2, 
    matches,
    None, 
    **draw_params)
# Display the matches
title = 'Feature Matching with SIFT and FLANN Matcher'
display_img(flann_matches, title)
