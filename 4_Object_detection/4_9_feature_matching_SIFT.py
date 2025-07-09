# Feature Matching Brute Force Matching with SIFT Descriptors
# This code demonstrates how to perform feature matching between two images using
# SIFT (Scale-Invariant Feature Transform) descriptors and the Brute Force Matcher in OpenCV.
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
# Create a Brute Force Matcher object
bf = cv2.BFMatcher()
# Match descriptors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
# Apply ratio test to filter out weak matches
good_matches = []
for m, n in matches:
    if m.distance < 0.90* n.distance:
        good_matches.append(m)
print(f'Number of matches: {len(matches)}')
print(f'Number of good matches: {len(good_matches)}')

# Sort good matches by distance (best matches first)
# good_matches = sorted(good_matches, key=lambda x: x.distance)

# Draw matches
sift_matches = cv2.drawMatchesKnn(
    img1, 
    keypoints1, 
    img2, 
    keypoints2, 
    [ [m] for m in good_matches[:25] ], 
    # [ [m] for m, n in matches[:25] ], # if there is no good_matches
    None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)   

# Match descriptors
matches = bf.match(descriptors1, descriptors2)
# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)
# Draw matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:25], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# Display the matches

# plt.figure(figsize=(12, 6))
# plt.imshow(img_matches)
# plt.title('Feature Matches (SIFT + Brute Force)')
# plt.axis('off')
# plt.show()

title = 'Feature Matching with SIFT and Brute Force Matcher'
display_img(sift_matches, title)