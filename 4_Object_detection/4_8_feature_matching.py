# Feature Matching Brute Force Matching with ORB Descriptor
# This code demonstrates how to perform feature matching between two images using 
# ORB (Oriented FAST and Rotated BRIEF) descriptors and the Brute Force Matcher in OpenCV.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
img1 = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample2_template.jpg', 0)  # Query image
img2 = cv2.imread('C:/Users/pc/Desktop/Yaz/opencv/0_Data/sample2.jpg', 0)  # Train image
# Initialize ORB detector
orb = cv2.ORB_create()
# Find keypoints and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
# Create a Brute Force Matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors
matches = bf.match(descriptors1, descriptors2)
# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)
# Draw matches
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:25], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# Display the matches
plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title('Feature Matches (ORB + Brute Force)')
plt.axis('off')
plt.show()

