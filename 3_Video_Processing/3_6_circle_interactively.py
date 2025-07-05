import cv2
import time

# path = 'C:/Users/pc/Desktop/Yaz/opencv/0_Data/myvideo.mp4'
path = 0

cap = cv2.VideoCapture(path)  # Open the default camera
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Callbacks to handle mouse events
def draw_circle(event, x, y, flags, param):
    global center, clicked
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        center = (x, y)
        clicked = True  # Set the flag to indicate that the circle should be drawn
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right mouse button clicked
        clicked = False  # Reset the flag to stop drawing the circle
    
# Globals
center = (0, 0 )  
clicked = False

# Connect the mouse callback function to the OpenCV window
cv2.namedWindow('Video Frame')
cv2.setMouseCallback('Video Frame', draw_circle)

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        print("Error: Could not read frame.")
        break
    if clicked:
        # Draw a circle at the center of the frame
        cv2.circle(frame, center, 50, (0, 255, 0), 3)

    # Display the frame in a window
    time.sleep(1/20)  # Optional: Add a small delay to control frame rate
    cv2.imshow('Video Frame', frame)

    # Wait for 1 ms and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()