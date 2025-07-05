import cv2
import time

# path = 'C:/Users/pc/Desktop/Yaz/opencv/0_Data/myvideo.mp4'
path = 0

cap = cv2.VideoCapture(path)  # Open the default camera
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Callbacks to handle mouse events
def draw_rectangle(event, x, y, flags, param):
    global pt1, pt2, top_left_clicked, bottom_right_clicked
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        if top_left_clicked and bottom_right_clicked:
            # Reset the rectangle if both corners are clicked
            pt1 = (x, y)
            pt2 = (x, y)
            top_left_clicked = False
            bottom_right_clicked = False

        if not top_left_clicked:
            pt1 = (x, y)    # Set the top-left corner of the rectangle
            top_left_clicked = True
        
        elif not bottom_right_clicked:
            pt2 = (x, y)
            bottom_right_clicked = True

# Globals
pt1 = (0, 0 )  # Starting point of the rectangle
pt2 = (0, 0 )  # Ending point of the rectangle
top_left_clicked = False
bottom_right_clicked = False

# Connect the mouse callback function to the OpenCV window
cv2.namedWindow('Video Frame')
cv2.setMouseCallback('Video Frame', draw_rectangle)


while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        print("Error: Could not read frame.")
        break
    # Draw the rectangle on the frame if both corners are clicked
    if top_left_clicked:
        cv2.circle(frame, pt1, 5, (0, 255, 0), -1)

    if top_left_clicked and bottom_right_clicked:
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        # Reset the flags after drawing the rectangle

    # Display the frame in a window
    time.sleep(1/20)  # Optional: Add a small delay to control frame rate
    cv2.imshow('Video Frame', frame)

    # Wait for 1 ms and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()