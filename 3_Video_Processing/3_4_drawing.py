import cv2
import time

path = 'C:/Users/pc/Desktop/Yaz/opencv/0_Data/myvideo.mp4'
cap = cv2.VideoCapture(path)  # Open the default camera
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video resolution: {width}x{height}")

#Top left corner of the rectangle
x = width // 2
y = height // 2
# Width and height of the rectangle
w = width // 4
h = height // 4

# bottom right corner of the rectangle x + w, y + h

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)  # Draw a rectangle on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a rectangle on the frame
    
    # Display the frame in a window
    time.sleep(1/20)  # Optional: Add a small delay to control frame rate
    cv2.imshow('Video Frame', frame)

    # Wait for 1 ms and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


