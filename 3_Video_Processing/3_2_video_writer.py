import cv2

cap = cv2.VideoCapture(0)  # Open the default camera
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('myvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20.0, (width, height))
if not writer.isOpened():
    print("Error: Could not open video writer.")
    exit()

print(f"Video resolution: {width}x{height}")

while True:
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        print("Error: Could not read frame.")
        break
    # Write the frame to the video file
    writer.write(frame)

    # Display the frame in a window
    cv2.imshow('Video Frame', frame)

    # Wait for 1 ms and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
writer.release()
cv2.destroyAllWindows()


