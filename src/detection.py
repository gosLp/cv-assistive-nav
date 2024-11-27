import cv2

from ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('yolov8n.pt')

# Load camera for detection
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from video capture")
        break

    # Perform detection
    results = model(frame)

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('Frame', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the capture and close all windows
cap.release()
cv2.destroyAllWindows()




