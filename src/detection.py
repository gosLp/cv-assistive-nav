import cv2
from ultralytics import YOLO

def detect_objects(cap, model):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video capture")
        return [], None

    # Run object detection on the frame
    results = model(frame)

    # Extract object details from the results
    objects_detected = []
    for detection in results[0].boxes:
        label = model.names[int(detection.cls)]
        confidence = detection.conf
        box = detection.xyxy  # Bounding box coordinates
        objects_detected.append({
            'label': label,
            'confidence': float(confidence),
            'box': box.tolist()
        })

    return objects_detected, frame
