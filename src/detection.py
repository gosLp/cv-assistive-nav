import cv2
from ultralytics import YOLO

def detect_objects(frame, model):
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

    return objects_detected



def detect_objects_test_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        yield objects_detected, frame  # Yield for use in main.py

    cap.release()

