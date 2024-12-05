import cv2
from ultralytics import YOLO

def detect_objects(frame, model):
    # Run object detection on the frame
    results = model(frame)

    # Extract object details from the results
    objects_detected = []


    # fram center corddinates
    frame_center = (frame.shape[1]/2, frame.shape[0]/2)
    for detection in results[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = detection
        label = model.names[int(class_id)]

        # Calculate the center of the bounding box
        obj_center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Calculate the distance from the frame center to the object center
        distance_from_center_x = obj_center[0] - frame_center[0]
        distance_from_center_y = obj_center[1] - frame_center[1]

        # determine direction of object from frame center
        if abs(distance_from_center_x) < frame.shape[1] * 0.1:
            direction = "center"
        elif distance_from_center_x > 0:
            direction = "right"
        else:
            direction = "left"
        # box = detection.xyxy  # Bounding box coordinates
        objects_detected.append({
            'label': label,
            'confidence': float(confidence),
            'bbox': (x1, y1, x2, y2),
            "direction": direction,
            "distance_from_center": (distance_from_center_x, distance_from_center_y),
            "avg_depth": None
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

