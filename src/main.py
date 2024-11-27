import cv2
from ultralytics import YOLO
from src.detection import detect_objects
from src.captioning import SceneCaptioner

def main():
    # Initialize YOLO model and video capture
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    # Initialize the scene captioner
    scene_captioner = SceneCaptioner()

    try:
        while True:
            # Detect objects
            detected_objects, frame = detect_objects(cap, model)

            # Generate a scene caption
            scene_description = scene_captioner.generate_caption(detected_objects)
            if scene_description:
                print(scene_description)

            # Display the annotated frame with detected objects
            if frame is not None:
                annotated_frame = model(frame)[0].plot()  # Visual representation
                cv2.imshow('YOLOv8 Detection', annotated_frame)

            # Press 'q' to quit the video feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
