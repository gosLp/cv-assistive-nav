import cv2
import torch
import numpy as np
import logging


class DepthEstimator:
    def __init__(self):
        self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

    def estimate_depth(self, frame):
        # frame_resized = cv2.resize(frame, (256, 256))
        # convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # transform the frame to the models espected format
        input_batch = self.transform(frame_rgb).to(self.device)
        # Check the input tensor shape
        logging.info(f"Input batch shape for depth estimation: {input_batch.shape}")

        #         # Fix the input batch shape to ensure it is correct
        # if input_batch.shape[1] == 1:
        #     input_batch = input_batch.squeeze(1)  # Remove the extra dimension if added mistakenly


        # Ensure input is properly shaped: [batch, channels, height, width]
        if input_batch.ndim == 3:
            input_batch = input_batch.unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()

        # Normalize depth for visualization
        depth_normalized = cv2.normalize(
            depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        return depth_map, depth_normalized
    def analyze_object_distances(self, frame, detected_objects):
        try:
            depth_map, _= self.estimate_depth(frame)
            logging.info(f"Depth map shape: {depth_map.shape}")

            # Get the height and width of the original frame and the depth map
            frame_height, frame_width, _ = frame.shape
            depth_height, depth_width = depth_map.shape

            for obj in detected_objects:
                x1, y1, x2, y2 = obj['bbox']
                # Clip coordinates to be within frame bounds
                # x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Scale bounding box coordinates to match the depth map size
                x1_scaled = int(x1 / frame_width * depth_width)
                y1_scaled = int(y1 / frame_height * depth_height)
                x2_scaled = int(x2 / frame_width * depth_width)
                y2_scaled = int(y2 / frame_height * depth_height)

                # Clip coordinates to be within frame bounds
                x1_scaled = max(0, min(x1_scaled, depth_width - 1))
                y1_scaled = max(0, min(y1_scaled, depth_height - 1))
                x2_scaled = max(0, min(x2_scaled, depth_width - 1))
                y2_scaled = max(0, min(y2_scaled, depth_height - 1))

                obj_roi = depth_map[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
                if obj_roi.size == 0:
                    logging.warning(f"Empty ROI for object: {obj['label']}, bbox: ({x1}, {y1}), ({x2}, {y2})")
                    continue
                avg_depth = np.mean(obj_roi)
                obj['avg_depth'] = avg_depth
            

            return detected_objects
        except Exception as e:
            logging.error(f"Depth estimation error: {e}")
            return detected_objects

