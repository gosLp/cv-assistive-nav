import time
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneCaptioner:
    def __init__(self):
        # Initialize BLIP model and processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.last_caption_time = time.time()

    def generate_caption(self, detected_objects):
        current_time = time.time()

        # Generate a new caption every 2 seconds
        if current_time - self.last_caption_time < 2:
            return None  # Skip captioning if we just generated a caption

        self.last_caption_time = current_time

        if not detected_objects:
            return "No obstacles detected. Path is clear."

        descriptions = []
        for obj in detected_objects:
            descriptions.append(f"{obj['label']} detected with {obj['confidence']:.2f} confidence")

        description_text = " and ".join(descriptions)
        return f"Scene: {description_text}"
