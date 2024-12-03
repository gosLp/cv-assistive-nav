from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class SceneCaptioner:
    def __init__(self):
        # Initialize BLIP model and processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def generate_caption(self, frame):
        # Convert OpenCV frame to PIL Image
        image = Image.fromarray(frame)

        # Process the image and generate a caption
        inputs = self.processor(images=image, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
        caption_ids = self.model.generate(**inputs)
        caption = self.processor.decode(caption_ids[0], skip_special_tokens=True)
        
        return caption
