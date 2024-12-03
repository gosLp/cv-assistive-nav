import cv2
from ultralytics import YOLO
from src.detection import detect_objects
from src.captioning import SceneCaptioner
from src.guidance import NavigationGuide
from src.tts import TextToSpeech 
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio
import os
import time

# Load the OpenAI API key from the .env file
load_dotenv()
api_key = os.getenv("API_KEY")


# validate api key
if not api_key:
    raise ValueError("API key is not set. Please set your API key in the .env file.")

client = AsyncOpenAI(api_key=api_key)

class NavigationAssistant:
    def __init__(self):
        # Start a conversation with the GPT model (continuous context)
        self.conversation_history = []

    async def refine_guidance(self, scene_description, rule_based_guidance):
        # Create a new prompt with context
        prompt = f"""
        You are a navigation assistant for a visually impaired pedestrian. Based on the following information, provide concise guidance for safe movement:

        Scene Description: {scene_description}
        Initial Guidance: {rule_based_guidance}

        Make sure to keep the guidance simple and actionable, focusing only on hazard avoidance and safe movement.
        Provide it in the style of short commands, such as "Move to the left to avoid the person ahead."

        Guidance:
        """

        # Append to conversation history for continuous context
        self.conversation_history.append({"role": "user", "content": prompt})

        # Generate response using GPT-4
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation_history,
            max_tokens=50,
            temperature=0.7,
        )

        # Extract the response
        guidance = response.choices[0].message.content
        # Add response to conversation history to maintain context
        self.conversation_history.append({"role": "assistant", "content": guidance})

        return guidance

def main():
    # Initialize YOLO model and video capture
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    # Initialize the scene captioner, navigation guide, and navigation assistant
    scene_captioner = SceneCaptioner()
    navigation_guide = NavigationGuide()
    navigation_assistant = NavigationAssistant()
    tts = TextToSpeech()

    frame_count = 0
    N = 20  # Only process every Nth frame for efficiency

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video capture")
                break

            frame_count += 1

            # Detect objects every frame (useful for immediate navigation feedback)
            detected_objects = detect_objects(frame, model)

            # Only process every Nth frame for BLIP captioning and LLM guidance refinement
            if frame_count % N == 0:
                # Generate a scene caption using the frame (skip frames to reduce load)
                scene_caption = scene_captioner.generate_caption(frame)

                # Generate rule-based guidance based on detected objects
                rule_based_guidance = ""
                if detected_objects:
                    for obj in detected_objects:
                        if obj['label'] == "person" and obj['confidence'] > 0.5:
                            rule_based_guidance += "A person is ahead. Move cautiously. "
                        elif obj['label'] == "car" and obj['confidence'] > 0.5:
                            rule_based_guidance += "A car is nearby. Stay on the sidewalk. "

                if not rule_based_guidance:
                    rule_based_guidance = "The path seems clear. You may proceed."

                # Use GPT-4 to refine the guidance with continuous conversation context
                refined_guidance = asyncio.run(navigation_assistant.refine_guidance(scene_caption, rule_based_guidance))
                timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time()))
                print(f"Timestamp: {timestamp} | Guidance: {refined_guidance}")

                # generate audio from the refined guidance using tts model
                for (sampling_rate, audio_chunk) in tts.generate_audio(refined_guidance, scene_caption):
                    # Play the audio chunk
                    print(f"Generated audio chunk with shape: {audio_chunk.shape}")
                    # Add audio playback functionality here

            # Display the annotated frame with detected objects (optional for debugging)
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

    

#     # Initialize the scene captioner and navigation guide
#     scene_captioner = SceneCaptioner()
#     navigation_guide = NavigationGuide()

#     last_guidance_time = time.time()

#     try:
#         while True:
#             # Detect objects and get the frame
#             detected_objects, frame = detect_objects(cap, model)

#             # Generate a scene caption using the frame
#             if frame is not None:
#                 scene_caption = scene_captioner.generate_caption(frame)
#                 if scene_caption:
#                     print(f"Scene Description: {scene_caption}")

#             # Provide navigation guidance every 2 seconds
#             current_time = time.time()
#             if current_time - last_guidance_time >= 2:
#                 last_guidance_time = current_time
#                 if detected_objects and scene_caption:
#                     guidance = navigation_guide.generate_guidance(detected_objects, scene_caption)
#                     print(f"Guidance: {guidance}")

#             # Display the annotated frame with detected objects
#             if frame is not None:
#                 annotated_frame = model(frame)[0].plot()  # Visual representation
#                 cv2.imshow('YOLOv8 Detection', annotated_frame)

#             # Press 'q' to quit the video feed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
