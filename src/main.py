import cv2
from ultralytics import YOLO
from src.detection import detect_objects
from src.captioning import SceneCaptioner
from src.guidance import NavigationGuide
from src.tts import TextToSpeech 
from dotenv import load_dotenv
from openai import OpenAI
import simpleaudio as sa  # Lightweight audio playback module
import asyncio
import os
import time
import concurrent.futures  # Import added
from queue import Queue, Empty 
import logging
from threading import Thread

# Setup logging to help with debugging
logging.basicConfig(level=logging.INFO)

# Load the OpenAI API key from the .env file
load_dotenv()
api_key = os.getenv("API_KEY")


# validate api key
if not api_key:
    raise ValueError("API key is not set. Please set your API key in the .env file.")

# client = OpenAI(api_key=api_key)

class NavigationAssistant:
    def __init__(self):
        self.client = OpenAI(api_key=api_key)
        # Start a conversation with the GPT model (continuous context)
        self.conversation_history = []

    def refine_guidance(self, scene_description, rule_based_guidance):
        try:
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
            response = self.client.chat.completions.create(
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
        except Exception as e:
            logging.error(f"Error in refine_guidance: {e}")
            return "I'm sorry, I couldn't process the guidance at the moment. Please try again later."

# async def guidance_worker(queue, assistant, tts, output_file):
#     """Worker function that processes the guidance queue"""
#     while True:
#         try:
#             # Wait to get an item from the queue
#             scene_caption, rule_based_guidance = await queue.get()

#             # Log processing information
#             logging.info("Processing guidance from queue.")

#             # Refine the guidance with GPT-4
#             refined_guidance = await assistant.refine_guidance(scene_caption, rule_based_guidance)
#             timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time()))
#             guidance_output = f"Timestamp: {timestamp} | Guidance: {refined_guidance}\n"
#             print(f"Timestamp: {timestamp} | Guidance: {refined_guidance}")

#             # Write the guidance to the output file
#             with open(output_file, "a") as file:
#                 file.write(guidance_output)

#             # Generate audio using TTS
#             await asyncio.get_event_loop().run_in_executor(None, tts.generate_audio, refined_guidance)

#             # Mark the task as done
#             queue.task_done()

#         except Empty:
#             # If queue is empty, just keep looping
#             logging.info("No guidance in the queue, waiting for more tasks...")
#             continue
#         except Exception as e:
#             logging.error(f"Error in guidance_worker: {e}")

def main():
    # test video instead of live feed
    video_file  = "../videos/test.mp4"
    # Initialize YOLO model and video capture
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    # Initialize the scene captioner, navigation guide, and navigation assistant
    scene_captioner = SceneCaptioner()
    navigation_guide = NavigationGuide()
    navigation_assistant = NavigationAssistant()
    tts = TextToSpeech()

    # guidance_queue = asyncio.Queue()
    output_file = "guidance_output.txt"

    # Clear or create the output file before running
    with open(output_file, "w") as file:
        file.write("Guidance Output Log\n")
        file.write("===================\n")


    # Start the guidance worker coroutine
    # asyncio.create_task(guidance_worker(guidance_queue, navigation_assistant, tts, output_file))


    frame_count = 0
    N = 20  # Only process every Nth frame for efficiency

    # Executor for running blocking code
    # executor = concurrent.futures.ThreadPoolExecutor()

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
                try:
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
                    refined_guidance = navigation_assistant.refine_guidance(scene_caption, rule_based_guidance)
                    timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time()))
                    guidance_output = f"Timestamp: {timestamp} | Guidance: {refined_guidance}\n"
                    print(guidance_output)
                    # Write the guidance to the output file
                    with open(output_file, "a") as file:
                        file.write(guidance_output)

                    # Generate audio using TTS
                    tts.generate_audio(refined_guidance)
                except Exception as e:
                    logging.error(f"Error during fram processing main loop: {e}")

                # # Use GPT-4 to refine the guidance with continuous conversation context
                # refined_guidance = await navigation_assistant.refine_guidance(scene_caption, rule_based_guidance)
                # timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time()))
                # print(f"Timestamp: {timestamp} | Guidance: {refined_guidance}")

                # Generate audio using TTS asynchronously to prevent blocking
                # asyncio.get_event_loop().run_in_executor(executor, tts.generate_audio, refined_guidance)

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
       # Run the main coroutine in the current event loop
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Shutting down gracefully.")

    


