import cv2
import subprocess
from ultralytics import YOLO
from src.detection import detect_objects
from src.captioning import SceneCaptioner
from src.guidance import NavigationGuide
from src.tts import TextToSpeech
from src.depth_estimation import DepthEstimator
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
import torch
import ffmpeg
# from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip


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

    def refine_guidance(self, scene_description, object_context):
        logging.info(f"scene...with context: {scene_description}")
        logging.info(f"Refining guidance with GPT-4...with context: {object_context}")
        try:
            # Create a new prompt with context
            prompt = f"""
            You are a navigation assistant for a visually impaired pedestrian. Based on the following information, provide short, concise guidance for safe movement:

            Scene Description: {scene_description}
            Detected Objects and Distances and distance to center:
            {object_context}


            Make sure to keep the guidance simple and actionable, focusing only on hazard avoidance and safe movement. if the object is not directly in front of the person, avoid causing confusion by providing unnecessary details.
            Provide it in the style of short commands, such as "Move to the left to avoid the person ahead. ". Avoid mentioning people or obstacles that do not require immediate action or is too far away from the path of the person.

            Guidance (max 1-2 commands):
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

def main():
    # test video instead of live feed
    video_file  = "../videos/test.mp4"
    # Initialize YOLO model and video capture
    model = YOLO('yolov8n.pt')
    output_video = "result.mp4"
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    # Initialize the scene captioner, navigation guide, and navigation assistant
    scene_captioner = SceneCaptioner()
    # navigation_guide = NavigationGuide()
    navigation_assistant = NavigationAssistant()
    tts = TextToSpeech()
    depth_estimator = DepthEstimator()

    frames_dir = "./frames"
    os.makedirs(frames_dir, exist_ok=True)
    # Directory for audio files
    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)

    output_file = f"guidance_output_{video_file[-8:]}.txt"

    # Clear or create the output file before running
    with open(output_file, "w") as file:
        file.write("Guidance Output Log\n")
        file.write("===================\n")


    # Start the guidance worker coroutine
    # asyncio.create_task(guidance_worker(guidance_queue, navigation_assistant, tts, output_file))


    frame_count = 0
    N = 30  # Only process every Nth frame for efficiency
    guidance_timestamps = []

    # Executor for running blocking code
    # executor = concurrent.futures.ThreadPoolExecutor()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video capture")
                break

            frame_count += 1
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 # Current time in seconds


            # Detect objects every frame (useful for immediate navigation feedback)
            detected_objects = detect_objects(frame, model)
            logging.info(f"Detected Objects: {detected_objects}")

            # Only process every Nth frame for BLIP captioning and LLM guidance refinement
            if frame_count % N == 0:
                try:
                    logging.info(f"Processing frame {frame_count} at {current_time:.2f} seconds...")
                    depth_estimator.analyze_object_distances(frame, detected_objects)
                    # logging.info(f"Object Distances: {object_distances}")
                    # Generate a scene caption using the frame (skip frames to reduce load)
                    scene_caption = scene_captioner.generate_caption(frame)
                    # Generate rule-based guidance based on detected objects
                    rule_based_guidance = ""
                    if detected_objects:
                        for obj in detected_objects:
                            logging.info(f"Object: {obj}")
                            avg_depth = obj.get('avg_depth', None)
                            logging.info(f"Object: {obj['label']}, Confidence: {obj['confidence']}, Depth: {avg_depth}")
                            direction = obj.get("direction", "unknown")
                            logging.info(f"Object: {obj['label']}, Confidence: {obj['confidence']}, Depth: {avg_depth}, Direction: {direction}")
                            if avg_depth is not None and avg_depth < 200: # Example threshold for close objects
                                if direction == "center":
                                    if obj['label'] == "person" and obj['confidence'] > 0.5:
                                        rule_based_guidance += f"A person is approximately {int(avg_depth)} cm ahead. Move cautiously. "
                                    elif obj['label'] == "car" and obj['confidence'] > 0.5:
                                        rule_based_guidance += f"A car is {int(avg_depth)} cm nearby. Stay on the sidewalk. "
                                    elif obj['label'] == "bike" and obj['confidence'] > 0.5:
                                        rule_based_guidance += f"A bike is approaching on the right at {int(avg_depth)} cm. Keep left. "
                                    elif obj['label'] == "bench" and obj['confidence'] > 0.5:
                                        rule_based_guidance += f"A bench is {int(avg_depth)} cm on your left. "
                                    elif obj['label'] == "dog" and obj['confidence'] > 0.5:
                                        rule_based_guidance += f"A dog is {int(avg_depth)} cm ahead. Proceed with caution. "
                                    else:
                                        rule_based_guidance += f"An object: {obj['label']} is {int(avg_depth)} cm ahead. "
                                # Add more conditions for other obstacles or hazards
                                elif direction == "left" and avg_depth < 200:
                                    rule_based_guidance += f"A {obj['label']} is {int(avg_depth)} cm to your left. "
                                elif direction == "right" and avg_depth < 200:
                                    rule_based_guidance += f"A {obj['label']} is {int(avg_depth)} cm to your right. "
                                else:
                                    rule_based_guidance += f"An object: {obj['label']} is {int(avg_depth)} cm ahead. "
                            elif avg_depth is not None and 200 <= avg_depth < 350:
                                rule_based_guidance += f"An object: {obj['label']} is far ahead. "
                                if obj['confidence'] > 0.5:
                                    if direction == "center":
                                        rule_based_guidance += f"Be aware: {obj['label']} is approximately {int(avg_depth)} cm ahead. "
                                    elif direction == "left":
                                        rule_based_guidance += f"Be aware: {obj['label']} is {int(avg_depth)} cm to your left. "
                                    elif direction == "right":
                                        rule_based_guidance += f"Be aware: {obj['label']} is {int(avg_depth)} cm to your right. "

                    
                    if not rule_based_guidance:
                        rule_based_guidance = "The path seems clear. You may proceed."

                    
                    # Use GPT-4 to refine the guidance with continuous conversation context
                    refined_guidance = navigation_assistant.refine_guidance(scene_caption, rule_based_guidance)
                    # Generate audio using TTS
                    audio_filename = f"{audio_dir}/audio_{video_file[-9:]}_{frame_count}_{current_time}.wav"
                    tts.generate_audio(refined_guidance, audio_filename)
                    guidance_timestamps.append((current_time, audio_filename, refined_guidance))

                    timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time()))
                    guidance_output = f"Timestamp: {timestamp} | Guidance: {refined_guidance}\n"
                    with open(output_file, "a") as file:
                        file.write(guidance_output)
                    
                    # Log the guidance
                    logging.info(guidance_output)
                    # Write the guidance to the output file
                    with open(output_file, "a") as file:
                        file.write(guidance_output)

                    
                except Exception as e:
                    logging.error(f"Error during fram processing main loop: {e}")


            # Display the annotated frame with detected objects (optional for debugging)
            annotated_frame = model(frame)[0].plot()  # Visual representation
            # Save the frame as an image file
            frame_filename = os.path.join(frames_dir, f"{video_file[-8:]}_frame_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, annotated_frame)
            cv2.imshow('YOLOv8 Detection', annotated_frame)

            # Press 'q' to quit the video feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        create_video_from_frames(frames_dir, f"annotated_output_{video_file[-8:]}", video_file)
        cap.release()
        cv2.destroyAllWindows()

        # Usage
        text_overlayed_video_path = f"text_overlayed_video_{video_file[-8:]}"
        # final_output_path = 'final_video.mp4'

        # create_text_overlayed_video(video_path, guidance_timestamps, text_overlayed_video_path)
        merge_audio_with_video(f"annotated_output_{video_file[-8:]}", guidance_timestamps, output_video, text_overlayed_video_path)

        # Merge the annotation video and guidance audio
        # merge_audio_with_video('annotated_output.mp4', guidance_timestamps, output_video)

def create_video_from_frames(frames_dir, output_path, video_file):
    try:
        # use ffmpeg to create a video from the frames
        # the frames must be sequentially named 
        command = [
            "ffmpeg",
            "-framerate", "30",
            "-i", f"{frames_dir}/{video_file[-8:]}_frame_%05d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        subprocess.run(command, check=True)
        logging.info(f"Video created: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error creating video from frames: {e}")

def merge_audio_with_video(video_path, guidance_timestamps, output_path, text_overlayed_video_path):
    try:
        # Step 1: Add Text Overlays to the Video
        # text_overlay_video = "text_overlay_output.mp4"

        filter_complex = []
        for timestamp, _, guidance_text in guidance_timestamps:
            # Escape special characters in guidance text for ffmpeg
            escaped_guidance_text = guidance_text.replace(':', '\\:').replace(',', '\\,').replace('\'', "\\'")
            filter_complex.append(
                f"drawtext=fontfile=/Library/Fonts/Arial.ttf:text='{escaped_guidance_text}':"
                f"fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:"
                f"x=(w-text_w)/2:y=10:enable='between(t,{timestamp},{timestamp + 3})'"
            )

        filter_complex_str = ",".join(filter_complex)

        # Create video with text overlays only
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', filter_complex_str,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            text_overlayed_video_path,
            '-y'
        ]

        logging.info(f"FFmpeg Command for Text Overlay: {' '.join(ffmpeg_command)}")

        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"FFmpeg Error during text overlay: {result.stderr}")
            return False

#         # Step 2: Add Audio Sequentially
#         current_video = text_overlay_video

#         for timestamp, audio_file, _ in guidance_timestamps:
#             # Verify that audio file exists
#             if not os.path.exists(audio_file):
#                 logging.warning(f"Audio file does not exist: {audio_file}")
#                 continue

#             output_with_audio = f"temp_output_{timestamp}.mp4"

#             # Prepare FFmpeg command to add audio with delay
#             ffmpeg_command = [
#                 'ffmpeg',
#                 '-i', current_video,
#                 '-i', audio_file,
#                 '-filter_complex', f"[1:a]adelay={int(timestamp * 1000)}|{int(timestamp * 1000)}:all=1[a];[0:a][a]amix=inputs=2",
#                 '-c:v', 'copy',
#                 '-c:a', 'aac',
#                 '-b:a', '192k',
#                 output_with_audio,
#                 '-y'
#             ]

#             logging.info(f"FFmpeg Command for Adding Audio: {' '.join(ffmpeg_command)}")

#             result = subprocess.run(ffmpeg_command, capture_output=True, text=True)

#             if result.returncode != 0:
#                 logging.error(f"FFmpeg Error during audio merging: {result.stderr}")
#                 return False

#             # Update current video to the output of the last step
#             current_video = output_with_audio

#         # Move final video to output path
#         os.rename(current_video, output_path)
#         logging.info(f"Successfully merged video and audio: {output_path}")
#         return True

    except Exception as e:
        logging.error(f"Unexpected error in merge_audio_with_video: {e}")
        return False



if __name__ == "__main__":
       # Run the main coroutine in the current event loop
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Shutting down gracefully.")

    


