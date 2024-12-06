# Vision-Based Assistive Navigation System

This project is a vision-based assistive navigation tool designed to help visually impaired users navigate their environment safely. Using a combination of object detection, scene captioning, and a language model for guidance, the system provides real-time hazard avoidance instructions through audio prompts.

**Team Members**: Clifford Reeve Menezes, Pratheek Prakash Shetty  
**Course**: Fall 2024 ECE 4554/5554 Computer Vision Course Project, Virginia Tech  
**Project Website**: [Sample Link - To be updated](https://goslp.github.io/cv-assistive-nav/)

## Table of Contents
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Usage Instructions](#usage-instructions)
- [Features](#features)
- [Contact](#contact)

## Installation

To get started, clone the repository and install the required dependencies.

### Step 1: Clone the Repository

```bash
git clone https://github.com/gosLp/cv-assistive-nav.git
cd cv-assistive-nav
```

### Step 2: Set Up Environment
Ensure you are using Python 3.8 or later. It is recommended to create a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

### Step 3: Install Required Packages
Install the required Python dependencies:


```bash
pip install -r requirements.txt
```
## Running the Project
### Step 1: Update main.py
Before running the project, update main.py to either use a pre-recorded video or a live video feed:

1. Using a Video File


Update the line in main.py to specify the path to your video file:

```bash
video_file = "path/to/your/video.mp4"
```

2. Using a Live Video Feed

If you want to use a live camera feed, replace the cv2.VideoCapture line in main.py:


```bash
cap = cv2.VideoCapture(0)  # Use index 0 for default webcam
```

### Step 2: Run the Script
Once you have updated main.py, you can run the script as follows:

```bash
python main.py
```
This will process the video, detect objects, and provide real-time guidance with audio.


## Usage Instructions
Input Source: You can choose either a pre-recorded video file or a live feed from your webcam.
Output: The system will provide guidance instructions through audio generated using text-to-speech.

## Important Notes
Audio Guidance: Audio guidance will play out loud for each detected hazard in the path.

API Key: Make sure to include your OpenAI API key in the .env file: place a .env file in the src/ directory with

```bash
API_KEY=your_openai_api_key
```

This key is used to generate guidance with the language model.

## Features
1. Object Detection: Uses YOLOv8 nano to detect common obstacles such as cars, benches, and pedestrians.
2. Scene Captioning: BLIP model is used to generate contextual descriptions of the scene.
3. Language Model Guidance: ChatGPT-4 is used to provide navigation guidance based on detected objects and scene descriptions.
4. Audio Prompt: Text-to-speech conversion provides real-time audible instructions to assist the user.