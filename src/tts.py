# import pyttsx3

# class TextToSpeech:
#     def __init__(self):
#         # Initialize the TTS engine
#         self.engine = pyttsx3.init()
#         self.engine.setProperty('rate', 150)  # Adjust speaking speed (default is 200)
#         self.engine.setProperty('volume', 0.9)  # Set volume level (0.0 to 1.0)

#     def generate_audio(self, text, output_file):
#         # Synthesize speech immediately
#         # self.engine.say(text) # Commented out to avoid immediate speech synthesis
#         self.engine.save_to_file(text, output_file)
#         self.engine.runAndWait()


from gtts import gTTS
import os

class TextToSpeech:
    def __init__(self):
        pass

    def generate_audio(self, text, output_filename):
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_filename)
