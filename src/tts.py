import pyttsx3

class TextToSpeech:
    def __init__(self):
        # Initialize the TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Adjust speaking speed (default is 200)
        self.engine.setProperty('volume', 0.9)  # Set volume level (0.0 to 1.0)

    def generate_audio(self, text):
        # Synthesize speech immediately
        self.engine.say(text)
        self.engine.runAndWait()
