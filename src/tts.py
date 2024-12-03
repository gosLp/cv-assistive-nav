import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from threading import Thread

class TextToSpeech:
    def __init__(self, model_name="parler-tts/parler-tts-mini-v1", device="mps", dtype=torch.bfloat16):
        # Initialize device, dtype, and model name
        self.torch_device = device
        self.torch_dtype = dtype
        self.model_name = model_name

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(self.model_name).to(self.torch_device, dtype=self.torch_dtype)

        # Audio configurations
        self.sampling_rate = self.model.audio_encoder.config.sampling_rate
        self.frame_rate = self.model.audio_encoder.config.frame_rate

    def generate_audio(self, text, description, play_steps_in_s=0.5):
        # Tokenize inputs
        inputs = self.tokenizer(description, return_tensors="pt").to(self.torch_device)
        prompt = self.tokenizer(text, return_tensors="pt").to(self.torch_device)

        # Set up streamer
        play_steps = int(self.frame_rate * play_steps_in_s)
        streamer = ParlerTTSStreamer(self.model, device=self.torch_device, play_steps=play_steps)

        # Create generation arguments
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            prompt_input_ids=prompt.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_attention_mask=prompt.attention_mask,
            streamer=streamer,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10,
        )

        # Initialize Thread for generation
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield chunks of audio in real-time
        for new_audio in streamer:
            if new_audio.shape[0] == 0:
                break
            print(f"Sample of length: {round(new_audio.shape[0] / self.sampling_rate, 4)} seconds")
            yield self.sampling_rate, new_audio
