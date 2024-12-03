from transformers import pipeline

class NavigationGuide:
    def __init__(self):
        # Load a small, fast LLM (TinyBERT-like) for refining guidance
        self.llm = pipeline('text-generation', model='distilgpt2')

    def generate_guidance(self, detected_objects, scene_caption):
        # Base prompt for the LLM
        base_prompt = "You are a navigation assistant for a visually impaired person. Based on the following scene, refine the guidance to be more natural and helpful."

        # Rule-based filtering to generate initial guidance
        guidance_text = ""

        for obj in detected_objects:
            label = obj['label']
            confidence = obj['confidence']
            if label == "person" and confidence > 0.5:
                guidance_text += "There is a person ahead. Be cautious. "
            elif label == "car" and confidence > 0.5:
                guidance_text += "There is a car nearby. Stay to the side. "
            elif label == "chair" and confidence > 0.5:
                guidance_text += "There is a chair in your path. Walk around it. "

        # If no specific hazards are detected, provide generic guidance
        if not guidance_text:
            guidance_text = "The path seems clear. You may proceed."

        # Use LLM to refine the rule-based guidance
        input_prompt = f"{base_prompt}\n\nScene Description: {scene_caption}\nInitial Guidance: {guidance_text}\nRefined Guidance:"
        
        # Generate the refined response using the LLM
        refined_guidance = self.llm(
            input_prompt,
            max_new_tokens=50,
            num_return_sequences=1,
            pad_token_id=self.llm.tokenizer.eos_token_id
        )[0]['generated_text']

        # Extract the refined guidance portion from the generated text
        refined_guidance = refined_guidance.split("Refined Guidance:")[-1].strip()

        return refined_guidance
