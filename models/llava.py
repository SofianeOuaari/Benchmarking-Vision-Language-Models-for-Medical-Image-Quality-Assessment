# llava.py

# Imports
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from .base_model import BaseVLM

from conf import get_model_loading_config

# Class: LlavaVLM
class LlavaVLM(BaseVLM):
     # Class Function: Initializing model and processor
     def __init__(self, name, device):
        print(f"[LLaVA] Loading quantized model: {name}")

        # Loading processor
        processor = LlavaNextProcessor.from_pretrained(name)

        # Init model loading config
        model_loading_config = get_model_loading_config()

        # Load model
        model = LlavaNextForConditionalGeneration.from_pretrained(
            name,
            **model_loading_config,
        ).eval()

        self.prompt_len = 0

        super().__init__(model, processor, device)

    # Class Function: Receiving batch and prepares Inputs
     def processor_function(self, batch):
        
        # Init batch
        image = batch["image"]
        question = batch["prompt"]
        
        # Creating input Conversation
        conversation = [
            {"role": "user", "content": 
             [
                {"type": "text", "text": question},
                {"type": "image"},]
            },
        ]

        # Prepares Inputs
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        self.prompt_len = inputs["input_ids"].shape[1]
        return inputs

    # Class Function: Decoding outputs
     def decode(self, outputs):
        return self.processor.decode(outputs[0][self.prompt_len:], skip_special_tokens=True).strip()