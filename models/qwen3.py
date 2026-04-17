# qwen.py

# Imports
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base_model import BaseVLM
import torch
from conf import get_model_loading_config

# Class: QwenVLM3
class Qwen3VLM(BaseVLM):
     # Class Function: Initializing model and processor
    def __init__(self, name, device):
        print(f"[Qwen3-VL] Loading model: {name}")

        # Loading processor
        self.processor = AutoProcessor.from_pretrained(name)

        # Init model loading config
        model_loading_config = get_model_loading_config()

        # Diffrentiate between MoE vs. Dense
        if "30B-A3B" in name:
            ModelClass = Qwen3VLMoeForConditionalGeneration
        else:
            ModelClass = Qwen3VLForConditionalGeneration

        # Load model
        self.model = ModelClass.from_pretrained(
            name,
            **model_loading_config,
        ).eval()

        # Set device and pad token
        self.device = device
        self.model.generation_config.pad_token_id = (
            self.model.generation_config.pad_token_id or self.model.config.eos_token_id
        )

        self.prompt_len = 0

        # Qwen3-VL docs recommend left padding for batch generation
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    # Class Function: Receiving batch and prepares Inputs
    def processor_function(self, batch):
        
        # Init batch
        image = batch["image"]
        question = batch["prompt"]
        
        # Creating input Conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16,)
       
        # Preparing Inputs 
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        )

        self.prompt_len = inputs["input_ids"].shape[1]

        # Shift inputs to correct device
        embed_device = self.model.get_input_embeddings().weight.device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(embed_device)

        return inputs

    # Class Function: Decoding outputs
    def decode(self, outputs):
        return self.processor.decode(outputs[0][self.prompt_len:], skip_special_tokens=True).strip()