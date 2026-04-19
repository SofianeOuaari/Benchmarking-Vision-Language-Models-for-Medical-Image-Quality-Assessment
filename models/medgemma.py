# medgemma.py

# Imports
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base_model import BaseVLM
from conf import get_model_loading_config
import os


# Class: MedGemmaVLM
class MedGemmaVLM(BaseVLM): # !Only working with bfloat16! 
    
    # Initializer
    def __init__(self, name, device):
        print(f"[MedGemma] Loading model: {name}")

        # Read HF-Token from env (exported in SBATCH)
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token is None:
            raise RuntimeError(
                "No HUGGINGFACE_HUB_TOKEN/HF_TOKEN found in environment. "
                "Please set an HF token with access to the gated model."
            )
        
        # Loading processor 
        processor = AutoProcessor.from_pretrained(name, token=token)

        # Init model loading config
        model_loading_config = get_model_loading_config()

        # Load model
        model = AutoModelForImageTextToText.from_pretrained(
            name,
            **model_loading_config,
        ).eval()

        # Ensure pad token id exists (fallback to eos if missing)
        if getattr(model.generation_config, "pad_token_id", None) is None:
            eos_id = getattr(model.generation_config, "eos_token_id", None)
            if eos_id is not None:
                model.generation_config.pad_token_id = eos_id

        super().__init__(model, processor, device)
        self._last_input_len = None
    
    # Class Function: Receiving batch and prepares Inputs
    def processor_function(self, batch):
        
        # Init batch
        image = batch["image"].convert("RGB")
        question = batch["prompt"]

        # Creating input Conversation
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert clinician."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image},
                ],
            },
        ]
        
        # Prepares Inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)

        # Track input length so we can slice it off after generation
        self._last_input_len = inputs["input_ids"].shape[-1]

        return inputs

    # Class Function (Outputs -> Sting): Decoding outputs
    def decode(self, outputs):
         ids = outputs[0][self._last_input_len:]
         decoded = self.processor.decode(ids, skip_special_tokens=True)
         return decoded