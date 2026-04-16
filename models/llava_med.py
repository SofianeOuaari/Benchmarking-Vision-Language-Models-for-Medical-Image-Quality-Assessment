# llava_med.py

from transformers import AutoProcessor, LlavaForConditionalGeneration
from .base_model import BaseVLM
from conf import get_model_loading_config


class LlavaMedVLM(BaseVLM):
    # Class Function: Initializing model and processor
    def __init__(self, name, device):
        print(f"[LLaVA-Med] Loading model: {name}")

        # Processor
        processor = AutoProcessor.from_pretrained(name)

        # Model loading config 
        model_loading_config = get_model_loading_config()

        # Load model
        model = LlavaForConditionalGeneration.from_pretrained(
            name,
            **model_loading_config,
        ).eval()

        self.prompt_len = 0

        super().__init__(model, processor, device)

    def processor_function(self, batch):
        # Init batch
        image = batch["image"]
        question = batch["prompt"]

        # Manually mimic chat template, bc otherwise need to upgrade the transformer version
        bos = self.processor.tokenizer.bos_token or ""

        # Prepares Inputs
        prompt = f"{bos}[INST] <image>\n{question} [/INST]"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        self.prompt_len = inputs["input_ids"].shape[1]

        return inputs
    
    # Class Function: Decoding outputs
    def decode(self, outputs):
        return self.processor.decode(outputs[0][self.prompt_len:], skip_special_tokens=True).strip()