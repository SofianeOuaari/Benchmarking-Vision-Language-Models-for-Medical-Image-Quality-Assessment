# internvl.py

import site
site.ENABLE_USER_SITE = False
import torch
from transformers import AutoTokenizer, AutoModel, GenerationConfig
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from .base_model import BaseVLM
from conf import get_model_loading_config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Helper Function: Build image transform
def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# Class: InternVLVLM
class InternVLVLM(BaseVLM):
    # Class Function: Initializing model and processor
    def __init__(self, name, device):
        print(f"[InternVL] Loading model: {name}")

        self.device = device
        self.image_size = 448

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            use_fast=False,
        )

        # Init model loading config 
        model_loading_config = get_model_loading_config()

        # Load InternVL chat model
        self.model = AutoModel.from_pretrained(
            name,
            **model_loading_config,
        ).eval()
        
        self.transform = build_transform(self.image_size)
    
    # Class Function: Receiving batch and prepares Inputs
    def processor_function(self, batch):
        image = batch["image"]
        pixel_values = (
            self.transform(image)
            .unsqueeze(0)
            .to(torch.bfloat16)
            .to(self.device)
        )
        
        return pixel_values

    # Class Function: Call model to generate outputs
    def __call__(self, batch, **gen_kwargs):
        pixel_values = self.processor_function(batch)
        prompt = "<image>\n" + batch.get("prompt", "Describe the image.")
        generation_config = dict(
            max_new_tokens=gen_kwargs.get("max_new_tokens", 512),
            do_sample=False,
        )
        return self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config=generation_config,
        )

    # Class Function: Decoding outputs
    def decode(self, outputs):
        return outputs
