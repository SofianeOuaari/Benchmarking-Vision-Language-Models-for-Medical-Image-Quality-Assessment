import os
import logging
from PIL import Image
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
     LlavaForConditionalGeneration,
)
import numpy as np

logger = logging.getLogger(__name__)


class BaseCombinedExtractor:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._load(model_id)

    def _load(self, model_id: str):
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        logger.info(f"Loading processor for {model_id}")
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        except Exception:
            logger.warning("Processor.from_pretrained failed with token; retrying without token")
            self.processor = AutoProcessor.from_pretrained(model_id)

        logger.info(f"Loading model {model_id}")
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
        ).eval()
        
        self.model.to(self.device)
        logger.info(f"Model loaded to device {self.device}")

    def apply_template(self, image: Image.Image, prompt_text: str):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert clinician."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image", "image": image}]}
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs

    def extract(self, image: Image.Image, prompt_text: str = "Describe the image briefly.") -> Any:
        inputs = self.apply_template(image, prompt_text)
        inputs = inputs.to("cuda")
        logger.debug("Running forward pass for model %s", self.model_id)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            combined_hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            combined_hidden_states = outputs.last_hidden_state
        else:
            raise RuntimeError("Unable to find hidden states in model outputs")

        if combined_hidden_states.dim() == 3:
            multimodal_embedding_vector = combined_hidden_states.mean(dim=1)
        elif combined_hidden_states.dim() == 2:
            multimodal_embedding_vector = combined_hidden_states
        else:
            raise RuntimeError(f"Unexpected hidden state dimensions: {combined_hidden_states.shape}")

        return multimodal_embedding_vector.to(torch.float32).squeeze(0).cpu().numpy()

   


class MedGemmaCombinedExtractor(BaseCombinedExtractor):
    def _load(self, model_id: str):
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token is None:
            raise RuntimeError(
                "No HUGGINGFACE_HUB_TOKEN/HF_TOKEN found in environment. MedGemma requires a gated HF token."
            )
        logger.info(f"Loading MedGemma processor for {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)

        logger.info(f"Loading MedGemma model {model_id}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        self.model.to(self.device)

    def extract(self, image: Image.Image, prompt_text: str = "Describe the image briefly.") -> Any:
        logger.debug("MedGemma extract for %s", self.model_id)
        return super().extract(image, prompt_text)


class LLaVaCombinedExtractor(BaseCombinedExtractor):
    def _load(self, model_id: str):
        logger.info(f"Loading LlavaNext processor/model for {model_id}")
        # Use Llava-specific classes when available
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).eval()
        self.model.to(self.device)

    def apply_template(self, image: Image.Image, prompt_text: str):
        # Try the standard chat-style tokenization first, fall back to the
        # [INST] style prompt used in some Llava examples.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert clinician."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image", "image": image}]}
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            )
            return inputs
        except Exception:
            bos = getattr(self.processor.tokenizer, "bos_token", "") or ""
            prompt = f"{bos}[INST] <image>\n{prompt_text} [/INST]"
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            return inputs

    def extract(self, image: Image.Image, prompt_text: str = "Describe the image briefly.") -> Any:
        logger.debug("LLaVa extract for %s", self.model_id)
        inputs = self.apply_template(image, prompt_text)
        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            combined_hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            combined_hidden_states = outputs.last_hidden_state
        else:
            raise RuntimeError("Unable to find hidden states in LLaVa outputs")

        pooled = combined_hidden_states.mean(dim=1) if combined_hidden_states.dim() == 3 else combined_hidden_states
        return pooled.squeeze(0).cpu().numpy()


class LLaVaMedCombinedExtractor(BaseCombinedExtractor):
    def _load(self, model_id: str):
        logger.info(f"Loading LlavaNext processor/model for {model_id}")
        # Use Llava-specific classes when available
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).eval()
        self.model.to(self.device)

    def apply_template(self, image: Image.Image, prompt_text: str):
        # Try the standard chat-style tokenization first, fall back to the
        # [INST] style prompt used in some Llava examples.
        bos = self.processor.tokenizer.bos_token or ""

        prompt = f"{bos}[INST] <image>\n{prompt_text} [/INST]"
        try:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to("cuda")
            return inputs
        except Exception:
            bos = getattr(self.processor.tokenizer, "bos_token", "") or ""
            prompt = f"{bos}[INST] <image>\n{prompt_text} [/INST]"
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            return inputs

    def extract(self, image: Image.Image, prompt_text: str = "Describe the image briefly.") -> Any:
        logger.debug("LLaVa Med extract for %s", self.model_id)
        inputs = self.apply_template(image, prompt_text)
        inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            combined_hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            combined_hidden_states = outputs.last_hidden_state
        else:
            raise RuntimeError("Unable to find hidden states in LLaVa outputs")

        pooled = combined_hidden_states.mean(dim=1) if combined_hidden_states.dim() == 3 else combined_hidden_states
        return pooled.squeeze(0).cpu().numpy()

class LingshuCombinedExtractor(BaseCombinedExtractor):
    def _load(self, model_id: str):
        logger.info(f"Loading Lingshu processor/model for {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.model.to(self.device)

    def apply_template(self, image: Image.Image, prompt_text: str):
        # Uses qwen_vl_utils to prepare image/video inputs
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"image": image}, {"text": prompt_text}]}]
        # Match the pattern used in extract_embeddings_qwen.py: return tokenized text and image/video inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        return inputs

    def extract(self, image: Image.Image, prompt_text: str = "Describe the image briefly.") -> Any:
        logger.debug("Lingshu extract for %s", self.model_id)
        return super().extract(image, prompt_text)


class Qwen2_5CombinedExtractor(LingshuCombinedExtractor):
    def _load(self, model_id: str):
        logger.info(f"Loading Qwen2.5 processor/model for {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.model.to(self.device)

    def extract(self, image: Image.Image, prompt_text: str = "Describe the image briefly.") -> Any:
        logger.debug("Qwen2.5 extract for %s", self.model_id)
        return super().extract(image, prompt_text)


class Qwen3CombinedExtractor(Qwen2_5CombinedExtractor):
    def _load(self, model_id: str):
        logger.info(f"Loading Qwen3 processor/model for {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.model.to(self.device)

    def extract(self, image: Image.Image, prompt_text: str = "Describe the image briefly.") -> Any:
        logger.debug("Qwen3 extract for %s", self.model_id)
        return super().extract(image, prompt_text)


class InternVLCombinedExtractor(BaseCombinedExtractor):
    def _load(self, model_id: str):
        logger.info(f"Loading InternVL model and processor for {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        self.model.to(self.device)

    def apply_template(self, image: Image.Image, prompt_text: str):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert clinician."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image", "image": image}]}
        ]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        return inputs

    def extract(self, image: Image.Image, prompt_text: str = "Describe the image briefly.") -> Any:
        logger.debug("InternVL extract for %s", self.model_id)
        return super().extract(image, prompt_text)


def get_extractor_for_model(model_id_or_name: str, device: str = "cuda"):
    key = model_id_or_name.lower()
    if "medgemma" in key:
        return MedGemmaCombinedExtractor(model_id_or_name, device=device)
    if "internvl" in key or "intervl" in key:
        return InternVLCombinedExtractor(model_id_or_name, device=device)
    if "qwen3" in key:
        return Qwen3CombinedExtractor(model_id_or_name, device=device)
    if "qwen2.5" in key:
        return Qwen2_5CombinedExtractor(model_id_or_name, device=device)
    if "llava" in key:
        if "med" in key:
            return LLaVaMedCombinedExtractor(model_id_or_name, device=device)
        else:
            return LLaVaCombinedExtractor(model_id_or_name, device=device)
    if "lingshu" in key:
        return LingshuCombinedExtractor(model_id_or_name, device=device)
    return BaseCombinedExtractor(model_id_or_name, device=device)