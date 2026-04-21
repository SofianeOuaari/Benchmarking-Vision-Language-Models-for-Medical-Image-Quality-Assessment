# assessment_vlms/utils/utils.py
import random
import numpy as np
from conf import get_random_seed
import argparse

# Helperfunction: Extracting name from model path
def get_print_name(name: str) -> str:
    # LLava med
    if "llava-med" in name.lower() and "mistral-7b" in name.lower():
        return "LLava-Med-Mistral-7b"
    # LLava
    elif "llava" and "mistral" in name.lower():
        return "LLava-Mistral-7B"
    elif "llava" and "vicuna" in name.lower():
        return "LLava-Vicuna-7B"
    elif "llava" and "llama3" in name.lower():
        return "LLava-Llama3-8B"
    # Qwen 2.5
    elif "qwen2.5-vl-3b" in name.lower():
        return "Qwen2.5-3B"
    elif "qwen2.5-vl-7b" in name.lower():
        return "Qwen2.5-7B"
    elif "qwen2.5-vl-32b" in name.lower():
        return "Qwen2.5-32B"
    # Qwen 3
    elif "qwen3-vl-2b" in name.lower():
        return "Qwen3-2B"
    elif "qwen3-vl-4b" in name.lower():
        return "Qwen3-4B"
    elif "qwen3-vl-8b" in name.lower():
        return "Qwen3-8B"
    elif "qwen3-vl-30b-a3b" in name.lower():
        return "Qwen3-30B"
    # Lingshu
    elif "lingshu-7b" in name.lower():
        return "Lingshu-7B"
    elif "lingshu-32b" in name.lower():
        return "Lingshu-32B"
    # Intern
    elif "internvl3_5-2b" in name.lower():
        return "InternVL-2B"
    elif "internvl3_5-8b" in name.lower():
        return "InternVL-8B"
    elif "internvl3_5-14b" in name.lower():
        return "InternVL-14B"
    # Paligemma
    elif "paligemma2-3b" in name.lower():
        return "PaliGemma-3B"
    elif "paligemma2-10b" in name.lower():
        return "PaliGemma-10B"
    # Medgemma
    elif "medgemma-4b" in name.lower():
        return "MedGemma-4B"
    elif "medgemma-27b" in name.lower():
        return "MedGemma-27B"
    else:
        raise ValueError(f"Unknown model name for print extraction: {name}")
  
# Helperfunction: Check if torch is available
def is_torch_available() -> bool:
    try:
        return True
    except ImportError:
        return False

# Helperfunction: Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        help='Modality type (e.g. "["oct", ...]"  or "oct"). If not set: use all modalities.',
    )

    parser.add_argument(
        "--severity",
        type=str,
        default=None,
        help='Severity of corruptions (e.g. "[2,3]" or "3"). If not set: use all severity levels.',
    )

    parser.add_argument(
        "--max_images_per_npz",
        type=int,
         default=None,
        help="Max images per .npz (e.g. 10). If not set: use all images.",
    )

    parser.add_argument(
        "--corruption",
        type=str,
        default=None,
        help='Corruption types (e.g. "brightness" or "["brightness", "contrast"]"). If not set: use all corruptions.',
    )

    return parser.parse_args()

# Function: Set random seed from config    
def set_seed_from_config():
    seed = get_random_seed()
    set_seed(seed)

# Function: Set random seed for reproducibility  
def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    if is_torch_available():
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
