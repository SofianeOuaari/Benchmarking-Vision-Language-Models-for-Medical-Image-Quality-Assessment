# conf/config.py
from pathlib import Path
import yaml
from typing import Dict, Any
import torch  

# Path to config.yaml in the same folder
CONFIG_PATH = Path(__file__).with_name("config.yaml")

# Function: Load configuration from YAML file
def load_config():
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)

# Function: Get random seed from config
def get_random_seed() -> int:
    cfg = load_config()
    return int(cfg.get("random_seed", 42)) # 42 is just a fallback default if key is missing

# Function: Get model configuration list
def get_model_config() -> list:
    cfg = load_config()
    
    # Get supported models
    model_fams = cfg.get("supported_models", {})
    
    models = list()
    
    for model in model_fams.values():
        models.extend(model)

    return models

# Function: Get logging configuration
def get_logging_config() -> Dict[str, str]:
    cfg = load_config()
    return cfg.get("logging", {}) 

# Function: Get data_path from config
def get_data_path() -> Dict[str, str]:
    cfg = load_config()

    return cfg.get("data_paths", {})

# Function: Get model loading configuration
def get_model_loading_config() -> Dict[str, Any]:
    model_loading_cfg = load_config().get("model_loading", {})
   
    dtype_map = {
        "bfloat16": torch.bfloat16,
        # ...
    }

    # Map torch_dtype string to actual torch dtype
    if "torch_dtype" in model_loading_cfg:
        dtype_str = model_loading_cfg["torch_dtype"].lower()
        if dtype_str in dtype_map:
            model_loading_cfg["torch_dtype"] = dtype_map[dtype_str]
        else:
            raise ValueError(f"Unsupported torch_dtype in config.yaml: {model_loading_cfg['torch_dtype']}")
        
    return model_loading_cfg

