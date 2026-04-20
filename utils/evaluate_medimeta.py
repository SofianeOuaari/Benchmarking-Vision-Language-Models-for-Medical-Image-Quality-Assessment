# evaluate_medimeta.py

# Import necessary libraries
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import torch
import csv
import logging

# Import assessment_vlms modules
from models import load_vlm_model
from prompts import get_prompt
from data import sample_modality_corruption_severity
from utils import get_print_name
from conf import get_model_config, get_data_path, get_logging_config

# Init logging 
logging_info = get_logging_config()    
LOG_LEVEL = logging_info['log_level']
LOG_FORMAT = logging_info['log_format']
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Function: Evaluate MediMeta-C dataset
def evaluate_medimeta(
        split: str = "test",            # test/val
        severities = 3,                 # Severity level or list of levels
        modalities = None,              # or ["fundus", "ct", "mri", ...] to force a subset
        corruptions = None,             # or ["brightness","contrast",...] to force a subst
        max_images_per_npz: int = 2     # Limit to x images per .npz
):
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get model names from config
    model_names = get_model_config()

    # Get data path from config
    data_paths = get_data_path()

    # Logging info
    logger.info("START: Loading MediMeta-C images")
    
    # Load MediMeta-C images
    images = sample_modality_corruption_severity(
        split= split,
        severity = severities,           
        modalities= modalities,        
        corruptions= corruptions,       
        max_images_per_npz = max_images_per_npz      
    )

    # Logging info
    logger.info("END: Loading MediMeta-C images")

    # Create csv file to save outputs
    fields = ["model_name", "modality", "corruption", "severity", "index", "model_output"]
    csv_file_name = data_paths['output_dir']+ "/test_output.csv"

    # Run each model sequentially
    for name in model_names:
        logger.info(f"START: Evaluating model {name}")

        # Load model
        model = load_vlm_model(name, device)

        # Get print name for model
        print_name = get_print_name(name)

        # Check if file already exists
        file_exists = os.path.exists(csv_file_name)

        # Open CSV once and keep Writer
        with open(csv_file_name, "a", newline="", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)

            # Only add Header when file wasn't created yet
            if not file_exists:
                csvwriter.writerow(fields)

            # FOR LOOP: Each model evaluates every Image
            for meta, img in images:

                # Print status update
                print(f"[{name}]: Evaluating: {meta['modality']}, {meta['corruption']}, {meta['severity']}, {meta['index']}")

                # Create batch
                batch = {"image": img, "prompt": f"You are seeing a medical image of {meta['modality']}. Note that {get_prompt('BIASED_INSTITUTION2')} " + get_prompt("JUST_RATING_PROMPT_5")}
               
                # Preprocess and generate
                inputs = model.processor_function(batch)

                # Handle InternVL differently
                if "internvl" in name.lower():
                    outputs = model(batch, max_new_tokens=200)
                else:
                    outputs = model(inputs, max_new_tokens=200)

                # Decode
                decoded_output = model.decode(outputs)

                # Append output to csv content
                row = [
                    print_name,
                    meta['modality'],
                    meta['corruption'],
                    meta['severity'],
                    meta['index'],
                    decoded_output.replace("\n", "")
                ]
                csvwriter.writerow(row)

                # Empty buffer and sync file
                csvfile.flush()
                os.fsync(csvfile.fileno())

            # Free GPU memory between runs
            del model
            torch.cuda.empty_cache()

        # Logging info
        logger.info(f"END: Evaluating model {name}")

    return csv_file_name