# assessment_vlms/data/medimeta_local.py
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple, Dict, Any, Sequence
import numpy as np
from PIL import Image
from conf import get_data_path

# Root of the downloaded dataset
dir = get_data_path()['images_dir']
DATASET_ROOT = Path(dir)

# Helperfunction: Convert .npz to list of PIL images
def _npz_to_pil_list(npz_path: Path) -> List[Image.Image]:
    
    # Load .npz file
    data = np.load(npz_path, allow_pickle=True)
    # Get image array
    arr = data["images"]

    # Init imgs list
    imgs = []

    # Comvert each to PIL
    for i in range(arr.shape[0]): # Note for me: arr.shape -> (1000, 224, 224, 3)  
        img_np = arr[i]
        img = Image.fromarray(img_np.astype(np.uint8), mode="RGB").convert("RGB")
        imgs.append(img)

    return imgs

# Helperfunction: Get .npz path
def get_npz_path(
    root: Path,
    modality: str,
    split: str,                 # "val" or "test"
    corruption: Optional[str],  # e.g. "brightness", None => clean
    severity: Optional[int]     # 1..5, None => clean
) -> Path:
    """
    Note file structure: 
    - clean:     <root>/<modality>/<split>/clean.npz
    - corrupted: <root>/<modality>/<split>/<corruption>_severity_<k>.npz
    """
    # Building base-path
    base = root / modality / split

    # If clean FIXME NOT WORKING YET:
    if corruption is None or severity is None:
        return base / "clean.npz"
    
    # Else corrupted:
    return base / f"{corruption}_severity_{int(severity)}.npz"

# Helperfunction: List available modalities under root
def list_modalities(root: Path = DATASET_ROOT) -> List[str]:
    return sorted([p.name for p in (root).iterdir() if p.is_dir()])

# Helperfunction: List available corruptions for a modality (file stems before _severity_k.npz)
def list_corruptions(modality: str, split: str = "test", root: Path = DATASET_ROOT) -> List[str]:
    folder = root / modality / split
    corrs = set()
    for p in folder.glob("*_severity_*.npz"):
        stem = p.stem  # e.g. "motion_blur_severity_3"
        corr = stem.rsplit("_severity_", 1)[0]
        corrs.add(corr)
    return sorted(corrs)

# Helperfunction: Extract the severity levels available (Assuming each corruption has same severities!)
def list_severity(modality: str, split: str = "test", root: Path = DATASET_ROOT) -> List[int]:
    folder = root / modality / split
    severities = set()
    for p in folder.glob("*_severity_*.npz"):
        stem = p.stem  
        num = int(stem.rsplit("_severity_", 1)[1])
        severities.add(num)
    return sorted(severities)

# Helperfunction: Load clean images and append to output
def load_clean_images(root, m, split, max_images_per_npz, output):
    # Also add clean image
    clean_npz_path = root / m / split / "clean.npz"  
    
    # Skip if file doesn't exist 
    if not clean_npz_path.exists():
        pass
    else:
        # Load clean images
        imgs = _npz_to_pil_list(clean_npz_path)
        
        for i, img in enumerate(imgs):
            # If max_images_per_npz is set, break if exceeded
            if max_images_per_npz is not None and i >= max_images_per_npz:
                break
            meta = {
                "modality": m,
                "split": split,
                "corruption": "clean",
                "severity": 0,
                "index": i,
            }
            # append to out
            output.append((meta, img))
    return output

# Helperfunction: Load corrupted images and append to output
def load_corrupted_images(root, m, split, c, max_images_per_npz, serv_list, output):
    # Iterate over severities
    for s in serv_list:
    
        # Get npz path
        npz_path = get_npz_path(root, m, split, c, s)
        
        # Skip if file doesn't exist
        if not npz_path.exists():
            print(f"Warning: File not found: {npz_path}")
            continue
        
        # Load images
        imgs = _npz_to_pil_list(npz_path)

        for i, img in enumerate(imgs):
            # If max_images_per_npz is set, break if exceeded
            if max_images_per_npz is not None and i >= max_images_per_npz:
                break
            # Build meta
            meta = {
                "modality": m,
                "split": split,
                "corruption": c,
                "severity": s,
                "index": i,
            }
            # Append img and meta to output
            output.append((meta, img))
    return output

# Function: Sample all images per (modality × corruption x severity)
def sample_modality_corruption_severity(
    split: str = "test",
    severity: Optional[Sequence[int]] = None,     # None => discover all
    modalities: Optional[Sequence[str]] = None,   # None => discover all
    corruptions: Optional[Sequence[str]] = None,  # None => discover all
    max_images_per_npz: Optional[int] = None,     # None => use all images in each .npz
    root: Path = DATASET_ROOT,
) -> List[Tuple[Dict[str, Any], Image.Image]]:
    
    # Declare output
    output: List[Tuple[Dict[str, Any], Image.Image]] = []
    
    # Set modalities
    if modalities is None:
        modalities = list_modalities(root)
    elif isinstance(modalities, str):
        modalities = [modalities]
    else:
        modalities = list(modalities)

     # Set corruption
    ref_m = modalities[0] # !! Assuming all modalities have same corruptions and severities!!

    if corruptions is None:
        corr_list = ["clean"] + list_corruptions(ref_m, split, root)
    elif isinstance(corruptions, str):
        corr_list = [corruptions]
    else:
        corr_list = list(corruptions)

    # Set severity 
    if severity is None:
        serv_list = list_severity(ref_m, split, root)
    elif isinstance(severity, int):
        serv_list = [severity]            
    else:
        serv_list = list(severity) 

    # Iterate over modalities
    for m in modalities:

        # Iterate over corruptions
        for c in corr_list:

            # If clean, load clean images, else load corrupted images
            if c == "clean":
                output = load_clean_images(root, m, split, max_images_per_npz, output)
            else:
                output = load_corrupted_images(root, m, split, c, max_images_per_npz, serv_list, output)  
    
    return output

