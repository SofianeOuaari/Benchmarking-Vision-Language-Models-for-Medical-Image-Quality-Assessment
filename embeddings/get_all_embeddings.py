import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Ensure project root is on the import path so `conf` and `data` modules resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from conf.config import get_model_config
from data.medimeta_dataloader import (
    list_modalities,
    list_corruptions,
    list_serverity,
    get_npz_path,
    DATASET_ROOT,
)
from embeddings.combined_embeddings.extractors import get_extractor_for_model
from prompts.prompts import get_prompt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _parse_list_arg(value: str) -> List[str]:
    if value is None:
        return []
    value = value.strip()
    if value.lower() == "all":
        return ["all"]
    return [v.strip() for v in value.split(",") if v.strip()]
# Helper: load images from npz into PIL list
def _npz_to_pil_list(npz_path: Path) -> List[Image.Image]:
    data = np.load(npz_path, allow_pickle=True)
    arr = data["images"]
    imgs = []
    for i in range(arr.shape[0]):
        img_np = arr[i]
        img = Image.fromarray(img_np.astype(np.uint8), mode="RGB").convert("RGB")
        imgs.append(img)
    return imgs


def preload_modality_images(modality: str, split: str = "test", root: Path = DATASET_ROOT) -> Dict[str, Dict[int, List[Image.Image]]]:
    """Load all images for a modality into memory organized by corruption and severity.

    Returns a dict: { corruption_name -> { severity -> [Image, ...] } }
    """
    corruptions = list_corruptions(modality, split, root)
    severities = list_serverity(modality, split, root)

    images_store: Dict[str, Dict[int, List[Image.Image]]] = {}

    # Load clean
    clean_npz = get_npz_path(root, modality, split, None, None)
    if clean_npz.exists():
        try:
            imgs = _npz_to_pil_list(clean_npz)
            images_store["clean"] = {0: imgs}
            logger.info(f"Loaded clean images for modality {modality}: {len(imgs)} images")
        except Exception as e:
            logger.warning(f"Failed to load clean npz {clean_npz}: {e}")

    for corr in corruptions:
        images_store[corr] = {}
        for sev in severities:
            npz_path = get_npz_path(root, modality, split, corr, sev)
            if not npz_path.exists():
                logger.debug(f"Missing file {npz_path}, skipping")
                continue
            try:
                imgs = _npz_to_pil_list(npz_path)
                images_store[corr][sev] = imgs
                logger.info(f"Loaded {len(imgs)} images for {modality}/{corr}/sev{sev}")
            except Exception as e:
                logger.warning(f"Failed to load npz {npz_path}: {e}")

    return images_store



def generate_embeddings_from_store(images_store: Dict[str, Dict[int, List[Image.Image]]], extractor, prompt_text: str, num_samples: Optional[int] = None) -> Dict[str, Dict[int, np.ndarray]]:
    """Generate embeddings for all (corr, sev) combinations using the given extractor.

    Returns dict: { corruption -> { severity -> np.ndarray(samples, dim) } }
    """
    out: Dict[str, Dict[int, np.ndarray]] = {}
    for corr, sev_map in images_store.items():
        out[corr] = {}
        for sev, imgs in sev_map.items():
            imgs_slice = imgs[:num_samples] if num_samples else imgs
            embs = []
            for img in tqdm(imgs_slice, desc=f"Extracting {corr} sev{sev}", leave=False):
                emb = extractor.extract(img, prompt_text=prompt_text)
                embs.append(emb)
            if embs:
                out[corr][sev] = np.vstack(embs)
    return out

def compute_2d_representation_by_severity(embeddings_dict: Dict[str, Dict[int, np.ndarray]]) -> Dict:
    all_embeddings = []
    labels = []
    for corruption, severity_dict in embeddings_dict.items():
        for severity, emb_array in severity_dict.items():
            all_embeddings.append(emb_array)
            labels.extend([int(severity)] * len(emb_array))
    all_embeddings = np.vstack(all_embeddings)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    return {"coordinates": embeddings_2d, "labels": labels, "pca_explained_variance": pca.explained_variance_ratio_.tolist()}

def compute_2d_representation_by_corruption(embeddings_dict: Dict[str, Dict[int, np.ndarray]]) -> Dict:
    all_embeddings = []
    labels = []
    for corruption, severity_dict in embeddings_dict.items():
        for severity, emb_array in severity_dict.items():
            all_embeddings.append(emb_array)
            labels.extend([corruption] * len(emb_array))
    all_embeddings = np.vstack(all_embeddings)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    return {"coordinates": embeddings_2d, "labels": labels, "pca_explained_variance": pca.explained_variance_ratio_.tolist()}

def plot_2d_embeddings_by_severity(
    embeddings_2d: np.ndarray,
    labels: List[int],
    title: str = "2D Combined Embedding Space (by severity)",
    output_path: Optional[Path] = None,
    ):
    fig = go.Figure()
    
    severity_levels = sorted(set(labels), key=lambda x: (x != 0, x))
    colors = px.colors.sample_colorscale('Bluered', np.linspace(0, 1, len(severity_levels)))
    color_map = {sev: colors[i] for i, sev in enumerate(severity_levels)}
    
    for severity in severity_levels:
        mask = np.array([l == severity for l in labels])
        marker_dict = dict(size=8, color=color_map[severity], opacity=0.7)
        if severity == 0:
            marker_dict['symbol'] = 'star'
            marker_dict['size'] = 12
        fig.add_trace(go.Scatter(
            x=embeddings_2d[mask, 0],
            y=embeddings_2d[mask, 1],
            mode='markers',
            name=f"Severity {severity}",
            marker=marker_dict
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="Severity",
        legend=dict(font=dict(size=14)),
        width=1000,
        height=800
    )
    
    logger.info(f"Trying to save to: {output_path}")
    fig.write_image(str(output_path))
    logger.info(f"Plot saved to {output_path}")


def plot_2d_embeddings_by_corruption(
    embeddings_2d: np.ndarray,
    labels: List[str],
    title: str = "2D Combined Embedding Space (by corruption)",
    output_path: Optional[Path] = None,
):
    fig = go.Figure()
    
    corruption_types = sorted(set(labels), key=lambda x: (x != "clean", x))
    colors = px.colors.qualitative.Plotly[:len(corruption_types)]
    color_map = {corr: colors[i] for i, corr in enumerate(corruption_types)}
    
    for corruption in corruption_types:
        mask = np.array([c == corruption for c in labels])
        marker_dict = dict(size=8, color=color_map[corruption], opacity=0.7)
        if corruption == "clean":
            marker_dict['symbol'] = 'star'
            marker_dict['size'] = 12
        fig.add_trace(go.Scatter(
            x=embeddings_2d[mask, 0],
            y=embeddings_2d[mask, 1],
            mode='markers',
            name=corruption,
            marker=marker_dict
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="Corruption",
        legend=dict(font=dict(size=14)),
        width=1000,
        height=800
    )
    
    logger.info(f"Trying to save to: {output_path}")
    fig.write_image(str(output_path))
    logger.info(f"Plot saved to {output_path}")

def main(
    modalities_arg: str = "all",
    models_arg: str = "all",
    split: str = "test",
    num_samples: int = None,
    prompt_text: str = None,
    device: str = "cuda",
    reuse_existing: bool = False,
    output_dir: str = None,
    plot_by: str = "severity",
):
    # Resolve modalities
    if modalities_arg.lower() == "all":
        modalities = list_modalities()
    else:
        modalities = [m.strip() for m in modalities_arg.split(",") if m.strip()]

    # Resolve models
    if models_arg.lower() == "all":
        models = get_model_config()
    else:
        models = [m.strip() for m in models_arg.split(",") if m.strip()]

    logger.info(f"Running get_all_embeddings for modalities={modalities} models={models} split={split}")


    

    for modality in modalities:
        # Preload images for this modality once (avoid re-reading .npz for every model)
        logger.info(f"Preloading images for modality {modality}")
        images_store = preload_modality_images(modality, split, DATASET_ROOT)
        if not images_store:
            logger.warning(f"No images found for modality {modality}, skipping")
            continue

        prompt_text = f"You're seeing a medical image of {modality}. " + get_prompt("JUST_RATING_PROMPT_5")

        for model_id in models:
            try:
                logger.info(f"Starting run: modality={modality} model={model_id}")

                # Initialize extractor for this model
                extractor = get_extractor_for_model(model_id, device=device)
                model = extractor.model  # Keep reference to free later
                processor = extractor.processor

                # Create per-run output directory
                out_dir = Path(output_dir) if output_dir else Path(__file__).parent / "output_combined_per_corruption"
                out_dir = out_dir / modality / Path(model_id).name
                out_dir.mkdir(parents=True, exist_ok=True)

                # Embeddings file
                embeddings_file = out_dir / f"combined_embeddings_{split}.npz"
                if reuse_existing and embeddings_file.exists():
                    logger.info(f"Reusing existing combined embeddings from {embeddings_file}")
                    loaded = np.load(embeddings_file, allow_pickle=True)
                    embeddings_dict = loaded["embeddings_dict"].item()
                else:
                    # Use preloaded images to generate embeddings (faster and avoids repeated IO)
                    embeddings_dict = generate_embeddings_from_store(images_store, extractor, prompt_text, num_samples)
                    np.savez(embeddings_file, embeddings_dict=embeddings_dict)
                    logger.info(f"Combined embeddings saved to {embeddings_file}")

                # Compute 2D and save
                if plot_by == "severity":
                    result_2d = compute_2d_representation_by_severity(embeddings_dict)
                    results_file = out_dir / f"combined_embeddings_2d_severity_{split}.json"
                    plot_path = out_dir / f"combined_embeddings_2d_severity_{split}.pdf"
                    plot_title = f"{modality.upper()} - {Path(model_id).name} Combined Embeddings (2D PCA by severity)"
                elif plot_by == "corruption":
                    result_2d = compute_2d_representation_by_corruption(embeddings_dict)
                    results_file = out_dir / f"combined_embeddings_2d_corruption_{split}.json"
                    plot_path = out_dir / f"combined_embeddings_2d_corruption_{split}.pdf"
                    plot_title = f"{modality.upper()} - {Path(model_id).name} Combined Embeddings (2D PCA by corruption)"
                else:
                    raise ValueError(f"Invalid plot_by: {plot_by}")
                
                results_to_save = {
                    "coordinates": result_2d["coordinates"].tolist(),
                    "pca_explained_variance": result_2d["pca_explained_variance"],
                    "labels": result_2d["labels"],
                }
                with open(results_file, "w") as f:
                    json.dump(results_to_save, f)
                logger.info(f"2D results saved to {results_file}")

                # Plot
                if plot_by == "severity":
                    plot_2d_embeddings_by_severity(result_2d["coordinates"], result_2d["labels"], title=plot_title, output_path=plot_path)
                elif plot_by == "corruption":
                    plot_2d_embeddings_by_corruption(result_2d["coordinates"], result_2d["labels"], title=plot_title, output_path=plot_path)

                logger.info(f"Completed successfully: modality={modality} model={model_id}")
                # Free GPU memory between runs
                del model
                del processor
                del extractor
                del result_2d
                torch.cuda.empty_cache()
                print("------- Freed GPU memory DONE -------")
            except Exception as e:
                logger.exception(f"Failed for modality={modality} model={model_id}: {e}")
                
            

if __name__ == "__main__":
    import argparse

    

    parser = argparse.ArgumentParser(description="Orchestrate combined embeddings across many models and modalities")
    parser.add_argument("--modality", type=str, default="all", help="Modality to process or 'all' to run all modalities")
    parser.add_argument("--model-id", type=str, default="all", help="Model id or comma-separated list. Use 'all' to use models from conf/config.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split to use")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of samples per severity")
    parser.add_argument("--output-dir", type=str, default=None, help="Base output directory (overrides default)")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to include with the image when extracting combined embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use, e.g., cuda or cpu")
    parser.add_argument("--reuse-existing", action="store_true", help="If set, reuse existing saved combined embeddings when available")
    parser.add_argument("--plot-by", type=str, default="corruption", choices=["severity", "corruption"], help="Plot by severity levels or corruption types")

    args = parser.parse_args()
    pio.get_chrome()

    main(
        modalities_arg=args.modality,
        models_arg=args.model_id,
        split=args.split,
        num_samples=args.num_samples,
        prompt_text=args.prompt,
        device=args.device,
        reuse_existing=args.reuse_existing,
        output_dir=args.output_dir,
        plot_by=args.plot_by,
    )