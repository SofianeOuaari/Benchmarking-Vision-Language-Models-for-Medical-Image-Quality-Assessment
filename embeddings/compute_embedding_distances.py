from pathlib import Path
import logging
from typing import List, Optional
import numpy as np
import csv
import torch
# Ensure repo root is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conf.config import get_model_config
from data.medimeta_dataloader import list_modalities, list_corruptions
from embeddings.get_all_embeddings import (
    preload_modality_images,
    generate_embeddings_from_store,
)
from prompts.prompts import get_prompt


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def _flatten_corruption_arr(embeddings_dict, corr: str):
    """Return a single array with all samples for corruption `corr` (combine severities).
    Returns None if no embeddings present for that corruption."""
    if corr not in embeddings_dict:
        return None
    sev_map = embeddings_dict[corr]
    arrs = [a for a in (sev_map.get(s) for s in sorted(sev_map.keys())) if a is not None and a.size]
    if not arrs:
        return None
    return np.vstack(arrs)


def main(
    modalities_arg: str = "all",
    models_arg: str = "all",
    split: str = "test",
    num_samples: Optional[int] = None,
    prompt_text: Optional[str] = None,
    device: str = "cuda",
    reuse_existing: bool = False,
    output_dir: Optional[str] = None,
):
    if modalities_arg.lower() == "all":
        modalities = list_modalities()
    else:
        modalities = [m.strip() for m in modalities_arg.split(",") if m.strip()]

    if models_arg.lower() == "all":
        models = get_model_config()
    else:
        models = [m.strip() for m in models_arg.split(",") if m.strip()]

    out_base = Path(output_dir) if output_dir else Path(__file__).parent / "output_combined_per_corruption_distances"
    out_base.mkdir(parents=True, exist_ok=True)

    for modality in modalities:
        logger.info(f"Processing modality {modality}")
        # discover corruption names to use as columns (consistent across models)
        corruption_names = sorted(list_corruptions(modality, split))

        csv_path = out_base / f"{modality}_embedding_distances.csv"
        write_header = not csv_path.exists()

        # Preload images once per modality so we can generate embeddings if needed
        images_store = preload_modality_images(modality, split)
        if not images_store:
            logger.warning(f"No images preloaded for modality {modality}, skipping")
            continue
        prompt_text = f"You're seeing a medical image of {modality}. " + get_prompt("JUST_RATING_PROMPT_5")
        for model_id in models:
            try:
                logger.info(f"Processing model {model_id} for modality {modality}")
                out_dir = out_base / modality / Path(model_id).name
                out_dir.mkdir(parents=True, exist_ok=True)
                embeddings_file = out_dir / f"combined_embeddings_{split}.npz"

                if reuse_existing and embeddings_file.exists():
                    logger.info(f"Reusing embeddings at {embeddings_file}")
                    loaded = np.load(embeddings_file, allow_pickle=True)
                    embeddings_dict = loaded["embeddings_dict"].item()
                else:
                    # generate embeddings for this modality+model
                    from embeddings.combined_embeddings.extractors import get_extractor_for_model
                    extractor = get_extractor_for_model(model_id, device=device)
                    prompt_text_loc = prompt_text or (f"You're seeing a medical image of {modality}. ")
                    embeddings_dict = generate_embeddings_from_store(images_store, extractor, prompt_text_loc, num_samples)
                    np.savez(embeddings_file, embeddings_dict=embeddings_dict)
                    logger.info(f"Saved embeddings to {embeddings_file}")
                    del extractor
                    torch.cuda.empty_cache()

                # compute clean center
                clean_arr = None
                if "clean" in embeddings_dict and 0 in embeddings_dict["clean"]:
                    clean_arr = embeddings_dict["clean"][0]
                else:
                    # try to flatten whatever is present under 'clean'
                    clean_arr = _flatten_corruption_arr(embeddings_dict, "clean")

                if clean_arr is None or clean_arr.size == 0:
                    logger.warning(f"No clean embeddings for modality {modality} model {model_id}; distances will be empty")
                    distances = ["" for _ in corruption_names]
                else:
                    clean_center = np.mean(clean_arr, axis=0)
                    distances = []
                    for corr in corruption_names:
                        corr_arr = _flatten_corruption_arr(embeddings_dict, corr)
                        if corr_arr is None:
                            distances.append("")
                            continue
                        corr_center = np.mean(corr_arr, axis=0)
                        d = float(np.linalg.norm(corr_center - clean_center))
                        distances.append(d)

                # write CSV row
                with open(csv_path, "a", newline="") as csvf:
                    writer = csv.writer(csvf)
                    if write_header:
                        writer.writerow(["model_id"] + corruption_names)
                        write_header = False
                    writer.writerow([Path(model_id).name] + distances)

                logger.info(f"Wrote distances for {modality} / {model_id} to {csv_path}")

            except Exception as e:
                logger.exception(f"Failed while processing {modality} {model_id}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate embedding center distances by corruption for each modality and model")
    parser.add_argument("--modality", type=str, default="all", help="Modality to process or 'all' to run all modalities")
    parser.add_argument("--model-id", type=str, default="all", help="Model id or comma-separated list. Use 'all' to use models from conf/config.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split to use")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of samples per severity (for speed)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output base directory")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to include with the image when extracting combined embeddings")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use, e.g., cuda or cpu")
    parser.add_argument("--reuse-existing", action="store_true", help="If set, reuse existing saved combined embeddings when available")

    args = parser.parse_args()

    main(
        modalities_arg=args.modality,
        models_arg=args.model_id,
        split=args.split,
        num_samples=args.num_samples,
        prompt_text=args.prompt,
        device=args.device,
        reuse_existing=args.reuse_existing,
        output_dir=args.output_dir,
    )
