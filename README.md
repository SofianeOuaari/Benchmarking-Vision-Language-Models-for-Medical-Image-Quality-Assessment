# Assessing VLM Reliability for Medical Image Quality Evaluation Under Corruption and Bias

We present a benchmark using the MediMeta-C dataset to evaluate how VLMs assess medical image quality in a zero-shot setting, across seven corruption types and five severity levels. The study investigates three dimensions: VLM sensitivity to specific image degradation patterns; how corruptions alter embedding space geometry and its relationship to quality perception; and how textual biases related to demographics, expertise, infrastructure, and institution inappropriately influence quality scores. By characterizing these vulnerabilities and establishing baselines across medical imaging modalities, we identify key challenges for safe clinical deployment of VLMs in automated quality assessment.

![VLM as a judge — unbiased](figures/overall_vlm_as_a_judge_framework_no_bias.png)
![VLM as a judge — biased](figures/overall_vlm_as_a_judge_framework_with_bias.png)
![Scores on clean images (radar)](figures/scores_clean_radar_chart.png)
![Average scores per corruption](figures/average_per_corruption.png)

---

The pipeline evaluates multiple VLMs on MediMeta-C across modalities, corruptions, and severity levels. Raw model outputs are saved to CSV; utilities handle score parsing, plotting, and PDF export.

## Setup

### Environment

```bash
pip install -r requirements.txt
```

### Hugging Face access

Some models require authentication:

```bash
huggingface-cli login
export HUGGINGFACE_HUB_TOKEN=...
```

## Configuration

All configuration lives in `conf/config.yaml`:

- `random_seed`
- model loading options (`device_map`, `torch_dtype`, `trust_remote_code`)
- `supported_models` — list of checkpoints to run
- `data_paths`: `images_dir` (`data/MediMeta-C`), `output_dir` (`outputs/`), `analysis_dir` (`analysis/`)
- logging: `log_level`, `log_format`

## Dataset

Download instructions: https://huggingface.co/datasets/razaimam45/MediMeta-C

MediMeta-C `.npz` files are expected at:

- Clean: `<data_root>/<modality>/<split>/clean.npz`
- Corrupted: `<data_root>/<modality>/<split>/<corruption>_severity_<k>.npz`

Example:

```
data/MediMeta-C/oct/test/clean.npz
data/MediMeta-C/oct/test/brightness_severity_2.npz
```

## Running evaluations

Entry point is `main.py`, which calls `evaluate_medimeta(...)`.

```bash
python main.py \
  --modality "oct" \
  --corruption "['clean','brightness']" \
  --severity "2" \
  --max_images_per_npz 10
```

Omitting any flag causes the script to run over all available values for that option.

**Modalities:** `aml`, `fundus`, `mammo_calc`, `mammo_mass`, `oct`, `pbc`, `pneumonia`

**Corruptions:** `brightness`, `contrast`, `gaussian_noise`, `impulse_noise`, `motion_blur`, `pixelate`, `zoom_blur`

**Severity:** `1`–`5`

Note: Qwen-VL may require newer GPUs. The core evaluation logic is in `evaluate_medimeta.py`.

## Prompts

Prompts are defined in `prompts/prompts.py`. Each prompt has three parts:

1. Context — brief description of the modality
2. Bias (optional) — demographic, institutional, or expertise framing; remove this part for unbiased evaluation
3. Rating instruction — asks the model to score image quality

Example:

```python
"You are seeing a medical image of {meta['modality']}. Note that {get_prompt('BIASED_INSTITUTION2')} "
+ get_prompt("JUST_RATING_PROMPT_5")
```

To change prompts, update the prompt construction inside `evaluate_medimeta`.

## Output

Results are saved to `outputs/test_output.csv` with columns:

```
model_name, modality, corruption, severity, index, model_output
```

## Embeddings

Extract embeddings across modalities and models:

```bash
# All modalities and all supported models
python embeddings/get_all_embeddings.py --modality all --model-id all --reuse-existing

# Single modality, specific models
python embeddings/get_all_embeddings.py --modality aml --model-id google/medgemma-4b-it,chaoyinshe/llava-med-v1.5-mistral-7b-hf
```

Compute distances between corruption cluster centers and the clean center:

```bash
python embeddings/compute_embedding_distances.py --modality aml --model-id all --num-samples 100 --device cpu --reuse-existing
```

This produces per-modality CSVs (`{modality}_embedding_distances.csv`) with Euclidean distances between each corruption mean and the clean mean embedding.

## Analysis and plots

Analysis functions live in `utils/analysis.py`. They load the CSV, parse a numeric score from the model output (`x/5`), generate a plot, and save it as a PDF to `analysis/`.

To run analysis without re-running evaluation, comment out `evaluate_medimeta` in `main.py` and call the desired function directly.

```python
plot_vlm_corruption_heatmaps(Path("outputs/test_output.csv"))
```

Available functions:

- `scores_clean_radar_chart_swapped(csv_path)`
- `scores_corruption_radar_chart(csv_path)`
- `scores_corruption_radar_chart_per_severity(csv_path)`
- `plot_vlm_corruption_heatmaps(csv_path)`
- `plot_avg_corruption_heatmap_all_models(csv_path)`
- `plot_colored_vs_grey(csv_path)`
- `plot_medical_vs_nonmedical(csv_path)`
- `exploratory_data_analysis(dataset_root)`
