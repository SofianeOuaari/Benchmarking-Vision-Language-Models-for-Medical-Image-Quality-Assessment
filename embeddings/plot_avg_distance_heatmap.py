from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def discover_csv_files(input_dir: Path):
    csvs = sorted(input_dir.glob("*.csv"))
    return csvs


def modality_name_from_filename(p: Path) -> str:
    # remove suffixes like '_embedding_distances.csv' or '_embeddings_distances.csv'
    name = p.stem
    # handle possible different stems
    for suffix in ("_embedding_distances", "_embeddings_distances", "_distances"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def read_and_average(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "model_id" in df.columns:
        df = df.drop(columns=["model_id"])
    # all remaining columns should be numeric distances; coerce and take mean ignoring NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.mean(axis=0)


def build_avg_matrix(csv_paths):
    # collect all corruption names for consistent columns
    all_corruptions = []
    averages = {}
    for p in csv_paths:
        avg = read_and_average(p)
        averages[modality_name_from_filename(p)] = avg
        all_corruptions.extend(list(avg.index))

    all_corruptions = list(dict.fromkeys(all_corruptions))  # preserve order

    matrix = pd.DataFrame(index=averages.keys(), columns=all_corruptions, dtype=float)
    for modality, series in averages.items():
        matrix.loc[modality, series.index] = series.values

    return matrix


def plot_heatmap(matrix: pd.DataFrame, output_file: Path, cmap: str = "RdBu"):
    # mask NaNs
    mask = matrix.isna()

    # Make figure larger for readability: scale with number of corruption types and modalities
    plt.figure(figsize=(10, 7))

    # Use center=None (default). Annotate with 2 decimals. Force fmt to .2f where values exist
    sns.set(style="white")

    ax = sns.heatmap(matrix.astype(float), cmap=cmap, mask=mask, annot=True, fmt=".2f",
                     cbar_kws={"label": "Average distance"}, linewidths=0.5)

    ax.set_xlabel("Corruption type")
    ax.set_ylabel("Modality")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Plot average embedding distances heatmap")
    p.add_argument("--input-dir", type=Path, default=Path("output_combined_per_corruption_distances"),
                   help="Directory with the per-modality distance CSV files")
    p.add_argument("--output-file", type=Path, default=Path("output_combined_per_corruption_distances/avg_distance_heatmap.png"),
                   help="Where to save the resulting heatmap PNG")
    p.add_argument("--cmap", type=str, default="coolwarm", help="Matplotlib colormap to use (default: RdBu)")
    return p.parse_args()


def main():
    args = parse_args()
    csvs = discover_csv_files(args.input_dir)
    if not csvs:
        raise SystemExit(f"No CSV files found in {args.input_dir}")

    matrix = build_avg_matrix(csvs)
    # sort modalities alphabetically for consistent display (optional)
    matrix = matrix.sort_index()

    # optionally reorder columns: keep appearance order from files; currently preserved
    plot_heatmap(matrix, args.output_file, cmap=args.cmap)
    print(f"Saved heatmap to {args.output_file}")


if __name__ == "__main__":
    main()
