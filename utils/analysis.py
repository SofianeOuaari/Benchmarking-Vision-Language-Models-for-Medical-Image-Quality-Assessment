# utils/analysis.py

from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from conf import get_data_path

# Global variables
COLORED_MODALITIES = {"pbc", "fundus", "aml"}
GREY_MODALITIES = {"oct", "pneumonia", "mammo_mass", "mammo_calc"}
MEDICAL_VLMS = {"MedGemma-27B", "MedGemma-4B", "LLava-Med-Mistral-7b", "Lingshu-7B", "Lingshu-32B"}

# Load dir from config
dir = get_data_path()['analysis_dir']
OUT_DIR = Path(dir)

# Helperfunction: Extract score from output
def extract_score(text: str) -> float | None:

    if pd.isna(text):
        return None

    text = str(text)
    match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(10|5)', text)
    if match:
        return float(match.group(1))
    
    return None

# Helperfunction: Generate a grouped df with average scores
def average_score(df: pd.DataFrame) -> pd.DataFrame:
    
    # Group by model_name, modality, corruption, severity and calculate average score
    grouped_avgScore_df = (
        df.groupby(["model_name", "modality", "corruption", "severity"], as_index=False)["score"]
          .mean()
          .rename(columns={"score": "avg_score"}))
    
    return grouped_avgScore_df

# Helperfunction: Load CSV and add scores
def load_with_scores(csv_path: Path) -> pd.DataFrame:
    
    df = pd.read_csv(csv_path)
    df["score"] = df["model_output"].apply(extract_score)
    df = df.dropna(subset=["score"])
    return df

# Helperfunction: Assign color group
def assign_color_group(modality: str) -> str | None:
    if pd.isna(modality):
        return None
    m = str(modality).lower()
    if m in COLORED_MODALITIES:
        return "colored"
    if m in GREY_MODALITIES:
        return "grey"
    return None 

# Helperfunction: Merge all modality-csv-files to one csv file
def merge_csvs(csv_paths: list[Path], out_path: Path) -> None:
    df = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

# Function: Same as scores_clean_radar_chart but vlms and modalities swapped places
def scores_clean_radar_chart_swapped(csv_path: Path) -> None:
    
    # Load data with scores
    df = load_with_scores(csv_path)

    # Filter to clean corruption type
    clean_df = df[df["corruption"] == "clean"]

    # List of VLMs
    models = clean_df["model_name"].unique().tolist()

    # Compute average score per (model_name, modality) for clean only
    grouped = (
        clean_df
        .groupby(["model_name", "modality"], as_index=False)["score"]
        .mean()
    )

     # Pivot so each model_name is a row, each modality is a column
    table = grouped.pivot(index="modality", columns="model_name", values="score")

    # Ensure all categories exist as columns and in correct order
    table = table.reindex(columns=models)
    table = table.fillna(0)

    # Build radar chart
    fig = go.Figure()

    
    for modality, row in table.iterrows():
        r = row.values.tolist()
        r = r + [r[0]]  

        theta = models
        theta = theta + [theta[0]]

        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill=None,
            mode="lines+markers",
            name=modality
        ))

    fig.update_layout(
        title="Average clean score per VLM (Expertise bias 2)",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=True
    )

    # Show plot
    #fig.show()

    # Saving plots to local dir
    fig.write_image(OUT_DIR/"scores_clean_radar_chart_bias2.pdf")

# Function: Compute mean and standard error of the mean for clean images per model per modality
def _compute_statistics_clean_images(csv_path: Path) -> pd.DataFrame:
    
    # Load data with scores
    df = load_with_scores(csv_path)

    # Filter to clean corruption type
    clean_df = df[df["corruption"] == "clean"]

    # Compute mean and sem per (model_name, modality)
    grouped = (
        clean_df.groupby(["model_name", "modality"])["score"]
        .agg(['mean', 'sem'])
        .reset_index()
        .rename(columns={'mean': 'mean_score', 'sem': 'sem_score'})
    )

    # Save to CSV
    grouped.to_csv(OUT_DIR / "statistics_clean_images.csv", index=False)

    return grouped

# Function: Plot colored vs grey modalities (Based on clean images)
def plot_colored_vs_grey(csv_path: Path) -> None:
    
    # Load data with scores
    df = load_with_scores(csv_path)
    clean_df = df[df["corruption"] == "clean"].copy()

    # Assign color group
    clean_df["color_group"] = clean_df["modality"].apply(assign_color_group)
    clean_df = clean_df.dropna(subset=["color_group"])

    # Defining colors
    color_map = {
    "colored": "#a70000",
    "grey": "#525252",
}

    # --- 1st Overall colored vs grey ---
    overall = (
        clean_df.groupby("color_group", as_index=False)["score"]
          .mean()
          .rename(columns={"score": "avg_score"})
    )

    fig_overall = go.Figure()
    fig_overall.add_trace(
        go.Bar(
            x=overall["color_group"],
            y=overall["avg_score"],
            name="Overall",
            marker_color=overall["color_group"].map(color_map),
        )
    )
    fig_overall.update_layout(
        title="Overall evaluation on clean imgs: colored vs grey modalities (all VLMs)",
        xaxis_title="Modality group",
        yaxis_title="Average score (1-5)",
         yaxis=dict(
                range=[1, 5],
                tickmode="array",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["1", "2", "3", "4", "5"],
        )
    )
    # Show plot
    #fig_overall.show()

    # Saving plots to local dir
    fig_overall.write_image(OUT_DIR/"colored_vs_grey_overall.pdf")

    # --- 2nd Per VLM colored vs grey ---
    per_vlm = (
        clean_df.groupby(["model_name", "color_group"], as_index=False)["score"]
          .mean()
          .rename(columns={"score": "avg_score"})
    )

    pivot = per_vlm.pivot(index="model_name", columns="color_group", values="avg_score")

    fig_vlm = go.Figure()
    if "colored" in pivot.columns:
        fig_vlm.add_trace(
            go.Bar(
                name="Colored modalities",
                x=pivot.index,
                y=pivot["colored"],
                marker_color=color_map["colored"],
            )
        )
    if "grey" in pivot.columns:
        fig_vlm.add_trace(
            go.Bar(
                name="Grey modalities",
                x=pivot.index,
                y=pivot["grey"],
                marker_color=color_map["grey"],
            )
        )

    fig_vlm.update_layout(
        title="Colored vs grey modalities per VLM (Based on clean images)",
        xaxis_title="VLM",
        yaxis_title="Average score (1-5)",
        barmode="group",
        yaxis=dict(
                range=[1, 5],
                tickmode="array",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["1", "2", "3", "4", "5"],
        )
    )
    # Show plot
    #fig_vlm.show()

    # Saving plots to local dir
    fig_vlm.write_image(OUT_DIR/"colored_vs_grey_perVLM.pdf")

# Function: Plotting accumulated scores of all medical VLMs vs non-medical VLMs (Based on clean images)
def plot_medical_vs_nonmedical(csv_path: Path) -> None:
    
     # Load data and add scores
    df = load_with_scores(csv_path)
    clean_df = df[df["corruption"] == "clean"].copy()

    # Set VLM types
    clean_df["vlm_type"] = np.where(
        clean_df["model_name"].isin(MEDICAL_VLMS),
        "medical",
        "non-medical",
    )

    # Define colors for vlm_type
    type_color_map = {
        "medical": "#a70000",
        "non-medical": "#525252",
    }

    grouped = (
        clean_df.groupby("vlm_type", as_index=False)["score"]
          .mean()
          .rename(columns={"score": "avg_score"})
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["vlm_type"],
            y=grouped["avg_score"],
            marker_color=grouped["vlm_type"].map(type_color_map),
        )
    )
    fig.update_layout(
        title="Overall evaluation on clean images: medical vs non-medical VLMs",
        xaxis_title="VLM type",
        yaxis_title="Average score (1-5)",
        yaxis=dict(
        range=[1, 5],
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["1", "2", "3", "4", "5"],
        ),
    )
    # Show plot
    #fig.show()

    # Saving plots to local dir
    fig.write_image(OUT_DIR/"medical_vs_non-med.pdf")

# Function: Plot the average scores generated by two different prompts
def plot_standartPrompt_vs_contextPrompt(standartP_csv_path: Path, adaptedP_csv_path: Path):
    
     # Load data and add scores
    df_standard = load_with_scores(standartP_csv_path)
    df_adapted = load_with_scores(adaptedP_csv_path)

    # Tag each row with its prompt type
    df_standard["prompt_type"] = "standard"
    df_adapted["prompt_type"] = "adapted"

    # Combine both into one DataFrame
    df_all = pd.concat([df_standard, df_adapted], ignore_index=True)

    # Define colors for each prompt type
    type_color_map = {
        "standard": "#5F5F5F",
        "adapted": "#093586",
    }

    # Compute average score per prompt type
    grouped = (
        df_all.groupby("prompt_type", as_index=False)["score"]
              .mean()
              .rename(columns={"score": "avg_score"})
    )

    # Build bar chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["prompt_type"],
            y=grouped["avg_score"],
            marker_color=grouped["prompt_type"].map(type_color_map),
        )
    )

    fig.update_layout(
        title="Overall evaluation: standard vs adapted prompt",
        xaxis_title="Prompt type",
        yaxis_title="Average score (1–10)",
    )
    fig.show()

# Function: For Exploratory Data Analysis (EDA)
def exploratory_data_analysis(dataset_root: Path) -> pd.DataFrame:

    # Convert string to path
    dataset_root = Path(dataset_root)

    rows: list[dict] = []

    # For each modality
    for modality_dir in dataset_root.iterdir():
        # Checks if proper dir
        if not modality_dir.is_dir():
            continue
        if modality_dir.name.startswith("."):
            continue

        modality = modality_dir.name
        test_dir = modality_dir / "test"
        
        # Check if test_dir exits (Should exist normally)
        if not test_dir.exists():
            continue

        # Init counter vars    
        n_clean = 0
        n_corrupted = 0

        # Go through all the corruptions x severities in the specific modality dir
        for npz_path in sorted(test_dir.glob("*.npz")):
            
            # Load npz file  and get amount of images in npz file
            file = np.load(npz_path)
            arr = file["images"]
            num_imgs = int(arr.shape[0])

            # Add it  either to clean or corrupted
            if npz_path.stem == "clean":
                n_clean += num_imgs
            else:
                n_corrupted += num_imgs

        # Add for total amout of images
        n_total = n_clean + n_corrupted

        # Create entry for the current modality
        rows.append(
            {
                "modality": modality,
                "n_clean": n_clean,
                "n_corrupted": n_corrupted,
                "n_total": n_total,
            }
        )
    
    # Convert to pandas df and summarize
    summary_df = pd.DataFrame(rows).sort_values("modality").reset_index(drop=True)

    # Summaize total clean and total all
    total_clean = int(summary_df["n_clean"].sum())
    total_all = int(summary_df["n_total"].sum())

    # Print Results
    print("Images per modality (clean):")
    print(summary_df[["modality", "n_clean"]])
    print()
    print(f"Total clean Images: {total_clean}")
    print(f"Total Images (clean + corrupted): {total_all}")
    print()
    
    return summary_df

# Function: Generate radar chart for corrupted Images from each modality for each model
def scores_corruption_radar_chart(csv_path: Path) -> None:
    
    # Load data with scores
    df = load_with_scores(csv_path)

    # corruptions
    corruptions = ['brightness', 'contrast', 'gaussian_noise', 'impulse_noise', 'motion_blur', 'pixelate', 'zoom_blur']

    # Get models
    models = df["model_name"].unique().tolist()

    # Compute average score per (corruption, model) 
    grouped = (
        df.groupby(["corruption", "model_name"], as_index=False)["score"]
        .mean()
    )

    # Pivot: rows = corruption, columns = model
    table = grouped.pivot(index="corruption", columns="model_name", values="score")

    # Ensure all categories exist as columns and in correct order
    table = table.reindex(index=corruptions, columns=models).fillna(0)

    # Build radar chart
    fig = go.Figure()

    for corruption, row in table.iterrows():
        r = row.values.tolist() 
        theta = models

        r = r + [r[0]]
        theta = theta + [theta[0]]

        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill=None,
            name=corruption,
            mode="lines+markers"
        ))

    fig.update_layout(
        title="Average score per corruption",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=True
    )

    # Show plot
    #fig.show()

    # Saving plots to local dir
    fig.write_image(OUT_DIR/"average_per_corruption.pdf")

# Function: Save average scores per corruption to csv
def save_corruption_averages_to_csv(csv_path: Path) -> None:
    
    df = load_with_scores(csv_path)

    grouped = (
        df.groupby(["corruption", "model_name"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "avg_score"})
    )

    grouped.to_csv(OUT_DIR / "corruption_averages.csv", index=False)

# Function: Generate radar charts for corrupted Images per severity level per modality for each model
def scores_corruption_radar_chart_per_severity(csv_path: Path) -> None:
    
    df = load_with_scores(csv_path)
    df = df.copy()
    
    # Ensure severity is numeric
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")

    # Keep only rows with corruption not equal to "clean"
    df = df[df["corruption"].ne("clean")]


    # Get list of VLMs
    vlms = sorted(df["model_name"].unique().tolist())

    # For each severity level, create a radar chart
    for sev in sorted(df["severity"].unique().tolist()):
        sub = df[df["severity"] == sev]
        corruptions = sorted(sub["corruption"].unique().tolist())

        # avg score per (corruption, VLM)
        pivot = (
            sub.groupby(["corruption", "model_name"])["score"]
               .mean()
               .unstack("model_name")
               .reindex(columns=vlms)
        )

        fig = go.Figure()
        for corr, row in pivot.iterrows():
            r = row.to_numpy(dtype=float)
            fig.add_trace(
                go.Scatterpolar(
                    r=np.r_[r, r[0]],
                    theta=vlms + [vlms[0]],
                    mode="lines+markers",
                    name=str(corr),
                )
            )

        fig.update_layout(
            title=f"Avg score per VLM (severity={int(sev)})",
            polar=dict(
                radialaxis=dict(
                    range=[1, 5],
                    tickmode="array",
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=["1", "2", "3", "4", "5"],
                )
            ),
        )

        fig.write_image(f"analysis/corruption_averages/corruption_radar_severity_{sev}.pdf")

# Function: Generates Heatmaps for each vlm showing the change in rating for each modality and corruption comparred to clean images
def plot_vlm_corruption_heatmaps(csv_path: Path) -> None:

    df = load_with_scores(csv_path)
    corruption_categories = [
        "brightness",
        "contrast",
        "gaussian_noise",
        "impulse_noise",
        "motion_blur",
        "pixelate",
        "zoom_blur",
    ]

    # Avg per (model_name, modality, corruption) over all severities
    grouped = (
        df.groupby(["model_name", "modality", "corruption"], as_index=False)["score"]
          .mean()
          .rename(columns={"score": "avg_score"})
    )

    # Clean baseline per (model_name, modality)
    clean_baseline = (
        grouped[grouped["corruption"] == "clean"]
        .drop(columns=["corruption"])
        .rename(columns={"avg_score": "clean_score"})
    )

    corrupted = grouped[grouped["corruption"] != "clean"]

    merged = corrupted.merge(clean_baseline, on=["model_name", "modality"], how="left")

    # % change vs clean
    clean = merged["clean_score"].to_numpy(dtype=float)
    avg = merged["avg_score"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = (avg - clean) / clean * 100.0
    pct_change[~np.isfinite(pct_change)] = 0.0
    merged["pct_change"] = pct_change

    # Consstent color scale across all VLMs
    max_abs = float(np.nanmax(np.abs(merged["pct_change"].to_numpy(dtype=float)))) if len(merged) else 1.0
    if max_abs == 0:
        max_abs = 1.0

    vlms = sorted(merged["model_name"].unique().tolist())

    for vlm in vlms:
        vlm_df = merged[merged["model_name"] == vlm].copy()

        heatmap_table = vlm_df.pivot(
            index="modality",
            columns="corruption",
            values="pct_change",
        ).reindex(columns=corruption_categories).fillna(0.0)

        # Annotation strings
        annot_labels = heatmap_table.map(lambda x: f"{x:+.1f}%")

        plt.figure(figsize=(10, max(4, 0.5 * len(heatmap_table.index))))
        ax = sns.heatmap(
            heatmap_table,
            cmap="RdBu",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            annot=annot_labels,
            fmt="",
            linewidths=0.5,
            cbar_kws={"label": "Δ Score vs. clean (%)"},
        )

        # Set titles and labels
        ax.set_title(f"Score change per corruption compared to clean - {vlm}")
        ax.set_xlabel("Corruption")
        ax.set_ylabel("Modality")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save figure and close
        plt.savefig(f"analysis/heatmap_{vlm}.pdf", format="pdf")
        plt.close()

# Function: Generate heatmap averaged across all models
def plot_avg_corruption_heatmap_all_models(csv_path: Path) -> None:
    df = load_with_scores(csv_path)

    corruption_categories = [
        "brightness",
        "contrast",
        "gaussian_noise",
        "impulse_noise",
        "motion_blur",
        "pixelate",
        "zoom_blur",
    ]

    # Group per (model_name, modality, coruption) over all severities
    grouped = (
        df.groupby(["model_name", "modality", "corruption"], as_index=False)["score"]
          .mean()
          .rename(columns={"score": "avg_score"})
    )

    # Keep only corrupted entries
    corrupted = grouped[grouped["corruption"] != "clean"]

    # Clean baseline per (model_name, modality)
    clean_baseline = (
        grouped[grouped["corruption"] == "clean"]
        .drop(columns=["corruption"])
        .rename(columns={"avg_score": "clean_score"})
    )

    # Merging corrupted with clean baseline
    merged = corrupted.merge(clean_baseline, on=["model_name", "modality"], how="left")

    # % change vs clean (per model, per modality, per corruption)
    clean = merged["clean_score"].to_numpy(dtype=float)
    avg = merged["avg_score"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = (avg - clean) / clean * 100.0
    pct_change[~np.isfinite(pct_change)] = 0.0
    merged["pct_change"] = pct_change

    # Consistent color scale
    max_abs = float(np.nanmax(np.abs(merged["pct_change"].to_numpy(dtype=float)))) if len(merged) else 1.0
    if max_abs == 0:
        max_abs = 1.0

    # Average % change across all models for each (modality, corruption)
    avg_across_models = (
        merged.groupby(["modality", "corruption"], as_index=False)["pct_change"]
              .mean()
    )

    heatmap_table = (
        avg_across_models.pivot(index="modality", columns="corruption", values="pct_change")
        .reindex(columns=corruption_categories)
        .sort_index()
        .fillna(0.0)
    )

    annot_labels = heatmap_table.map(lambda x: f"{x:+.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, max(4, 0.5 * len(heatmap_table.index))))
    ax = sns.heatmap(
        heatmap_table,
        cmap="RdBu",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        annot=annot_labels,
        fmt="",
        linewidths=0.5,
        cbar_kws={"label": "Δ Score vs. clean (%) (avg over models)"},
    )

    ax.set_title("Average score change per corruption compared to clean (avg over all VLMs)")
    ax.set_xlabel("Corruption")
    ax.set_ylabel("Modality")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(OUT_DIR / "heatmap_all_models.pdf", format="pdf")
    plt.close()


    # Function: Generate heatmap averaged across all models
def plot_avg_corruption_heatmap_all_models(csv_path: Path) -> None:
    df = load_with_scores(csv_path)

    corruption_categories = [
        "brightness",
        "contrast",
        "gaussian_noise",
        "impulse_noise",
        "motion_blur",
        "pixelate",
        "zoom_blur",
    ]

    # Group per (model_name, modality, coruption) over all severities
    grouped = (
        df.groupby(["model_name", "modality", "corruption"], as_index=False)["score"]
          .mean()
          .rename(columns={"score": "avg_score"})
    )

    # Keep only corrupted entries
    corrupted = grouped[grouped["corruption"] != "clean"]

    # Clean baseline per (model_name, modality)
    clean_baseline = (
        grouped[grouped["corruption"] == "clean"]
        .drop(columns=["corruption"])
        .rename(columns={"avg_score": "clean_score"})
    )

    # Merging corrupted with clean baseline
    merged = corrupted.merge(clean_baseline, on=["model_name", "modality"], how="left")

    # % change vs clean (per model, per modality, per corruption)
    clean = merged["clean_score"].to_numpy(dtype=float)
    avg = merged["avg_score"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = (avg - clean) / clean * 100.0
    pct_change[~np.isfinite(pct_change)] = 0.0
    merged["pct_change"] = pct_change

    # Consistent color scale
    max_abs = float(np.nanmax(np.abs(merged["pct_change"].to_numpy(dtype=float)))) if len(merged) else 1.0
    if max_abs == 0:
        max_abs = 1.0

    # Average % change across all models for each (modality, corruption)
    avg_across_models = (
        merged.groupby(["modality", "corruption"], as_index=False)["pct_change"]
              .mean()
    )

    heatmap_table = (
        avg_across_models.pivot(index="modality", columns="corruption", values="pct_change")
        .reindex(columns=corruption_categories)
        .sort_index()
        .fillna(0.0)
    )

    annot_labels = heatmap_table.map(lambda x: f"{x:+.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, max(4, 0.5 * len(heatmap_table.index))))
    ax = sns.heatmap(
        heatmap_table,
        cmap="RdBu",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        annot=annot_labels,
        fmt="",
        linewidths=0.5,
        cbar_kws={"label": "Δ Score vs. clean (%) (avg over models)"},
    )

    ax.set_title("Average score change per corruption compared to clean (avg over all VLMs)")
    ax.set_xlabel("Corruption")
    ax.set_ylabel("Modality")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(OUT_DIR / "heatmap_all_models.pdf", format="pdf")
    plt.close()


def plot_correlation_heatmap(
    csv_path,
    agg_func="mean",
    corr_method="pearson",
    drop_constant=True,
    save_path=None 
):
    df = load_with_scores(csv_path)

    # Aggregate
    df_agg = (
        df
        .groupby(
            ["modality", "corruption", "severity", "index", "model_name"],
            as_index=False
        )
        .agg(score=("score", agg_func))
    )

    # Pivot
    wide = df_agg.pivot(
        index=["modality", "corruption", "severity", "index"],
        columns="model_name",
        values="score"
    )

    # Drop constant models
    if drop_constant:
        nunique = wide.nunique()
        wide = wide.loc[:, nunique > 1]

    # Correlation
    corr = wide.corr(method=corr_method).fillna(0)

    # Plot
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        labels=dict(x="Model", y="Model", color="Correlation"),
    )

    fig.update_traces(textfont_size=16)

    fig.update_layout(
        title=f"Model–Model Correlation ({corr_method.capitalize()})",
        xaxis_title="Model",
        yaxis_title="Model",
    )

    # Show
    fig.show()

    # Save to PDF if requested
    if save_path:
        fig.write_image(save_path, format="pdf", width=1000, height=800)
        print(f"Saved heatmap to {save_path}")

    return corr

# Function: Plot heatmaps showing the change in score for biased prompts compared to the unbiased prompt
def plot_vlm_bias_prompt_heatmaps(csv_path: Path, baseline_label: str = "Original") -> None:

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_with_scores(csv_path).copy()
    df["bias_status"] = df["bias_status"].astype(str).str.strip()

    # Mean per (model, modality, bias_status)
    grouped = (
        df.groupby(["model_name", "modality", "bias_status"], as_index=False)["score"]
          .mean()
          .rename(columns={"score": "avg_score"})
    )

    # Baseline (model, modality)
    baseline = (
        grouped[grouped["bias_status"].str.lower() == baseline_label.lower()]
        .drop(columns=["bias_status"])
        .rename(columns={"avg_score": "baseline_score"})
    )

    # Only biased Rows
    biased = grouped[grouped["bias_status"].str.lower() != baseline_label.lower()]
    merged = biased.merge(baseline, on=["model_name", "modality"], how="left")

    # % change vs. baseline
    base = merged["baseline_score"].to_numpy(dtype=float)
    avg = merged["avg_score"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = (avg - base) / base * 100.0
    pct_change[~np.isfinite(pct_change)] = 0.0
    merged["pct_change"] = pct_change

    # Get bias categories
    bias_categories = sorted(merged["bias_status"].unique().tolist())

    # Uniform color scale
    max_abs = float(np.nanmax(np.abs(merged["pct_change"].to_numpy(dtype=float)))) if len(merged) else 1.0
    if max_abs == 0:
        max_abs = 1.0

    # Per model heatmaps
    vlms = sorted(merged["model_name"].unique().tolist())
    for vlm in vlms:
        vlm_df = merged[merged["model_name"] == vlm].copy()

        heatmap_table = (
            vlm_df.pivot(index="modality", columns="bias_status", values="pct_change")
                 .reindex(columns=bias_categories)
                 .fillna(0.0)
        )

        annot_labels = heatmap_table.map(lambda x: f"{x:+.1f}%")

        # Plot
        plt.figure(
            figsize=(max(8, 0.7 * len(heatmap_table.columns)),
                     max(4, 0.5 * len(heatmap_table.index)))
        )
        ax = sns.heatmap(
            heatmap_table,
            cmap="RdBu",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            annot=annot_labels,
            fmt="",
            linewidths=0.5,
            cbar_kws={"label": f"Δ Score vs. {baseline_label} (%)"},
        )

        ax.set_title(f"Score change: Biased vs. Unbiased prompt - {vlm}")
        ax.set_xlabel("Bias prompt type")
        ax.set_ylabel("Modality")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plots
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(vlm))
        plt.savefig(OUT_DIR / f"heatmap_bias_{safe_name}.pdf", format="pdf")
        plt.close()

# Function: Generate one heatmap averaged across all models for biased prompts
def plot_avg_bias_prompt_heatmap_all_models(
    csv_path: Path,
    baseline_label: str = "Original",
) -> None:

    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_with_scores(csv_path).copy()
    df["bias_status"] = df["bias_status"].astype(str).str.strip()

    grouped = (
        df.groupby(["model_name", "modality", "bias_status"], as_index=False)["score"]
          .mean()
          .rename(columns={"score": "avg_score"})
    )

    baseline = (
        grouped[grouped["bias_status"].str.lower() == baseline_label.lower()]
        .drop(columns=["bias_status"])
        .rename(columns={"avg_score": "baseline_score"})
    )

    biased = grouped[grouped["bias_status"].str.lower() != baseline_label.lower()]
    merged = biased.merge(baseline, on=["model_name", "modality"], how="left")

    base = merged["baseline_score"].to_numpy(dtype=float)
    avg = merged["avg_score"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_change = (avg - base) / base * 100.0
    pct_change[~np.isfinite(pct_change)] = 0.0
    merged["pct_change"] = pct_change

    # Average across models for each (modality, bias_status)
    avg_across_models = (
        merged.groupby(["modality", "bias_status"], as_index=False)["pct_change"]
              .mean()
    )

    bias_categories = sorted(avg_across_models["bias_status"].unique().tolist())

    heatmap_table = (
        avg_across_models.pivot(index="modality", columns="bias_status", values="pct_change")
                        .reindex(columns=bias_categories)
                        .sort_index()
                        .fillna(0.0)
    )

    # Consistent color scale (based on averaged table)
    max_abs = float(np.nanmax(np.abs(heatmap_table.to_numpy(dtype=float)))) if heatmap_table.size else 1.0
    if max_abs == 0:
        max_abs = 1.0

    annot_labels = heatmap_table.map(lambda x: f"{x:+.1f}%")

    plt.figure(figsize=(max(8, 0.7 * len(heatmap_table.columns)), max(4, 0.5 * len(heatmap_table.index))))
    ax = sns.heatmap(
        heatmap_table,
        cmap="RdBu",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        annot=annot_labels,
        fmt="",
        linewidths=0.5,
        cbar_kws={"label": f"Δ Score vs. {baseline_label} (%)"},
    )

    ax.set_title(f"Score change: Biased Prompt vs Unbiased (avg over all VLMs)")
    ax.set_xlabel("Bias prompt type")
    ax.set_ylabel("Modality")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(OUT_DIR / "heatmap_bias_all_models.pdf", format="pdf")
    plt.close()