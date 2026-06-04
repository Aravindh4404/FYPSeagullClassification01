"""
Generate publication-ready plots from the latest dark pixel dataset (FINALE.csv).

Outputs (saved under Intensity_Distribution_Comparison_Finale/):
- Scatter plot of upperwing vs wingtip means with 1:1 reference
- Grouped bar chart of per-species mean intensities
- Box plot for wingtip dark-pixel percentages
- Wing and wingtip intensity distributions styled like prior figures
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle

plt.style.use("seaborn-v0_8-whitegrid")

SPECIES_COLORS = {
    "Glaucous_Winged_Gull": "#3274A1",
    "Slaty_Backed_Gull": "#E1812C",
}

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Wingtip_Dark_Pixel_Visualizations" / "FINALE.csv"
OUTPUT_DIR = BASE_DIR / "Intensity_Distribution_Comparison_Finale"
SCATTER_OUTPUT = OUTPUT_DIR / "upperwing_vs_wingtip_scatter.png"
BAR_OUTPUT = OUTPUT_DIR / "upperwing_vs_wingtip_bar.png"
DARK_BOX_OUTPUT = OUTPUT_DIR / "dark_pixel_percentage_box.png"
WING_DIST_OUTPUT = OUTPUT_DIR / "wing_intensity_distribution.png"
WINGTIP_DIST_OUTPUT = OUTPUT_DIR / "wingtip_intensity_distribution.png"

INTENSITY_BINS = list(range(0, 256, 15))
BIN_WIDTH = 15

METRIC_LABELS = {
    "mean_wing_intensity": "Upperwing mean",
    "mean_wingtip_intensity": "Wingtip mean",
}
METRIC_COLORS = {
    "Upperwing mean": "#4C78A8",
    "Wingtip mean": "#E15759",
}


def format_species_label(species: str) -> str:
    label = species.replace("_", " ")
    label = label.replace("Winged", "-winged").replace("Backed", "-backed")
    return label


def load_intensity_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    wing_cols = [col for col in df.columns if "mean_wing_intensity" in col]
    wingtip_cols = [col for col in df.columns if "mean_wingtip_intensity" in col]

    if not wing_cols or not wingtip_cols:
        raise ValueError("Missing required mean intensity columns in FINALE.csv")

    df = df.rename(
        columns={
            wing_cols[0]: "mean_wing_intensity",
            wingtip_cols[0]: "mean_wingtip_intensity",
        }
    )
    required = {"species", "mean_wing_intensity", "mean_wingtip_intensity"}
    df = df.dropna(subset=list(required))
    df = df[df["species"].isin(SPECIES_COLORS.keys())]
    df = ensure_dark_pixel_percentage(df)
    return df


def ensure_dark_pixel_percentage(df: pd.DataFrame) -> pd.DataFrame:
    preferred = "pct_wingtip_darker_than_wing"
    if preferred in df.columns:
        df["pct_darker_pixels"] = df[preferred].clip(0, 100)
        return df
    if "pct_darker_pixels" in df.columns:
        return df

    candidate_pairs = [
        ("wingtip_darker_than_wing_count", "wingtip_pixel_count"),
        ("darker_pixel_count", "wingtip_pixel_count"),
        ("method1_count", "total_filtered_pixels"),
    ]
    for numerator, denominator in candidate_pairs:
        if numerator in df.columns and denominator in df.columns:
            denom = df[denominator].replace(0, np.nan)
            df["pct_darker_pixels"] = (df[numerator] / denom) * 100
            df["pct_darker_pixels"] = df["pct_darker_pixels"].clip(0, 100)
            print(f"Calculated pct_darker_pixels using {numerator} / {denominator}")
            return df

    print("Warning: Could not determine pct_darker_pixels; box plot will be skipped.")
    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    total_rows = len(df)
    print("\nDATA SUMMARY: FINALE.csv (filtered wingtip)")
    print(f"Total images / scatter dots: {total_rows}")
    grouped = df.groupby("species")
    for species, subset in grouped:
        label = format_species_label(species)
        wing_mean = subset["mean_wing_intensity"].mean()
        wingtip_mean = subset["mean_wingtip_intensity"].mean()
        print(
            f"  {label}: n={len(subset)}, "
            f"upperwing mean={wing_mean:.2f}, wingtip mean={wingtip_mean:.2f}"
        )
    print(
        f"Overall means → upperwing: {df['mean_wing_intensity'].mean():.2f}, "
        f"wingtip: {df['mean_wingtip_intensity'].mean():.2f}\n"
    )


def plot_upperwing_vs_wingtip(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.scatterplot(
        data=df,
        x="mean_wing_intensity",
        y="mean_wingtip_intensity",
        hue="species",
        palette=SPECIES_COLORS,
        s=120,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
        ax=ax,
    )
    ax.plot([0, 255], [0, 255], linestyle="--", color="#2f2f2f", linewidth=1.4)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Upperwing mean intensity (0–255)")
    ax.set_ylabel("Wingtip mean intensity (0–255)")
    ax.set_title("Relationship Between Upperwing and Wingtip Tones (per bird)", fontsize=13, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = [format_species_label(lbl) if lbl in SPECIES_COLORS else lbl for lbl in labels]
    ax.legend(handles, formatted_labels, title="Species", frameon=True)

    fig.tight_layout()
    fig.savefig(SCATTER_OUTPUT, dpi=300)
    print(f"Saved scatter plot to {SCATTER_OUTPUT}")


def plot_species_bar_chart(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("species")[["mean_wing_intensity", "mean_wingtip_intensity"]]
        .mean()
        .reset_index()
        .melt(id_vars="species", var_name="metric", value_name="intensity")
    )
    summary["metric"] = summary["metric"].map(METRIC_LABELS)
    summary["species_display"] = summary["species"].apply(format_species_label)
    counts = df.groupby("species").size().to_dict()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=summary,
        x="species_display",
        y="intensity",
        hue="metric",
        palette=METRIC_COLORS,
        ax=ax,
    )
    ax.set_ylabel("Mean intensity (0–255)")
    ax.set_xlabel("Species")
    ax.set_ylim(0, 270)
    ax.set_title("Average Upperwing vs Wingtip Intensity (filtered data)", fontsize=12, fontweight="bold")
    ax.legend(title="Metric", frameon=True)

    for idx, species in enumerate(summary["species"].unique()):
        count = counts.get(species, 0)
        ax.text(idx, 262, f"n={count}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(BAR_OUTPUT, dpi=300)
    print(f"Saved bar plot to {BAR_OUTPUT}")


def plot_dark_pixel_boxplot(df: pd.DataFrame) -> None:
    column = "pct_darker_pixels"
    if column not in df.columns:
        print(f"Warning: Column '{column}' not available. Skipping box plot.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    box_df = df.copy()
    box_df["species_display"] = box_df["species"].apply(format_species_label)
    box_df = box_df.dropna(subset=[column])
    box_df[column] = box_df[column].clip(0, 100)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        data=box_df,
        x="species_display",
        y=column,
        palette=[SPECIES_COLORS[s] for s in SPECIES_COLORS if s in box_df["species"].unique()],
        ax=ax,
    )
    sns.stripplot(
        data=box_df,
        x="species_display",
        y=column,
        color="black",
        alpha=0.35,
        size=5,
        jitter=0.15,
        ax=ax,
    )

    ax.set_ylabel("Wingtip dark pixel percentage (%)")
    ax.set_xlabel("Species")
    ax.set_title("Distribution of dark wingtip pixels (FINALE data)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(DARK_BOX_OUTPUT, dpi=300)
    print(f"Saved dark pixel box plot to {DARK_BOX_OUTPUT}")


def create_histogram_kde(data, bins):
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    try:
        kde = stats.gaussian_kde(data, bw_method="scott")
        x_smooth = np.linspace(data.min(), data.max(), 200)
        y_smooth = kde(x_smooth)
    except Exception:
        x_smooth = bin_centers
        y_smooth = counts
    return bin_centers, counts, x_smooth, y_smooth


def add_intensity_reference(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    box_height = ylim[1] * 0.08
    box_y = ylim[0] - box_height * 2
    for i in range(len(INTENSITY_BINS) - 1):
        start = INTENSITY_BINS[i]
        gray_value = str(((INTENSITY_BINS[i] + INTENSITY_BINS[i + 1]) / 2) / 255.0)
        rect = Rectangle(
            (start, box_y),
            BIN_WIDTH,
            box_height,
            facecolor=gray_value,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.add_patch(rect)
    ax.set_ylim(box_y - box_height * 1.5, ylim[1])


def plot_intensity_distribution(df: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Mean intensity (0–255)")
    ax.set_ylabel("Density")

    for species in df["species"].unique():
        species_data = df[df["species"] == species][column]
        color = SPECIES_COLORS[species]
        label = format_species_label(species)

        ax.hist(
            species_data,
            bins=INTENSITY_BINS,
            alpha=0.6,
            color=color,
            density=True,
            edgecolor="black",
            linewidth=0.8,
            label=label,
        )

        _, _, x_smooth, y_smooth = create_histogram_kde(species_data, INTENSITY_BINS)
        ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.9)

        mean_val = species_data.mean()
        std_val = species_data.std()
        ax.axvline(mean_val, color=color, linestyle="--", alpha=0.8, linewidth=2)
        ax.text(
            mean_val,
            ax.get_ylim()[1] * 0.85,
            f"Mean: {mean_val:.1f}\nStd: {std_val:.1f}",
            color=color,
            fontweight="bold",
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=color),
        )

    add_intensity_reference(ax)
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.set_xlim(0, 255)
    ax.grid(True, alpha=0.3)

    fig.text(
        0.5,
        0.02,
        "Bin width: 15 units | Intensity reference shows absolute brightness values",
        ha="center",
        fontsize=9,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved intensity distribution plot to {output_path}")


def main() -> None:
    df = load_intensity_data()
    if df.empty:
        raise ValueError("No intensity records found for configured species.")
    print_dataset_summary(df)
    plot_upperwing_vs_wingtip(df)
    plot_species_bar_chart(df)
    plot_dark_pixel_boxplot(df)
    plot_intensity_distribution(df, "mean_wing_intensity", "Wing Intensity Distribution", WING_DIST_OUTPUT)
    plot_intensity_distribution(
        df,
        "mean_wingtip_intensity",
        "Wingtip Intensity Distribution",
        WINGTIP_DIST_OUTPUT,
    )


if __name__ == "__main__":
    main()

