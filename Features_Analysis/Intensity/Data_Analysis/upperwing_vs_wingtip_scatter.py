"""
Create visualizations showing how upperwing (mean wing) and filtered wingtip
intensities relate for every bird in the dark-pixel dataset.

Outputs:
- Scatter plot with equal axes and a 1:1 reference line (one dot per bird).
- Grouped bar chart summarizing per-species mean wing vs wingtip intensities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# Use same clean aesthetic for every plot
plt.style.use("seaborn-v0_8-whitegrid")

# Consistent colours with other intensity plots
SPECIES_COLORS = {
    "Glaucous_Winged_Gull": "#3274A1",  # Blue
    "Slaty_Backed_Gull": "#E1812C",  # Orange
}

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Wingtip_Dark_Pixel_Visualizations" / "dark_pixel_results_all_images.csv"
OUTPUT_DIR = BASE_DIR / "Intensity_Distribution_Comparison"
SCATTER_OUTPUT = OUTPUT_DIR / "upperwing_vs_wingtip_scatter.png"
BAR_OUTPUT = OUTPUT_DIR / "upperwing_vs_wingtip_bar.png"
DARK_BOX_OUTPUT = OUTPUT_DIR / "dark_pixel_percentage_box.png"
HIST_OUTPUT = OUTPUT_DIR / "dark_pixel_histogram_comparison.png"

# Histogram configuration similar to publish7/publish6
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
    """Return a publication-friendly species label."""
    label = species.replace("_", " ")
    label = label.replace("Winged", "-winged").replace("Backed", "-backed")
    return label


def load_intensity_data() -> pd.DataFrame:
    """Load and filter the per-bird intensity measurements."""
    df = pd.read_csv(DATA_PATH)
    required_cols = {"species", "mean_wing_intensity", "mean_wingtip_intensity"}

    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Missing required columns in intensity CSV: {missing}")

    df = df.dropna(subset=list(required_cols))
    df = df[df["species"].isin(SPECIES_COLORS.keys())]
    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print console summary of record counts and mean intensities."""
    total_rows = len(df)
    print("\nDATA SUMMARY: dark_pixel_results_all_images.csv (filtered wingtip)")
    print(f"Total images / scatter dots: {total_rows}")

    grouped = df.groupby("species")
    for species, subset in grouped:
        label = format_species_label(species)
        count = len(subset)
        wing_mean = subset["mean_wing_intensity"].mean()
        wingtip_mean = subset["mean_wingtip_intensity"].mean()
        print(
            f"  {label}: n={count}, "
            f"upperwing mean={wing_mean:.2f}, wingtip mean={wingtip_mean:.2f}"
        )

    overall_wing = df["mean_wing_intensity"].mean()
    overall_wingtip = df["mean_wingtip_intensity"].mean()
    print(
        f"Overall means → upperwing: {overall_wing:.2f}, "
        f"wingtip: {overall_wingtip:.2f}\n"
    )


def plot_upperwing_vs_wingtip(df: pd.DataFrame) -> None:
    """Generate the scatter plot with equal axes and a 1:1 reference line."""
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

    # Diagonal reference showing where wing and wingtip values match
    ax.plot(
        [0, 255],
        [0, 255],
        linestyle="--",
        color="#2f2f2f",
        linewidth=1.4,
        label="Upperwing = Wingtip",
    )

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Upperwing mean intensity (0–255)")
    ax.set_ylabel("Wingtip mean intensity (0–255)")
    ax.set_title(
        "Relationship Between Upperwing and Wingtip Tones (per bird)",
        fontsize=13,
        fontweight="bold",
    )

    # Tidy legend with formatted species labels plus the 1:1 reference line
    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = []
    for label in labels:
        if label in SPECIES_COLORS:
            formatted_labels.append(format_species_label(label))
        else:
            formatted_labels.append(label)
    ax.legend(handles, formatted_labels, title="Species", frameon=True)

    fig.tight_layout()
    fig.savefig(SCATTER_OUTPUT, dpi=300)
    print(f"Saved scatter plot to {SCATTER_OUTPUT}")


def plot_species_bar_chart(df: pd.DataFrame) -> None:
    """Generate grouped bar chart of per-species mean wing vs wingtip intensity."""
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
    ax.set_title(
        "Average Upperwing vs Wingtip Intensity (filtered dark-pixel data)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(title="Metric", frameon=True)

    for idx, species in enumerate(summary["species"].unique()):
        label = format_species_label(species)
        count = counts.get(species, 0)
        ax.text(
            idx,
            262,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(BAR_OUTPUT, dpi=300)
    print(f"Saved bar plot to {BAR_OUTPUT}")


def plot_dark_pixel_boxplot(df: pd.DataFrame) -> None:
    """Create a box plot of dark pixel percentages per species."""
    column = "pct_darker_pixels"
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {DATA_PATH.name}. "
            "Cannot build dark-pixel percentage plot."
        )

    box_df = df.copy()
    box_df["species_display"] = box_df["species"].apply(format_species_label)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=box_df,
        x="species_display",
        y=column,
        palette=[SPECIES_COLORS[s] for s in SPECIES_COLORS],
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
    ax.set_ylim(0, min(100, box_df[column].max() + 5))
    ax.set_title("Distribution of dark wingtip pixels (filtered data)")

    fig.tight_layout()
    fig.savefig(DARK_BOX_OUTPUT, dpi=300)
    print(f"Saved dark pixel box plot to {DARK_BOX_OUTPUT}")


def create_histogram_interpolation(data, bins):
    """Return smoothed histogram curve for publish7-like look."""
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    non_zero_mask = counts > 0

    if non_zero_mask.sum() < 2:
        x_points = bin_centers
        y_points = counts
    else:
        x_points = bin_centers[non_zero_mask]
        y_points = counts[non_zero_mask]

    interp = np.interp(np.linspace(x_points.min(), x_points.max(), 200), x_points, y_points)
    x_smooth = np.linspace(x_points.min(), x_points.max(), 200)
    return bin_centers, counts, x_smooth, interp


def add_intensity_reference(ax, bins):
    """Add grayscale reference strip below histogram (publish6 style)."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    box_height = ylim[1] * 0.08
    box_y = ylim[0] - box_height * 2

    for i in range(len(bins) - 1):
        start = bins[i]
        gray_value = str(((bins[i] + bins[i + 1]) / 2) / 255.0)
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


def plot_histogram_comparison(df: pd.DataFrame) -> None:
    """Replicate two-panel histogram layout using the filtered dataset."""
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    titles = ["Wing Intensity Distribution", "Wingtip Intensity Distribution"]
    data_columns = ["mean_wing_intensity", "mean_wingtip_intensity"]

    for ax, title, column in zip(axes, titles, data_columns):
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Mean intensity (0–255)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 255)

        for species in SPECIES_COLORS.keys():
            species_data = df[df["species"] == species][column]
            color = SPECIES_COLORS[species]
            label = format_species_label(species)

            ax.hist(
                species_data,
                bins=INTENSITY_BINS,
                alpha=0.55,
                color=color,
                density=True,
                edgecolor="black",
                linewidth=0.6,
                label=label,
            )

            _, _, x_smooth, y_smooth = create_histogram_interpolation(species_data, INTENSITY_BINS)
            ax.plot(x_smooth, y_smooth, color=color, linewidth=2.2)

            mean_val = species_data.mean()
            std_val = species_data.std()
            ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.6)
            ax.text(
                mean_val,
                ax.get_ylim()[1] * 0.9,
                f"Mean: {mean_val:.1f}\nStd: {std_val:.1f}",
                color=color,
                fontsize=10,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=color),
            )

        add_intensity_reference(ax, INTENSITY_BINS)
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(bottom=0.12)
    fig.text(
        0.5,
        0.04,
        "Bin width: 15 units | Reference strip shows absolute intensity values",
        ha="center",
        fontsize=10,
        style="italic",
    )
    fig.savefig(HIST_OUTPUT, dpi=300)
    print(f"Saved histogram comparison to {HIST_OUTPUT}")


def main() -> None:
    df = load_intensity_data()
    if df.empty:
        raise ValueError("No intensity records found for the configured species.")
    print_dataset_summary(df)
    plot_upperwing_vs_wingtip(df)
    plot_species_bar_chart(df)
    plot_dark_pixel_boxplot(df)
    plot_histogram_comparison(df)


if __name__ == "__main__":
    main()

