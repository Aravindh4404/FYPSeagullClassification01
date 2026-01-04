"""
Create visualizations showing how upperwing (mean wing) and filtered wingtip
intensities relate for every bird in the dataset.

FILTERS OUT images where wing intensity is darker than wingtip intensity.

Outputs:
- Scatter plot with equal axes and a 1:1 reference line (one dot per bird).
- Grouped bar chart summarizing per-species mean wing vs wingtip intensities.
- Box plot for dark pixel percentage distribution.
- Wing and wingtip intensity distribution histograms (matching reference style).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from matplotlib.patches import Rectangle

# Use same clean aesthetic for every plot
plt.style.use("seaborn-v0_8-whitegrid")

# Consistent colours with other intensity plots
SPECIES_COLORS = {
    "Glaucous_Winged_Gull": "#3274A1",  # Blue
    "Slaty_Backed_Gull": "#E1812C",  # Orange
}

# Path to your all.csv file - MODIFY THIS PATH TO YOUR ACTUAL FILE LOCATION
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "D:\FYPSeagullClassification01\Features_Analysis\Intensity\Data_Analysis\Wingtip_Dark_Pixel_Visualizations\dark_pixel_results_all_images.csv"

# Output directories for filtered and unfiltered data
OUTPUT_DIR_FILTERED = BASE_DIR / "Intensity_Distribution_Filtered"
OUTPUT_DIR_UNFILTERED = BASE_DIR / "Intensity_Distribution_Unfiltered"

# Filtered outputs (primaries darker than upperparts)
SCATTER_OUTPUT_FILTERED = OUTPUT_DIR_FILTERED / "upperparts_vs_primaries_scatter_filtered.png"
BAR_OUTPUT_FILTERED = OUTPUT_DIR_FILTERED / "upperparts_vs_primaries_bar_filtered.png"
DARK_BOX_OUTPUT_FILTERED = OUTPUT_DIR_FILTERED / "dark_pixel_percentage_box_filtered.png"
UPPERPARTS_DIST_OUTPUT_FILTERED = OUTPUT_DIR_FILTERED / "upperparts_intensity_distribution_filtered.png"
PRIMARIES_DIST_OUTPUT_FILTERED = OUTPUT_DIR_FILTERED / "primaries_intensity_distribution_filtered.png"

# Unfiltered outputs (all data)
SCATTER_OUTPUT_UNFILTERED = OUTPUT_DIR_UNFILTERED / "upperparts_vs_primaries_scatter_all.png"
BAR_OUTPUT_UNFILTERED = OUTPUT_DIR_UNFILTERED / "upperparts_vs_primaries_bar_all.png"
DARK_BOX_OUTPUT_UNFILTERED = OUTPUT_DIR_UNFILTERED / "dark_pixel_percentage_box_all.png"
UPPERPARTS_DIST_OUTPUT_UNFILTERED = OUTPUT_DIR_UNFILTERED / "upperparts_intensity_distribution_all.png"
PRIMARIES_DIST_OUTPUT_UNFILTERED = OUTPUT_DIR_UNFILTERED / "primaries_intensity_distribution_all.png"

# CSV outputs for filtered and removed images
FILTERED_CSV = OUTPUT_DIR_FILTERED / "filtered_images_list.csv"
REMOVED_CSV = OUTPUT_DIR_FILTERED / "removed_images_list.csv"

# Define bin configuration - 15-unit bins matching reference image
INTENSITY_BINS = list(range(0, 256, 15))  # [0, 15, 30, 45, ..., 240, 255]
BIN_WIDTH = 15

METRIC_LABELS = {
    "mean_wing_intensity": "Upperparts mean",
    "mean_wingtip_intensity": "Primaries mean",
}
METRIC_COLORS = {
    "Upperparts mean": "#3274A1",  # Blue (matching Glaucous-winged Gull)
    "Primaries mean": "#E1812C",   # Orange (matching Slaty-backed Gull)
}


def format_species_label(species: str) -> str:
    """Return a publication-friendly species label."""
    label = species.replace("_", " ")
    label = label.replace("Winged", "-winged").replace("Backed", "-backed")
    return label


def load_intensity_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and filter the per-bird intensity measurements from all.csv.
    Returns: (filtered_df, unfiltered_df, removed_df)
    """
    df = pd.read_csv(DATA_PATH)

    print(f"\nLoaded CSV with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}\n")

    # Map the column names from your CSV to the expected names
    wing_cols = [col for col in df.columns if 'mean_wir' in col.lower() or 'mean_wing' in col.lower()]
    wingtip_cols = [col for col in df.columns if 'mean_wingtip' in col.lower()]

    if wing_cols and wingtip_cols:
        wing_col = wing_cols[0]
        wingtip_col = wingtip_cols[0]

        print(f"Using wing intensity column: {wing_col}")
        print(f"Using wingtip intensity column: {wingtip_col}")

        df = df.rename(columns={
            wing_col: 'mean_wing_intensity',
            wingtip_col: 'mean_wingtip_intensity'
        })
    else:
        raise ValueError(
            f"Could not find appropriate intensity columns.\n"
            f"Wing columns found: {wing_cols}\n"
            f"Wingtip columns found: {wingtip_cols}"
        )

    required_cols = {"species", "mean_wing_intensity", "mean_wingtip_intensity"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Missing required columns in intensity CSV: {missing}")

    # Drop rows with missing values in required columns
    df = df.dropna(subset=list(required_cols))

    # Filter for species in SPECIES_COLORS (before any other filtering)
    df = df[df["species"].isin(SPECIES_COLORS.keys())].copy()

    # Store the unfiltered data
    unfiltered_df = df.copy()
    unfiltered_df = ensure_dark_pixel_percentage(unfiltered_df)

    # CRITICAL FILTER: Separate rows where wing is darker than wingtip
    initial_count = len(df)

    # Check if there's a boolean column indicating wing darker than wingtip
    bool_cols = [col for col in df.columns if df[col].dtype == 'bool' or
                 (df[col].dtype == 'object' and df[col].str.upper().isin(['TRUE', 'FALSE']).any())]

    if bool_cols:
        filter_col = bool_cols[-1]
        print(f"Found filter column: {filter_col}")

        if df[filter_col].dtype == 'object':
            df[filter_col] = df[filter_col].str.upper() == 'TRUE'

        # Split into filtered (keep) and removed dataframes
        filtered_df = df[df[filter_col] == False].copy()
        removed_df = df[df[filter_col] == True].copy()
    else:
        print("No boolean filter column found. Calculating manually...")
        # Keep rows where wingtip is darker (lower intensity) than wing
        filtered_df = df[df['mean_wingtip_intensity'] < df['mean_wing_intensity']].copy()
        removed_df = df[df['mean_wingtip_intensity'] >= df['mean_wing_intensity']].copy()

    print(f"Total images: {initial_count}")
    print(f"Filtered images (wingtip darker): {len(filtered_df)}")
    print(f"Removed images (wing darker or equal): {len(removed_df)}\n")

    # Ensure dark pixel percentage for all dataframes
    filtered_df = ensure_dark_pixel_percentage(filtered_df)
    removed_df = ensure_dark_pixel_percentage(removed_df)

    # Save image lists to CSV
    OUTPUT_DIR_FILTERED.mkdir(exist_ok=True, parents=True)

    # Check if image_name column exists
    if 'image_na' in filtered_df.columns:
        image_col = 'image_na'
    elif 'image_name' in filtered_df.columns:
        image_col = 'image_name'
    elif 'species' in filtered_df.columns:
        # Use index as image identifier
        image_col = None
    else:
        image_col = None

    if image_col:
        # Save filtered images list
        filtered_list = filtered_df[[image_col, 'species', 'mean_wing_intensity', 'mean_wingtip_intensity']].copy()
        filtered_list.to_csv(FILTERED_CSV, index=False)
        print(f"Saved filtered images list to {FILTERED_CSV}")

        # Save removed images list
        removed_list = removed_df[[image_col, 'species', 'mean_wing_intensity', 'mean_wingtip_intensity']].copy()
        removed_list.to_csv(REMOVED_CSV, index=False)
        print(f"Saved removed images list to {REMOVED_CSV}\n")
    else:
        print("Warning: No image name column found. Saving with index only.")
        filtered_df[['species', 'mean_wing_intensity', 'mean_wingtip_intensity']].to_csv(FILTERED_CSV)
        removed_df[['species', 'mean_wing_intensity', 'mean_wingtip_intensity']].to_csv(REMOVED_CSV)

    return filtered_df, unfiltered_df, removed_df


def ensure_dark_pixel_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure pct_darker_pixels exists by calculating it from available columns.
    """
    if "pct_darker_pixels" in df.columns and df["pct_darker_pixels"].notna().any():
        return df

    candidate_pairs = [
        ("darker_pixel_count", "wingtip_pixel_count"),
        ("dark_lt_30", "wingtip_pixel_count"),
        ("method3_count", "total_filtered_pixels"),
    ]

    for numerator, denominator in candidate_pairs:
        if numerator in df.columns and denominator in df.columns:
            denom_values = df[denominator].replace(0, np.nan)
            df["pct_darker_pixels"] = (df[numerator] / denom_values) * 100
            df["pct_darker_pixels"] = df["pct_darker_pixels"].clip(0, 100)
            print(f"Calculated pct_darker_pixels using {numerator} / {denominator}")
            return df

    print("Warning: Could not determine pct_darker_pixels; box plot will be skipped.")
    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print console summary of record counts and mean intensities."""
    total_rows = len(df)
    print("\nDATA SUMMARY: Filtered data (wingtip darker than wing)")
    print(f"Total images / scatter dots: {total_rows}")

    grouped = df.groupby("species")
    for species, subset in grouped:
        label = format_species_label(species)
        count = len(subset)
        wing_mean = subset["mean_wing_intensity"].mean()
        wingtip_mean = subset["mean_wingtip_intensity"].mean()
        print(
            f"  {label}: n={count}, "
            f"upperparts mean={wing_mean:.2f}, primaries mean={wingtip_mean:.2f}"
        )

    overall_wing = df["mean_wing_intensity"].mean()
    overall_wingtip = df["mean_wingtip_intensity"].mean()
    print(
        f"Overall means → upperparts: {overall_wing:.2f}, "
        f"primaries: {overall_wingtip:.2f}\n"
    )


def plot_upperwing_vs_wingtip(df: pd.DataFrame, output_path: Path, title_suffix: str = "") -> None:
    """Generate the scatter plot with equal axes and a 1:1 reference line."""
    output_path.parent.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(10, 10))
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
        label="Upperparts = Primaries",
    )

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Upperparts mean intensity (0–255)", fontsize=20, fontweight='bold')
    ax.set_ylabel("Primaries mean intensity (0–255)", fontsize=20, fontweight='bold')
    ax.set_title(
        f"Relationship Between Upperparts and Primaries Tones{title_suffix}",
        fontsize=22,
        fontweight="bold",
    )

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Tidy legend with formatted species labels plus the 1:1 reference line
    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = []
    for label in labels:
        if label in SPECIES_COLORS:
            formatted_labels.append(format_species_label(label))
        else:
            formatted_labels.append(label)
    ax.legend(handles, formatted_labels, title="Species", frameon=True, fontsize=16, title_fontsize=17)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved scatter plot to {output_path}")


def plot_species_bar_chart(df: pd.DataFrame, output_path: Path, title_suffix: str = "") -> None:
    """Generate grouped bar chart of per-species mean wing vs wingtip intensity."""
    output_path.parent.mkdir(exist_ok=True, parents=True)

    summary = (
        df.groupby("species")[["mean_wing_intensity", "mean_wingtip_intensity"]]
        .mean()
        .reset_index()
        .melt(id_vars="species", var_name="metric", value_name="intensity")
    )
    summary["metric"] = summary["metric"].map(METRIC_LABELS)
    summary["species_display"] = summary["species"].apply(format_species_label)

    counts = df.groupby("species").size().to_dict()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=summary,
        x="species_display",
        y="intensity",
        hue="metric",
        palette=METRIC_COLORS,
        ax=ax,
    )

    ax.set_ylabel("Mean intensity (0–255)", fontsize=18, fontweight='bold')
    ax.set_xlabel("Species", fontsize=18, fontweight='bold')
    ax.set_ylim(0, 270)
    ax.set_title(
        f"Average Upperparts vs Primaries Intensity{title_suffix}",
        fontsize=20,
        fontweight="bold",
    )

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.legend(title="Metric", frameon=True, fontsize=14, title_fontsize=15)

    for idx, species in enumerate(summary["species"].unique()):
        label = format_species_label(species)
        count = counts.get(species, 0)
        ax.text(
            idx,
            262,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved bar plot to {output_path}")


def plot_dark_pixel_boxplot(df: pd.DataFrame, output_path: Path, title_suffix: str = "") -> None:
    """Create a box plot of dark pixel percentages per species - styled like reference."""
    column = "pct_darker_pixels"
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found. Skipping dark pixel box plot.")
        return

    output_path.parent.mkdir(exist_ok=True, parents=True)

    # Prepare data
    box_df = df.copy()
    box_df["species_display"] = box_df["species"].apply(format_species_label)

    # Clean data: remove any NaN values
    box_df = box_df.dropna(subset=[column, 'species_display']).copy()
    box_df[column] = pd.to_numeric(box_df[column], errors='coerce')
    box_df = box_df.dropna(subset=[column])

    # Ensure percentage data is within valid bounds (0-100%)
    box_df[column] = box_df[column].clip(0, 100)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))

    # Get species order and colors
    species_order = sorted(box_df['species_display'].unique())
    n_species = len(species_order)

    # Use alternating blue and orange
    colors = []
    for i in range(n_species):
        if i % 2 == 0:
            colors.append(SPECIES_COLORS['Glaucous_Winged_Gull'])  # Blue
        else:
            colors.append(SPECIES_COLORS['Slaty_Backed_Gull'])  # Orange

    # Set meaningful y-axis limits
    y_min = max(0, box_df[column].min() - 2)
    y_max = min(100, box_df[column].max() + 2)

    if box_df[column].max() > 98:
        y_max = 101
    if box_df[column].min() < 2:
        y_min = -1

    # Create box plot
    sns.boxplot(
        data=box_df,
        x='species_display',
        y=column,
        ax=ax,
        palette=colors,
        order=species_order,
        medianprops={"color": "black", "linewidth": 2.2},
        whiskerprops={"linewidth": 1.4},
        boxprops={"linewidth": 1.4},
    )

    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Species', fontsize=18, fontweight='bold')
    ax.set_ylabel('Dark Pixel Percentage (%)', fontsize=18, fontweight='bold')
    ax.set_title(f'Primaries Dark Pixel Analysis by Species{title_suffix}',
                 fontsize=20, fontweight='bold', pad=12)

    # Increase tick label sizes - especially x-axis species names
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=18)  # Larger species names

    ax.grid(True, alpha=0.3)

    ymin = 0
    ymax = min(100, box_df[column].max() + 5)
    ax.set_ylim(ymin, ymax)

    # Add horizontal reference line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(0.02, 50.5, '50%', transform=ax.get_yaxis_transform(),
            fontsize=11, alpha=0.7, color='gray')

    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved dark pixel box plot to {output_path}")

    # Calculate and print summary statistics
    summary_stats = box_df.groupby('species_display')[column].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)

    print("\n" + "=" * 80)
    print(f"DARK PIXEL PERCENTAGE SUMMARY STATISTICS{title_suffix}")
    print("=" * 80)
    print(summary_stats)
    print("=" * 80 + "\n")


def create_histogram_kde(data, bins, color, alpha=0.6):
    """
    Create histogram and KDE smooth curve for intensity distribution.
    """
    # Calculate histogram for bars (using density=True)
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Use KDE for smooth curve
    try:
        kde = stats.gaussian_kde(data, bw_method='scott')
        x_smooth = np.linspace(data.min(), data.max(), 200)
        y_smooth = kde(x_smooth)
    except Exception as e:
        # Fallback if KDE fails
        x_smooth = bin_centers
        y_smooth = counts

    return bin_centers, counts, x_smooth, y_smooth


def add_intensity_reference(ax, bins):
    """Add intensity reference boxes below the x-axis (grayscale gradient)"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    box_height = ylim[1] * 0.08
    box_y_position = ylim[0] - box_height * 2

    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        intensity_value = bin_center / 255.0
        gray_color = str(intensity_value)

        rect = Rectangle((bin_start, box_y_position), BIN_WIDTH, box_height,
                         facecolor=gray_color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

        # Only show numbers at 0, 15, 30, 45, etc. (not the middle values like 7, 22, 37)
        # These are the bin centers, but we only want to label the bin edges
        # So we skip labeling here entirely - labels are handled by x-axis ticks

    ax.set_ylim(box_y_position - box_height * 1.5, ylim[1])


def plot_intensity_distribution(df: pd.DataFrame, column: str, title: str, output_path: Path, title_suffix: str = "") -> None:
    """
    Create intensity distribution histogram with KDE curves matching reference style.
    """
    output_path.parent.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_title(f'{title}{title_suffix}',
                 fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel('Mean Intensity (0-255)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Density', fontsize=20, fontweight='bold')

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Plot histogram and KDE for each species
    for species in df['species'].unique():
        species_data = df[df['species'] == species][column]
        color = SPECIES_COLORS[species]
        display_name = format_species_label(species)

        # Create histogram bars
        counts, bins, patches = ax.hist(
            species_data,
            bins=INTENSITY_BINS,
            alpha=0.6,
            color=color,
            density=True,
            edgecolor='black',
            linewidth=0.8,
            label=display_name
        )

        # Create smooth KDE curve
        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_kde(
            species_data, INTENSITY_BINS, color
        )

        ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                alpha=0.9, linestyle='-')

    # Add mean lines and statistics for each species
    for species in df['species'].unique():
        species_data = df[df['species'] == species][column]
        color = SPECIES_COLORS[species]
        mean_val = species_data.mean()
        std_val = species_data.std()

        ax.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax.text(
            mean_val, 0.85,
            f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
            transform=ax.get_xaxis_transform(),
            color=color,
            fontweight='bold',
            ha='center',
            va='top',
            fontsize=12,
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                alpha=0.8,
                edgecolor=color
            )
        )

    # Add intensity reference gradient at bottom
    add_intensity_reference(ax, INTENSITY_BINS)

    # Configure legend - larger font at top right
    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.0, 0.98),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=18,
        title_fontsize=19
    )

    # Set x-axis ticks - only show major ticks (0, 15, 30, etc.)
    x_ticks = list(range(0, 256, 15))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{i}' for i in x_ticks], rotation=0, ha='center', fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 255)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved intensity distribution plot to {output_path}")


def main() -> None:
    # Load data and get filtered, unfiltered, and removed dataframes
    filtered_df, unfiltered_df, removed_df = load_intensity_data()

    if filtered_df.empty and unfiltered_df.empty:
        raise ValueError("No intensity records found for the configured species.")

    # Print summaries
    print("\n" + "="*80)
    print("UNFILTERED DATA SUMMARY (All Images)")
    print("="*80)
    print_dataset_summary(unfiltered_df)

    print("\n" + "="*80)
    print("FILTERED DATA SUMMARY (Primaries darker than Upperparts)")
    print("="*80)
    print_dataset_summary(filtered_df)

    print("\n" + "="*80)
    print("REMOVED DATA SUMMARY (Upperparts darker than or equal to Primaries)")
    print("="*80)
    print_dataset_summary(removed_df)

    # Generate plots for FILTERED data (WITHOUT suffix in titles)
    print("\n" + "="*80)
    print("GENERATING FILTERED PLOTS...")
    print("="*80)
    plot_upperwing_vs_wingtip(filtered_df, SCATTER_OUTPUT_FILTERED)
    plot_species_bar_chart(filtered_df, BAR_OUTPUT_FILTERED)
    plot_dark_pixel_boxplot(filtered_df, DARK_BOX_OUTPUT_FILTERED)
    plot_intensity_distribution(
        filtered_df,
        'mean_wing_intensity',
        'Upperparts Intensity Distribution',
        UPPERPARTS_DIST_OUTPUT_FILTERED
    )
    plot_intensity_distribution(
        filtered_df,
        'mean_wingtip_intensity',
        'Primaries Intensity Distribution',
        PRIMARIES_DIST_OUTPUT_FILTERED
    )

    # Generate plots for UNFILTERED data
    print("\n" + "="*80)
    print("GENERATING UNFILTERED PLOTS...")
    print("="*80)
    plot_upperwing_vs_wingtip(unfiltered_df, SCATTER_OUTPUT_UNFILTERED)
    plot_species_bar_chart(unfiltered_df, BAR_OUTPUT_UNFILTERED)
    plot_dark_pixel_boxplot(unfiltered_df, DARK_BOX_OUTPUT_UNFILTERED)
    plot_intensity_distribution(
        unfiltered_df,
        'mean_wing_intensity',
        'Upperparts Intensity Distribution',
        UPPERPARTS_DIST_OUTPUT_UNFILTERED
    )
    plot_intensity_distribution(
        unfiltered_df,
        'mean_wingtip_intensity',
        'Primaries Intensity Distribution',
        PRIMARIES_DIST_OUTPUT_UNFILTERED
    )

    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nFiltered plots saved to: {OUTPUT_DIR_FILTERED}")
    print(f"Unfiltered plots saved to: {OUTPUT_DIR_UNFILTERED}")
    print(f"\nImage lists saved:")
    print(f"  - Filtered images: {FILTERED_CSV}")
    print(f"  - Removed images: {REMOVED_CSV}")
    print("="*80)


if __name__ == "__main__":
    main()