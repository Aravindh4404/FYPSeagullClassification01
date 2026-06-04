"""
Utility script to report wing and wingtip mean intensities along with image
counts per species, using the same CSV that powers the upperwing-vs-wingtip
scatter plot (filtered wingtip intensities).
"""

from pathlib import Path

import pandas as pd

SPECIES = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Wingtip_Dark_Pixel_Visualizations" / "dark_pixel_results_all_images.csv"


def load_filtered_data() -> pd.DataFrame:
    """Load the CSV and keep only rows we can use for summary statistics."""
    df = pd.read_csv(CSV_PATH)
    df = df[df["species"].isin(SPECIES)]
    df = df.dropna(subset=["mean_wing_intensity", "mean_wingtip_intensity"])
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print per-species means, counts, and overall averages."""
    grouped = df.groupby("species")
    mean_stats = grouped[["mean_wing_intensity", "mean_wingtip_intensity"]].mean()
    counts = grouped.size()

    print("Per-species mean intensities (filtered white spots removed):")
    print(mean_stats.rename(columns={
        "mean_wing_intensity": "upperwing_mean",
        "mean_wingtip_intensity": "wingtip_mean",
    }))

    print("\nImage count per species in CSV:")
    print(counts)

    print("\nOverall averages across both species:")
    print(f"Upperwing mean intensity: {df['mean_wing_intensity'].mean():.2f}")
    print(f"Wingtip mean intensity:  {df['mean_wingtip_intensity'].mean():.2f}")
    print(f"Total images used: {len(df)}")


def main() -> None:
    df = load_filtered_data()
    if df.empty:
        raise ValueError("No valid rows found for the configured species.")
    summarize(df)


if __name__ == "__main__":
    main()

