"""
Summarize upperwing and filtered wingtip intensities using the enhanced CSV
(`Wingtip_Intensity_Distribution_Enhanced/wingtip_intensity_distribution_enhanced.csv`).
"""

from pathlib import Path

import pandas as pd

SPECIES = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = (
    BASE_DIR.parent  # Intensity/
    / "Wingtip_Intensity_Distribution_Enhanced"
    / "wingtip_intensity_distribution_enhanced.csv"
)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[df["species"].isin(SPECIES)]
    df = df.dropna(subset=["mean_wing_intensity", "mean_wingtip_intensity"])
    return df


def summarize(df: pd.DataFrame) -> None:
    grouped = df.groupby("species")
    means = grouped[["mean_wing_intensity", "mean_wingtip_intensity"]].mean()
    counts = grouped.size()

    print("Per-species mean intensities (filtered dataset):")
    print(
        means.rename(
            columns={
                "mean_wing_intensity": "upperwing_mean",
                "mean_wingtip_intensity": "wingtip_mean",
            }
        )
    )

    print("\nImages per species:")
    print(counts)

    print("\nOverall averages:")
    print(f"Upperwing mean intensity: {df['mean_wing_intensity'].mean():.2f}")
    print(f"Wingtip mean intensity:  {df['mean_wingtip_intensity'].mean():.2f}")
    print(f"Total images: {len(df)}")


def main() -> None:
    df = load_data()
    if df.empty:
        raise ValueError("No rows available for configured species.")
    summarize(df)


if __name__ == "__main__":
    main()

