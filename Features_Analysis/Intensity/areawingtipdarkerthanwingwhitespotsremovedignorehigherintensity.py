import pandas as pd
import cv2
import numpy as np
import os
# Import configuration file
import sys
from pathlib import Path

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
from Features_Analysis.config import *


def analyze_wingtip_intensity_distribution_filtered(image_path, seg_path, species, file_name,
                                                    white_threshold=220, exclusion_method="threshold"):
    """
    Analyzes the intensity distribution of wingtip pixels with white spot removal.

    Args:
        image_path: Path to original image
        seg_path: Path to segmentation image
        species: Species name
        file_name: Image file name
        white_threshold: Intensity threshold above which pixels are considered white spots (default: 200)
        exclusion_method: Method for excluding white pixels ("threshold" or "percentile")
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Convert entire image to grayscale first
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Apply min-max normalization to the entire grayscale image
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

    # Extract wing and wingtip regions from the normalized grayscale image
    wing_region, wing_mask = extract_region(gray_img, segmentation_img, "wing")
    wingtip_region, wingtip_mask = extract_region(gray_img, segmentation_img, "wingtip")

    # Get wing pixels
    wing_pixels = wing_region[wing_mask > 0]

    if len(wing_pixels) == 0:
        print(f"No wing region found in {file_name}")
        return None

    # Calculate mean wing intensity
    mean_wing_intensity = np.mean(wing_pixels)

    # Get wingtip pixels from the normalized grayscale image
    wingtip_pixels = wingtip_region[wingtip_mask > 0]

    if len(wingtip_pixels) == 0:
        print(f"No wingtip region found in {file_name}")
        return None

    # FILTER OUT WHITE SPOTS/STREAKS
    if exclusion_method == "threshold":
        # Method 1: Simple threshold - exclude pixels above threshold
        filtered_wingtip_pixels = wingtip_pixels[wingtip_pixels < white_threshold]
    elif exclusion_method == "percentile":
        # Method 2: Percentile-based - exclude top 10% brightest pixels
        percentile_threshold = np.percentile(wingtip_pixels, 90)
        filtered_wingtip_pixels = wingtip_pixels[wingtip_pixels < percentile_threshold]
    else:
        # Default: use threshold method
        filtered_wingtip_pixels = wingtip_pixels[wingtip_pixels < white_threshold]

    # Count removed pixels
    removed_pixel_count = len(wingtip_pixels) - len(filtered_wingtip_pixels)
    removed_pixel_percentage = (removed_pixel_count / len(wingtip_pixels)) * 100

    if len(filtered_wingtip_pixels) == 0:
        print(f"No dark wingtip pixels found after filtering in {file_name}")
        return None

    # Define intensity ranges (bins)
    intensity_ranges = [
        (0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
        (50, 60), (60, 70), (70, 80), (80, 90), (90, 100),
        (100, 110), (110, 120), (120, 130), (130, 140), (140, 150),
        (150, 160), (160, 170), (170, 180), (180, 190), (190, 200),
        (200, 210), (210, 220), (220, 230), (230, 240), (240, 255)
    ]

    # Count pixels in each intensity range using FILTERED pixels
    range_counts = {}
    for start, end in intensity_ranges:
        # Count raw number of pixels in this range
        pixel_count = np.sum((filtered_wingtip_pixels >= start) & (filtered_wingtip_pixels < end))
        range_counts[f"intensity_{start}_{end}"] = pixel_count
        # Calculate percentage of FILTERED wingtip pixels in this range
        range_counts[f"pct_{start}_{end}"] = (pixel_count / len(filtered_wingtip_pixels)) * 100

    # Calculate wing-wingtip differences using FILTERED pixels
    # For each filtered wingtip pixel, calculate how much darker it is than the mean wing
    intensity_diffs = mean_wing_intensity - filtered_wingtip_pixels

    # Only keep positive differences (darker pixels)
    positive_diffs = intensity_diffs[intensity_diffs > 0]

    # Define difference thresholds
    diff_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Count pixels with differences above thresholds
    diff_counts = {}
    for threshold in diff_thresholds:
        pixel_count = np.sum(intensity_diffs > threshold)
        diff_counts[f"diff_gt_{threshold}"] = pixel_count
        diff_counts[f"pct_diff_gt_{threshold}"] = (pixel_count / len(filtered_wingtip_pixels)) * 100

    # Calculate statistics about very dark pixels using FILTERED pixels
    very_dark_counts = {}
    dark_thresholds = [30, 40, 50, 60]
    for threshold in dark_thresholds:
        pixel_count = np.sum(filtered_wingtip_pixels < threshold)
        very_dark_counts[f"dark_lt_{threshold}"] = pixel_count
        very_dark_counts[f"pct_dark_lt_{threshold}"] = (pixel_count / len(filtered_wingtip_pixels)) * 100

    # Prepare results with both original and filtered data
    results = {
        "image_name": file_name,
        "species": species,
        "mean_wing_intensity": mean_wing_intensity,

        # Original wingtip data (before filtering)
        "original_mean_wingtip_intensity": np.mean(wingtip_pixels),
        "original_wingtip_pixel_count": len(wingtip_pixels),

        # Filtered wingtip data (after removing white spots)
        "filtered_mean_wingtip_intensity": np.mean(filtered_wingtip_pixels),
        "filtered_wingtip_pixel_count": len(filtered_wingtip_pixels),

        # Filtering information
        "white_threshold_used": white_threshold,
        "removed_pixel_count": removed_pixel_count,
        "removed_pixel_percentage": removed_pixel_percentage,

        # Other statistics
        "wing_pixel_count": len(wing_pixels),
        "darker_pixel_count": len(positive_diffs),
        "pct_darker_pixels": (len(positive_diffs) / len(filtered_wingtip_pixels)) * 100,

        # Intensity distribution (based on filtered pixels)
        **range_counts,
        **diff_counts,
        **very_dark_counts
    }

    return results


def main():
    """
    Process all images for both species and save intensity distribution results with white spot filtering.
    """
    results = []

    # Configuration for white spot removal
    WHITE_THRESHOLD = 220  # Adjust this value as needed (200-240 range typically works well)
    EXCLUSION_METHOD = "threshold"  # or "percentile"

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} images...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f" Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")

            stats = analyze_wingtip_intensity_distribution_filtered(
                img_path, seg_path, species_name, file_name,
                white_threshold=WHITE_THRESHOLD, exclusion_method=EXCLUSION_METHOD
            )

            if stats:
                results.append(stats)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)

        results_dir = "Wingtip_Intensity_Distribution_Filtered220"
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "wingtip_intensity_distribution_filtered.csv")
        df.to_csv(csv_path, index=False)

        print(f"\nFiltered results saved to: {csv_path}")

        # Calculate averages by species
        # Select columns that are percentages for easier comparison
        pct_columns = [col for col in df.columns if col.startswith("pct_")]
        dark_columns = [col for col in df.columns if col.startswith("dark_")]
        diff_columns = [col for col in df.columns if col.startswith("diff_")]

        # Calculate species averages
        species_avg = df.groupby('species')[
            ['mean_wing_intensity', 'original_mean_wingtip_intensity', 'filtered_mean_wingtip_intensity',
             'removed_pixel_percentage'] + pct_columns + dark_columns + diff_columns
            ].mean().reset_index()

        avg_csv_path = os.path.join(results_dir, "wingtip_intensity_averages_filtered.csv")
        species_avg.to_csv(avg_csv_path, index=False)

        print(f"\nSpecies averages saved to: {avg_csv_path}")

        # Print summary comparison
        print("\n=== FILTERING SUMMARY ===")
        print("Original vs Filtered Mean Wingtip Intensity:")
        summary = df.groupby('species')[['original_mean_wingtip_intensity', 'filtered_mean_wingtip_intensity',
                                         'removed_pixel_percentage']].agg(['mean', 'std'])
        print(summary)


if __name__ == "__main__":
    main()
