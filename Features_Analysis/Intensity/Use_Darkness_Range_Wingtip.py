import pandas as pd
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from Features_Analysis.config import *  # Import configuration file
from Features_Analysis.Intensity.normalize_intensity_analysis import normalize_and_analyze_intensity


def analyze_wingtip_intensity_distribution(image_path, seg_path, species, file_name):
    """
    Analyzes the intensity distribution of wingtip pixels and their difference
    from the wing intensity for a single image using normalized values.
    """
    # Get normalized wing and wingtip regions
    wing_results = normalize_and_analyze_intensity(image_path, seg_path, species, file_name, "wing")
    wingtip_results = normalize_and_analyze_intensity(image_path, seg_path, species, file_name, "wingtip")
    
    if wing_results is None or wingtip_results is None:
        return None
        
    # Load images for pixel extraction
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)
    
    # Extract regions
    wing_region, wing_mask = extract_region(original_img, segmentation_img, "wing")
    wingtip_region, wingtip_mask = extract_region(original_img, segmentation_img, "wingtip")
    
    # Convert to grayscale and normalize
    gray_wing = cv2.cvtColor(wing_region, cv2.COLOR_BGR2GRAY)
    gray_wingtip = cv2.cvtColor(wingtip_region, cv2.COLOR_BGR2GRAY)
    
    # Get normalized pixels
    wing_pixels = gray_wing[wing_mask > 0]
    wingtip_pixels = gray_wingtip[wingtip_mask > 0]
    
    if len(wing_pixels) == 0 or len(wingtip_pixels) == 0:
        return None
        
    # Define intensity ranges (bins) for normalized values (0-255)
    intensity_ranges = [
        (0, 25), (25, 50), (50, 75), (75, 100),
        (100, 125), (125, 150), (150, 175), (175, 200),
        (200, 225), (225, 255)
    ]
    
    # Count pixels in each intensity range
    range_counts = {}
    for start, end in intensity_ranges:
        pixel_count = np.sum((wingtip_pixels >= start) & (wingtip_pixels < end))
        range_counts[f"intensity_{start}_{end}"] = pixel_count
        range_counts[f"pct_{start}_{end}"] = (pixel_count / len(wingtip_pixels)) * 100
    
    # Calculate wing-wingtip differences
    mean_wing_intensity = wing_results['mean_intensity']
    intensity_diffs = mean_wing_intensity - wingtip_pixels
    positive_diffs = intensity_diffs[intensity_diffs > 0]
    
    # Define difference thresholds for normalized values
    diff_thresholds = [25, 50, 75, 100]
    
    # Count pixels with differences above thresholds
    diff_counts = {}
    for threshold in diff_thresholds:
        pixel_count = np.sum(intensity_diffs > threshold)
        diff_counts[f"diff_gt_{threshold}"] = pixel_count
        diff_counts[f"pct_diff_gt_{threshold}"] = (pixel_count / len(wingtip_pixels)) * 100
    
    # Calculate statistics about very dark pixels
    very_dark_counts = {}
    dark_thresholds = [25, 50, 75]
    for threshold in dark_thresholds:
        pixel_count = np.sum(wingtip_pixels < threshold)
        very_dark_counts[f"dark_lt_{threshold}"] = pixel_count
        very_dark_counts[f"pct_dark_lt_{threshold}"] = (pixel_count / len(wingtip_pixels)) * 100
    
    # Prepare results
    results = {
        "image_name": file_name,
        "species": species,
        "mean_wing_intensity": mean_wing_intensity,
        "mean_wingtip_intensity": wingtip_results['mean_intensity'],
        "wing_pixel_count": len(wing_pixels),
        "wingtip_pixel_count": len(wingtip_pixels),
        "darker_pixel_count": len(positive_diffs),
        "pct_darker_pixels": (len(positive_diffs) / len(wingtip_pixels)) * 100,
        **range_counts,
        **diff_counts,
        **very_dark_counts
    }
    
    return results


def main():
    """
    Process all images for both species and save intensity distribution results.
    """
    results = []

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} images...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f" Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")

            stats = analyze_wingtip_intensity_distribution(img_path, seg_path, species_name, file_name)

            if stats:
                results.append(stats)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)

        results_dir = "Wingtip_Intensity_Distribution"
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "wingtip_intensity_distribution.csv")
        df.to_csv(csv_path, index=False)

        print(f"\nDetailed results saved to: {csv_path}")

        # Calculate averages by species
        # Select columns that are percentages for easier comparison
        pct_columns = [col for col in df.columns if col.startswith("pct_")]
        dark_columns = [col for col in df.columns if col.startswith("dark_")]
        diff_columns = [col for col in df.columns if col.startswith("diff_")]

        # Calculate species averages
        species_avg = df.groupby('species')[['mean_wing_intensity', 'mean_wingtip_intensity'] +
                                            pct_columns + dark_columns + diff_columns].mean().reset_index()

        avg_csv_path = os.path.join(results_dir, "wingtip_intensity_averages.csv")
        species_avg.to_csv(avg_csv_path, index=False)

        print(f"\nSpecies averages saved to: {avg_csv_path}")


if __name__ == "__main__":
    main()
