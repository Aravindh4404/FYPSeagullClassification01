import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy.stats import skew, kurtosis

import sys

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
from Features_Analysis.config import *

def analyze_wing_intensity(image_path, seg_path, species, file_name):
    """Analyzes wing intensity for a single image"""
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

    # Extract wing region from the normalized grayscale image
    wing_region, wing_mask = extract_region(gray_img, segmentation_img, "wing")

    # Get non-zero pixels (wing area)
    wing_pixels = wing_region[wing_mask > 0]

    if len(wing_pixels) == 0:
        print(f"No wing region found in {file_name}")
        return None

    # Calculate intensity metrics
    return {
        'image_name': file_name,
        'species': species,
        'mean_intensity': np.mean(wing_pixels),
        'std_intensity': np.std(wing_pixels),
        'median_intensity': np.median(wing_pixels),
        'min_intensity': np.min(wing_pixels),
        'max_intensity': np.max(wing_pixels),
        'skewness': skew(wing_pixels),
        'kurtosis': kurtosis(wing_pixels),
        'pixel_count': len(wing_pixels)
    }


def main():
    all_results = []

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} images...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f"  Processing image {i + 1}/{S}: {file_name}")

            stats = analyze_wing_intensity(img_path, seg_path, species_name, file_name)
            if stats:
                all_results.append(stats)

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)

        os.makedirs("Intensity_Results", exist_ok=True)

        # Save as CSV
        csv_path = "Intensity_Results/wing_intensity_analysis.csv"
        df.to_csv(csv_path, index=False)

        # Save as Pickle
        pkl_path = "Intensity_Results/wing_intensity_analysis.pkl"
        df.to_pickle(pkl_path)

        print(f"\nResults saved to:\n- {csv_path}\n- {pkl_path}")
    else:
        print("No results generated. Check if wing regions were detected.")


if __name__ == "__main__":
    main()
