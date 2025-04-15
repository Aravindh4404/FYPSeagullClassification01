import os
import numpy as np
import pandas as pd
import cv2
import sys
from pathlib import Path
from scipy.stats import skew, kurtosis

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from Features_Analysis.config import *
from Features_Analysis.image_normalization import minmax_normalize  # Import the same normalization used elsewhere


def analyze_wing_intensity(image_path, seg_path, species, file_name):
    """
    Analyzes wing intensity for a single image using the same consistent normalization
    approach used in other scripts.
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Convert entire image to grayscale and normalize once
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    normalized_img = minmax_normalize(gray_img)  # Normalize entire image first

    # Extract wing region using mask
    _, wing_mask = extract_region(original_img, segmentation_img, "wing")

    # Get normalized pixel values using mask
    wing_pixels = normalized_img[wing_mask > 0]

    if len(wing_pixels) == 0:
        print(f"No wing region found in {file_name}")
        return None

    # Calculate statistics directly from the normalized pixels
    mean_intensity = np.mean(wing_pixels)
    std_intensity = np.std(wing_pixels)
    median_intensity = np.median(wing_pixels)

    # Calculate additional statistics
    min_intensity = np.min(wing_pixels)
    max_intensity = np.max(wing_pixels)
    intensity_range = max_intensity - min_intensity

    # Calculate skewness and kurtosis (shape of distribution)
    pixel_skewness = skew(wing_pixels) if len(wing_pixels) > 2 else 0
    pixel_kurtosis = kurtosis(wing_pixels) if len(wing_pixels) > 3 else 0

    # Prepare results
    results = {
        "image_name": file_name,
        "species": species,
        "wing_pixel_count": len(wing_pixels),
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "median_intensity": median_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "intensity_range": intensity_range,
        "skewness": pixel_skewness,
        "kurtosis": pixel_kurtosis
    }

    return results


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

        os.makedirs("Intensity_Results/", exist_ok=True)

        # Save as CSV
        csv_path = "Intensity_Results//wing_intensity_analysis.csv"
        df.to_csv(csv_path, index=False)

        # Save as Pickle
        pkl_path = "Intensity_Results//wing_intensity_analysis.pkl"
        df.to_pickle(pkl_path)

        print(f"\nResults saved to:\n- {csv_path}\n- {pkl_path}")
    else:
        print("No results generated. Check if wing regions were detected.")


if __name__ == "__main__":
    main()