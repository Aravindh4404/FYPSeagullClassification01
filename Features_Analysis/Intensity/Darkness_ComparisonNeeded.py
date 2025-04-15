import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy.stats import skew, kurtosis

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from Features_Analysis.config import *  # Import configuration file
from Features_Analysis.image_normalization import minmax_normalize

def analyze_wingtip_darkness(image_path, seg_path, species, file_name):
    """
    Analyzes wingtip darkness using normalized intensity values (0-1 scale).
    Normalizes entire image first, then extracts regions.
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Convert to grayscale and normalize entire image
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    normalized_img = minmax_normalize(gray_img)  # Normalize entire image first

    # Extract masks for wing and wingtip regions using your config function
    _, wing_mask = extract_region(original_img, segmentation_img, "wing")
    _, wingtip_mask = extract_region(original_img, segmentation_img, "wingtip")

    # Get normalized pixel values using masks
    wing_pixels = normalized_img[wing_mask > 0]
    wingtip_pixels = normalized_img[wingtip_mask > 0]

    if len(wing_pixels) == 0 or len(wingtip_pixels) == 0:
        print(f"No wing/wingtip found in {file_name}")
        return None

    # Calculate normalized intensity metrics
    mean_wing = np.mean(wing_pixels)
    darker_pixels = wingtip_pixels[wingtip_pixels < mean_wing]

    return {
        "image_name": file_name,
        "species": species,
        "mean_wing_intensity": mean_wing,
        "std_wing_intensity": np.std(wing_pixels),
        "mean_wingtip_intensity": np.mean(wingtip_pixels),
        "std_wingtip_intensity": np.std(wingtip_pixels),
        "percentage_darker": (len(darker_pixels)/len(wingtip_pixels))*100 if len(wingtip_pixels) > 0 else 0,
        "mean_darker_wingtip_intensity": np.mean(darker_pixels) if len(darker_pixels) > 0 else 0,
        "std_darker_wingtip_intensity": np.std(darker_pixels) if len(darker_pixels) > 0 else 0,
        "wing_pixel_count": len(wing_pixels),
        "wingtip_pixel_count": len(wingtip_pixels),
        "darker_wingtip_pixels": len(darker_pixels),
    }

def main():
    """
    Process all images for both species and save results.
    """
    results = []

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} images...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f" Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")

            stats = analyze_wingtip_darkness(img_path, seg_path, species_name, file_name)

            if stats:
                results.append(stats)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)

        results_dir = "Darkness_Comparison_Analysis_Results"
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "wingtip_darkness_analysis.csv")
        df.to_csv(csv_path, index=False)

        print(f"\nResults saved to: {csv_path}")

        # Calculate averages by species
        species_avg = df.groupby('species')[
            ['percentage_darker', 'mean_wing_intensity', 'std_wing_intensity',
             'mean_wingtip_intensity', 'std_wingtip_intensity',
             'mean_darker_wingtip_intensity', 'std_darker_wingtip_intensity']
        ].mean().reset_index()

        avg_csv_path = os.path.join(results_dir, "wingtip_darkness_averages.csv")
        species_avg.to_csv(avg_csv_path, index=False)

        print(f"\nSpecies averages saved to: {avg_csv_path}")

if __name__ == "__main__":
    main()