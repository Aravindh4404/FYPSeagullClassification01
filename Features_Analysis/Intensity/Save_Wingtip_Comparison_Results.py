import os
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

from Features_Analysis.config import *

def analyze_wingtip_darkness(image_path, seg_path, species, file_name, wing_mean):
    """Analyzes wingtip darkness compared to wing mean intensity"""
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

    # Extract wingtip region from the normalized grayscale image
    wingtip_region, wingtip_mask = extract_region(gray_img, segmentation_img, "wingtip")
    
    wingtip_pixels = wingtip_region[wingtip_mask > 0]

    if len(wingtip_pixels) == 0:
        print(f"No wingtip region found in {file_name}")
        return None

    # Calculate darkness metrics
    darker_pixels = np.sum(wingtip_pixels < wing_mean) #wingtip pixels darker than mean of wing
    total_pixels = len(wingtip_pixels)
    percentage_darker = (darker_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    return {
        'image_name': file_name,
        'species': species,
        'percentage_darker': percentage_darker,
        'darker_pixel_count': darker_pixels,
        'total_wingtip_pixels': total_pixels
    }


def main():
    # Load wing data first
    wing_df = pd.read_csv("Wing_Greyscale_Intensity_Results/wing_intensity_analysis.csv")
    wing_data = wing_df.set_index('image_name')['mean_intensity'].to_dict()

    all_results = []

    for species_name, paths in SPECIES.items():
        print(f"\nProcessing {species_name} wingtip darkness...")
        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            if file_name not in wing_data:
                print(f"Skipping {file_name} - no wing data found")
                continue

            print(f" Processing {i + 1}/{len(image_paths[:S])}: {file_name}")
            result = analyze_wingtip_darkness(
                img_path, seg_path, species_name, file_name, wing_data[file_name]
            )
            if result:
                all_results.append(result)

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        os.makedirs("Darkness_Analysis_Results", exist_ok=True)
        output_path = os.path.join("Darkness_Analysis_Results", "wingtip_darkness_analysis.csv")
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    else:
        print("No results generated. Check input data.")


if __name__ == "__main__":
    main()
