import os
import numpy as np
import pandas as pd
import cv2
from scipy.stats import skew, kurtosis
from Features_Analysis.config import *

def analyze_wingtip_intensity(image_path, seg_path, species, file_name):
    """Analyzes wingtip intensity for a single image"""
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Extract wingtip region
    wingtip_region, wingtip_mask = extract_region(original_img, segmentation_img, "wingtip")

    # Convert to grayscale
    gray_wingtip = cv2.cvtColor(wingtip_region, cv2.COLOR_BGR2GRAY)

    # Get non-zero pixels (wingtip area)
    wingtip_pixels = gray_wingtip[wingtip_mask > 0]

    if len(wingtip_pixels) == 0:
        print(f"No wingtip region found in {file_name}")
        return None

    # Calculate intensity metrics
    return {
        'image_name': file_name,
        'species': species,
        'mean_intensity': np.mean(wingtip_pixels),
        'std_intensity': np.std(wingtip_pixels),
        'median_intensity': np.median(wingtip_pixels),
        'min_intensity': np.min(wingtip_pixels),
        'max_intensity': np.max(wingtip_pixels),
        'skewness': skew(wingtip_pixels),
        'kurtosis': kurtosis(wingtip_pixels),
        'pixel_count': len(wingtip_pixels)
    }

def main():
    all_results = []

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} wingtip intensity...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f"  Processing image {i + 1}/{S}: {file_name}")

            stats = analyze_wingtip_intensity(img_path, seg_path, species_name, file_name)
            if stats:
                all_results.append(stats)

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)

        os.makedirs("Intensity_Results", exist_ok=True)

        # Save as CSV
        csv_path = "Intensity_Results/wingtip_intensity_analysis.csv"
        df.to_csv(csv_path, index=False)

        # Save as Pickle
        pkl_path = "Intensity_Results/wingtip_intensity_analysis.pkl"
        df.to_pickle(pkl_path)

        print(f"\nResults saved to:\n- {csv_path}\n- {pkl_path}")
    else:
        print("No results generated. Check if wingtip regions were detected.")

if __name__ == "__main__":
    main()
