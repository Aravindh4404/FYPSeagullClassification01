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
from Features_Analysis.Intensity.normalize_intensity_analysis import analyze_normalized_wing_intensity

def analyze_wing_intensity(image_path, seg_path, species, file_name):
    """Analyzes wing intensity for a single image using normalized values"""
    return analyze_normalized_wing_intensity(image_path, seg_path, species, file_name)


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
