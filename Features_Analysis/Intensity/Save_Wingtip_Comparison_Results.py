import os
import numpy as np
import pandas as pd
import cv2
from scipy.stats import skew, kurtosis
from Features_Analysis.config import *
from Features_Analysis.Intensity.normalize_intensity_analysis import analyze_normalized_wingtip_darkness

from Features_Analysis.config import *


def analyze_wingtip_darkness(image_path, seg_path, species, file_name, wing_mean):
    """Analyzes wingtip darkness compared to wing mean intensity using normalized values"""
    return analyze_normalized_wingtip_darkness(image_path, seg_path, species, file_name, wing_mean)


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
