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
from Features_Analysis.Intensity_New.normalization_utils import load_and_normalize_image, extract_region

def analyze_wing_intensity(image_path, seg_path, species, file_name):
    """
    Analyzes wing intensity for a single image using normalized values.
    
    Args:
        image_path: Path to the original image
        seg_path: Path to the segmentation mask
        species: Species name
        file_name: Name of the file
        
    Returns:
        dict: Dictionary containing wing intensity metrics
    """
    # Load and normalize the image
    normalized_img, segmentation_img = load_and_normalize_image(image_path, seg_path)
    
    if normalized_img is None or segmentation_img is None:
        return None
    
    # Extract wing region
    _, wing_mask = extract_region(cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR), 
                                 segmentation_img, "wing")
    
    # Get normalized pixel values using mask
    wing_pixels = normalized_img[wing_mask > 0]
    
    if len(wing_pixels) == 0:
        print(f"No wing found in {file_name}")
        return None
    
    # Calculate intensity metrics
    mean_intensity = np.mean(wing_pixels)
    std_intensity = np.std(wing_pixels)
    median_intensity = np.median(wing_pixels)
    min_intensity = np.min(wing_pixels)
    max_intensity = np.max(wing_pixels)
    
    # Calculate distribution metrics
    skewness = skew(wing_pixels)
    kurt = kurtosis(wing_pixels)
    
    # Count pixels in different intensity ranges
    intensity_ranges = [
        (0, 25), (25, 50), (50, 75), (75, 100),
        (100, 125), (125, 150), (150, 175), (175, 200),
        (200, 225), (225, 255)
    ]
    
    range_counts = {}
    for start, end in intensity_ranges:
        pixel_count = np.sum((wing_pixels >= start) & (wing_pixels < end))
        range_counts[f"intensity_{start}_{end}"] = pixel_count
        range_counts[f"pct_{start}_{end}"] = (pixel_count / len(wing_pixels)) * 100
    
    # Prepare results
    results = {
        "image_name": file_name,
        "species": species,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "median_intensity": median_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "pixel_count": len(wing_pixels),
        **range_counts
    }
    
    return results

def main():
    """
    Process all images for both species and save wing intensity results.
    """
    all_results = []

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} wing intensity...")
        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f" Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")

            stats = analyze_wing_intensity(img_path, seg_path, species_name, file_name)
            if stats:
                all_results.append(stats)

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Create output directory using os.path.join
        output_dir = os.path.join(root_dir, "Wing_Intensity_Results_New")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV using os.path.join
        csv_path = os.path.join(output_dir, "wing_intensity_analysis.csv")
        df.to_csv(csv_path, index=False)
        
        # Save as Pickle using os.path.join
        pkl_path = os.path.join(output_dir, "wing_intensity_analysis.pkl")
        df.to_pickle(pkl_path)
        
        print(f"\nResults saved to:\n- {csv_path}\n- {pkl_path}")
        
        # Calculate and save species averages
        species_avg = df.groupby('species').agg({
            'mean_intensity': ['mean', 'std', 'min', 'max', 'median'],
            'std_intensity': ['mean', 'std'],
            'skewness': ['mean', 'std'],
            'kurtosis': ['mean', 'std'],
            'pixel_count': ['sum', 'mean']
        }).reset_index()
        
        avg_csv_path = os.path.join(output_dir, "wing_intensity_averages.csv")
        species_avg.to_csv(avg_csv_path, index=False)
        
        print(f"Species averages saved to: {avg_csv_path}")
    else:
        print("No results generated. Check if wing regions were detected.")

if __name__ == "__main__":
    main() 