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

def analyze_wingtip_darkness(image_path, seg_path, species, file_name, wing_mean=None):
    """
    Analyzes wingtip darkness compared to wing mean intensity using normalized values.
    
    Args:
        image_path: Path to the original image
        seg_path: Path to the segmentation mask
        species: Species name
        file_name: Name of the file
        wing_mean: Pre-calculated wing mean intensity (optional)
        
    Returns:
        dict: Dictionary containing wingtip darkness metrics
    """
    # Load and normalize the image
    normalized_img, segmentation_img = load_and_normalize_image(image_path, seg_path)
    
    if normalized_img is None or segmentation_img is None:
        return None
    
    # Extract wing and wingtip regions
    _, wing_mask = extract_region(cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR), 
                                 segmentation_img, "wing")
    _, wingtip_mask = extract_region(cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR), 
                                    segmentation_img, "wingtip")
    
    # Get normalized pixel values using masks
    wing_pixels = normalized_img[wing_mask > 0]
    wingtip_pixels = normalized_img[wingtip_mask > 0]
    
    if len(wing_pixels) == 0 or len(wingtip_pixels) == 0:
        print(f"No wing/wingtip found in {file_name}")
        return None
    
    # Calculate wing mean if not provided
    if wing_mean is None:
        wing_mean = np.mean(wing_pixels)
    
    # Calculate wingtip metrics
    mean_wingtip = np.mean(wingtip_pixels)
    std_wingtip = np.std(wingtip_pixels)
    
    # Calculate darker pixels (wingtip pixels darker than wing mean)
    darker_pixels = wingtip_pixels[wingtip_pixels < wing_mean]
    percentage_darker = (len(darker_pixels) / len(wingtip_pixels)) * 100
    
    # Calculate metrics for darker pixels
    mean_darker = np.mean(darker_pixels) if len(darker_pixels) > 0 else 0
    std_darker = np.std(darker_pixels) if len(darker_pixels) > 0 else 0
    
    # Calculate pixels with intensity differences above thresholds
    diff_thresholds = [25, 50, 75, 100]
    diff_counts = {}
    for threshold in diff_thresholds:
        pixel_count = np.sum(wingtip_pixels < (wing_mean - threshold))
        diff_counts[f"diff_gt_{threshold}"] = pixel_count
        diff_counts[f"pct_diff_gt_{threshold}"] = (pixel_count / len(wingtip_pixels)) * 100
    
    # Calculate very dark pixels (below specific thresholds)
    dark_thresholds = [30, 50, 75, 100]
    dark_counts = {}
    for threshold in dark_thresholds:
        pixel_count = np.sum(wingtip_pixels < threshold)
        dark_counts[f"dark_lt_{threshold}"] = pixel_count
        dark_counts[f"pct_dark_lt_{threshold}"] = (pixel_count / len(wingtip_pixels)) * 100
    
    # Count pixels in different intensity ranges
    intensity_ranges = [
        (0, 25), (25, 50), (50, 75), (75, 100),
        (100, 125), (125, 150), (150, 175), (175, 200),
        (200, 225), (225, 255)
    ]
    
    range_counts = {}
    for start, end in intensity_ranges:
        pixel_count = np.sum((wingtip_pixels >= start) & (wingtip_pixels < end))
        range_counts[f"intensity_{start}_{end}"] = pixel_count
        range_counts[f"pct_{start}_{end}"] = (pixel_count / len(wingtip_pixels)) * 100
    
    # Prepare results
    results = {
        "image_name": file_name,
        "species": species,
        "mean_wing_intensity": wing_mean,
        "mean_wingtip_intensity": mean_wingtip,
        "std_wingtip_intensity": std_wingtip,
        "percentage_darker": percentage_darker,
        "mean_darker_wingtip_intensity": mean_darker,
        "std_darker_wingtip_intensity": std_darker,
        "wing_pixel_count": len(wing_pixels),
        "wingtip_pixel_count": len(wingtip_pixels),
        "darker_wingtip_pixels": len(darker_pixels),
        **diff_counts,
        **dark_counts,
        **range_counts
    }
    
    return results

def main():
    """
    Process all images for both species and save wingtip darkness results.
    """
    # First, check if wing intensity data exists
    wing_intensity_file = "Wing_Intensity_Results_New/wing_intensity_analysis.csv"
    wing_data = {}
    
    if os.path.exists(wing_intensity_file):
        print(f"Loading wing intensity data from {wing_intensity_file}")
        wing_df = pd.read_csv(wing_intensity_file)
        wing_data = wing_df.set_index('image_name')['mean_intensity'].to_dict()
    else:
        print(f"Warning: Wing intensity file not found at {wing_intensity_file}")
        print("Will calculate wing mean intensity for each image individually.")
    
    all_results = []

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} wingtip darkness...")
        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            
            # Get wing mean if available
            wing_mean = wing_data.get(file_name)
            
            print(f" Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")
            stats = analyze_wingtip_darkness(img_path, seg_path, species_name, file_name, wing_mean)
            if stats:
                all_results.append(stats)

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Create output directory
        output_dir = "Wingtip_Darkness_Results_New"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, "wingtip_darkness_analysis.csv")
        df.to_csv(csv_path, index=False)
        
        # Save as Pickle
        pkl_path = os.path.join(output_dir, "wingtip_darkness_analysis.pkl")
        df.to_pickle(pkl_path)
        
        print(f"\nResults saved to:\n- {csv_path}\n- {pkl_path}")
        
        # Calculate and save species averages
        species_avg = df.groupby('species').agg({
            'percentage_darker': ['mean', 'std', 'min', 'max'],
            'mean_wing_intensity': ['mean', 'std'],
            'mean_wingtip_intensity': ['mean', 'std'],
            'mean_darker_wingtip_intensity': ['mean', 'std'],
            'wingtip_pixel_count': ['sum', 'mean'],
            'darker_wingtip_pixels': ['sum', 'mean']
        }).reset_index()
        
        avg_csv_path = os.path.join(output_dir, "wingtip_darkness_averages.csv")
        species_avg.to_csv(avg_csv_path, index=False)
        
        print(f"Species averages saved to: {avg_csv_path}")
    else:
        print("No results generated. Check if wingtip regions were detected.")

if __name__ == "__main__":
    main() 