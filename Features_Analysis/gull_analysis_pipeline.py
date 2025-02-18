"""
Pipeline for Gull Regional Analysis

This script:
1. Loads images and segmentation maps.
2. Extracts region pixels for each of: wingtip, wing, body, and head.
3. Computes intensity statistics (mean, std, median, skewness, kurtosis) for each region.
4. Aggregates the results and produces frequency plots.
5. Prints a textual summary report.

Dependencies:
- Python 3.x
- numpy
- opencv-python (cv2)
- matplotlib
- scipy (for skew, kurtosis)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

###############################################################################
# CONFIGURATION
###############################################################################

# Paths to your data (adjust as needed)
SLATY_BACKED_IMG_DIR = r"C:\Users\Aravindh P\OneDrive - University of Nottingham Malaysia\FYP\FYPSeagullClassification01\Features_Analysis\Original_Images\Slaty_Backed_Gull"
SLATY_BACKED_SEG_DIR = r"C:\Users\Aravindh P\OneDrive - University of Nottingham Malaysia\FYP\FYPSeagullClassification01\Features_Analysis\Colored_Images\Slaty_Backed_Gull"

GLAUCOUS_WINGED_IMG_DIR = r"C:\Users\Aravindh P\OneDrive - University of Nottingham Malaysia\FYP\FYPSeagullClassification01\Features_Analysis\Original_Images\Glaucous_Winged_Gull"
GLAUCOUS_WINGED_SEG_DIR = r"C:\Users\Aravindh P\OneDrive - University of Nottingham Malaysia\FYP\FYPSeagullClassification01\Features_Analysis\Colored_Images\Glaucous_Winged_Gull"

# Number of images per species to process
S = 5

# Define the BGR colors for each region based on your RGB swatches:
REGION_COLORS = {
    "wingtip": (0, 255, 0),      # Green in RGB → (0, 255, 0) in BGR
    "wing":    (0, 0, 255),      # Red in RGB → (0, 0, 255) in BGR
    "body":    (0, 255, 255),    # Yellow in RGB → (0, 255, 255) in BGR
    "head":    (255, 255, 0)   # Sky Blue (e.g., RGB (135,206,235)) → (235,206,135) in BGR
}

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def load_images_and_seg_maps(sb_img_dir, sb_seg_dir, gw_img_dir, gw_seg_dir, s):
    """
    Loads up to s images and segmentation maps for both species.
    """
    sb_images = sorted(os.listdir(sb_img_dir))[:s]
    sb_segs   = sorted(os.listdir(sb_seg_dir))[:s]
    gw_images = sorted(os.listdir(gw_img_dir))[:s]
    gw_segs   = sorted(os.listdir(gw_seg_dir))[:s]

    iSB, sSB = [], []
    iGW, sGW = [], []

    for img_name, seg_name in zip(sb_images, sb_segs):
        img_path = os.path.join(sb_img_dir, img_name)
        seg_path = os.path.join(sb_seg_dir, seg_name)
        img = cv2.imread(img_path)  # BGR
        seg = cv2.imread(seg_path)  # BGR
        if img is not None and seg is not None:
            iSB.append(img)
            sSB.append(seg)
        else:
            print(f"[Warning] Could not load: {img_name}, {seg_name}")

    for img_name, seg_name in zip(gw_images, gw_segs):
        img_path = os.path.join(gw_img_dir, img_name)
        seg_path = os.path.join(gw_seg_dir, seg_name)
        img = cv2.imread(img_path)  # BGR
        seg = cv2.imread(seg_path)  # BGR
        if img is not None and seg is not None:
            iGW.append(img)
            sGW.append(seg)
        else:
            print(f"[Warning] Could not load: {img_name}, {seg_name}")

    return iSB, sSB, iGW, sGW

def extract_region_pixels(image, seg_map, region_color):
    """
    Extracts pixels from the image corresponding to a region marked by the given color.
    """
    if image is None or seg_map is None:
        return np.array([])

    mask = (
        (seg_map[:, :, 0] == region_color[0]) &
        (seg_map[:, :, 1] == region_color[1]) &
        (seg_map[:, :, 2] == region_color[2])
    )
    pixels = image[mask]

    if pixels.size == 0:
        print(f"[Info] No pixels found for region color {region_color}")
    return pixels

def compute_statistics(pixel_values):
    """
    Computes intensity statistics on the pixel values.
    If pixel_values has shape (N,3) (color), it converts to grayscale by averaging.
    Returns a dictionary with mean, std, median, skew, and kurtosis.
    """
    if pixel_values.size == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'hist': None, 'skew': np.nan, 'kurtosis': np.nan
        }

    # Convert to grayscale (average the channels) if necessary.
    if len(pixel_values.shape) == 2 and pixel_values.shape[1] == 3:
        pixel_values = pixel_values.mean(axis=1)

    mean_val = np.mean(pixel_values)
    std_val  = np.std(pixel_values)
    med_val  = np.median(pixel_values)
    skew_val = skew(pixel_values)
    kurt_val = kurtosis(pixel_values)

    hist, bin_edges = np.histogram(pixel_values, bins=256, range=(0, 255))
    return {
        'mean': mean_val,
        'std': std_val,
        'median': med_val,
        'hist': (hist, bin_edges),
        'skew': skew_val,
        'kurtosis': kurt_val
    }

def error_bar(values, method='std'):
    """
    Computes an error bar (std or sem) for a list of values.
    """
    arr = np.array(values)
    if arr.size == 0:
        return np.nan
    if method == 'std':
        return np.std(arr)
    elif method == 'sem':
        return np.std(arr) / np.sqrt(len(arr))
    return np.nan

###############################################################################
# MAIN PIPELINE
###############################################################################

def main():
    # Load images and segmentation maps for both species
    iSB, sSB, iGW, sGW = load_images_and_seg_maps(
        SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR,
        GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR,
        S
    )

    # Prepare dictionaries to store stats for each region.
    # For each species, we store lists of statistics for each region.
    stats_sb = {}
    stats_gw = {}
    for region in REGION_COLORS:
        stats_sb[region] = {"means": [], "stds": [], "medians": [], "skews": [], "kurts": []}
        stats_gw[region] = {"means": [], "stds": [], "medians": [], "skews": [], "kurts": []}

    num_images = min(len(iSB), len(iGW))
    for idx in range(num_images):
        for region, color in REGION_COLORS.items():
            # Process Slaty-backed images:
            pixels_sb = extract_region_pixels(iSB[idx], sSB[idx], color)
            region_stats_sb = compute_statistics(pixels_sb)
            stats_sb[region]["means"].append(region_stats_sb['mean'])
            stats_sb[region]["stds"].append(region_stats_sb['std'])
            stats_sb[region]["medians"].append(region_stats_sb['median'])
            stats_sb[region]["skews"].append(region_stats_sb['skew'])
            stats_sb[region]["kurts"].append(region_stats_sb['kurtosis'])

            # Process Glaucous-winged images:
            pixels_gw = extract_region_pixels(iGW[idx], sGW[idx], color)
            region_stats_gw = compute_statistics(pixels_gw)
            stats_gw[region]["means"].append(region_stats_gw['mean'])
            stats_gw[region]["stds"].append(region_stats_gw['std'])
            stats_gw[region]["medians"].append(region_stats_gw['median'])
            stats_gw[region]["skews"].append(region_stats_gw['skew'])
            stats_gw[region]["kurts"].append(region_stats_gw['kurtosis'])

    # Aggregate the statistics for each region (compute mean and error bars)
    aggregated_stats_sb = {}
    aggregated_stats_gw = {}
    for region in REGION_COLORS:
        aggregated_stats_sb[region] = {
            "mean_of_means": np.nanmean(stats_sb[region]["means"]),
            "error_bar_means": error_bar(stats_sb[region]["means"]),
            "mean_of_stds": np.nanmean(stats_sb[region]["stds"]),
            "error_bar_stds": error_bar(stats_sb[region]["stds"]),
            "mean_of_medians": np.nanmean(stats_sb[region]["medians"]),
            "error_bar_medians": error_bar(stats_sb[region]["medians"]),
            "mean_of_skews": np.nanmean(stats_sb[region]["skews"]),
            "error_bar_skews": error_bar(stats_sb[region]["skews"]),
            "mean_of_kurts": np.nanmean(stats_sb[region]["kurts"]),
            "error_bar_kurts": error_bar(stats_sb[region]["kurts"])
        }
        aggregated_stats_gw[region] = {
            "mean_of_means": np.nanmean(stats_gw[region]["means"]),
            "error_bar_means": error_bar(stats_gw[region]["means"]),
            "mean_of_stds": np.nanmean(stats_gw[region]["stds"]),
            "error_bar_stds": error_bar(stats_gw[region]["stds"]),
            "mean_of_medians": np.nanmean(stats_gw[region]["medians"]),
            "error_bar_medians": error_bar(stats_gw[region]["medians"]),
            "mean_of_skews": np.nanmean(stats_gw[region]["skews"]),
            "error_bar_skews": error_bar(stats_gw[region]["skews"]),
            "mean_of_kurts": np.nanmean(stats_gw[region]["kurts"]),
            "error_bar_kurts": error_bar(stats_gw[region]["kurts"])
        }

    # Visualization: For each region, plot a histogram of the mean intensities
    for region in REGION_COLORS:
        valid_means_sb = [v for v in stats_sb[region]["means"] if not np.isnan(v)]
        valid_means_gw = [v for v in stats_gw[region]["means"] if not np.isnan(v)]
        plt.figure()
        if valid_means_sb:
            plt.hist(valid_means_sb, bins=10, alpha=0.5, label="SB Means")
        if valid_means_gw:
            plt.hist(valid_means_gw, bins=10, alpha=0.5, label="GW Means")
        plt.title(f"Histogram of {region.capitalize()} Mean Intensities")
        plt.xlabel("Mean Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    # Textual Report: Print a summary for each region
    for region in REGION_COLORS:
        print(f"==== {region.capitalize()} Summary ====")
        print("Slaty-backed (SB):")
        print(f"  Mean of Means: {aggregated_stats_sb[region]['mean_of_means']:.2f} ± {aggregated_stats_sb[region]['error_bar_means']:.2f}")
        print(f"  Mean of Stds : {aggregated_stats_sb[region]['mean_of_stds']:.2f} ± {aggregated_stats_sb[region]['error_bar_stds']:.2f}")
        print(f"  Mean of Medians: {aggregated_stats_sb[region]['mean_of_medians']:.2f} ± {aggregated_stats_sb[region]['error_bar_medians']:.2f}")
        print(f"  Mean of Skews: {aggregated_stats_sb[region]['mean_of_skews']:.2f} ± {aggregated_stats_sb[region]['error_bar_skews']:.2f}")
        print(f"  Mean of Kurtosis: {aggregated_stats_sb[region]['mean_of_kurts']:.2f} ± {aggregated_stats_sb[region]['error_bar_kurts']:.2f}")
        print("Glaucous-winged (GW):")
        print(f"  Mean of Means: {aggregated_stats_gw[region]['mean_of_means']:.2f} ± {aggregated_stats_gw[region]['error_bar_means']:.2f}")
        print(f"  Mean of Stds : {aggregated_stats_gw[region]['mean_of_stds']:.2f} ± {aggregated_stats_gw[region]['error_bar_stds']:.2f}")
        print(f"  Mean of Medians: {aggregated_stats_gw[region]['mean_of_medians']:.2f} ± {aggregated_stats_gw[region]['error_bar_medians']:.2f}")
        print(f"  Mean of Skews: {aggregated_stats_gw[region]['mean_of_skews']:.2f} ± {aggregated_stats_gw[region]['error_bar_skews']:.2f}")
        print(f"  Mean of Kurtosis: {aggregated_stats_gw[region]['mean_of_kurts']:.2f} ± {aggregated_stats_gw[region]['error_bar_kurts']:.2f}")
        print("\n")

    print("Next Steps / Discussion:")
    print("- Verify if these intensity-based features can differentiate between SB and GW.")
    print("- Consider adding texture-based features (e.g., GLCM) or other measures.")
    print("- Expand the analysis to more images (e.g., s=100) if feasible.")

if __name__ == "__main__":
    main()
