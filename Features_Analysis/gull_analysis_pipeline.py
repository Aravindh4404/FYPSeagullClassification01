"""
Pipeline for Gull Wingtip Analysis

This script demonstrates how to:
1. Load images and segmentation maps.
2. Extract wingtip pixels based on segmentation color.
3. Compute statistics (mean, std, median, histogram, skewness, kurtosis).
4. Aggregate and visualize the results.
5. Produce a brief report.

Dependencies:
- Python 3.x
- numpy
- opencv-python (cv2)
- matplotlib
- scipy (for skew, kurtosis) [optional, or implement your own if needed]
- Possibly scikit-image (for more advanced segmentation/analysis)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

###############################################################################
# CONFIGURATION
###############################################################################

# Paths to your data (adapt to your folder structure).
SLATY_BACKED_IMG_DIR = r"C:\Users\Aravindh P\OneDrive - University of Nottingham Malaysia\FYP\FYPSeagullClassification01\Features_Analysis\Original_Images\Slaty_Backed_Gull"
SLATY_BACKED_SEG_DIR = r"C:\Users\Aravindh P\OneDrive - University of Nottingham Malaysia\FYP\FYPSeagullClassification01\Features_Analysis\Colored_Images\Slaty_Backed_Gull"

GLAUCOUS_WINGED_IMG_DIR = r"C:\Users\Aravindh P\OneDrive - University of Nottingham Malaysia\FYP\FYPSeagullClassification01\Features_Analysis\Original_Images\Glaucous_Winged_Gull"
GLAUCOUS_WINGED_SEG_DIR = r"C:\Users\Aravindh P\OneDrive - University of Nottingham Malaysia\FYP\FYPSeagullClassification01\Features_Analysis\Colored_Images\Glaucous_Winged_Gull"

# Number of images per species you want to process
S = 5

# If wingtips are **green in RGB**, then in OpenCV’s BGR we also use (0, 255, 0)
WINGTIP_COLOR = (0, 255, 0)

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def load_images_and_seg_maps(sb_img_dir, sb_seg_dir, gw_img_dir, gw_seg_dir, s):
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

def extract_wingtip_pixels(image, seg_map, wingtip_color):
    if image is None or seg_map is None:
        return np.array([])

    mask = (
        (seg_map[:, :, 0] == wingtip_color[0]) &
        (seg_map[:, :, 1] == wingtip_color[1]) &
        (seg_map[:, :, 2] == wingtip_color[2])
    )
    pixels = image[mask]

    if pixels.size == 0:
        print("[Info] No wingtip pixels found with color", wingtip_color)
    return pixels

def compute_statistics(pixel_values):
    # If empty, return NaNs
    if pixel_values.size == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'hist': None, 'skew': np.nan, 'kurtosis': np.nan
        }

    # Convert color to grayscale by averaging channels if we have (N,3)
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
    arr = np.array(values)
    if arr.size == 0:
        return np.nan
    if method == 'std':
        return np.std(arr)
    elif method == 'sem':
        return np.std(arr) / np.sqrt(len(arr))
    return np.nan

###############################################################################
# MAIN
###############################################################################

def main():
    iSB, sSB, iGW, sGW = load_images_and_seg_maps(
        SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR,
        GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR,
        S
    )

    means_sb, stds_sb, medians_sb, skews_sb, kurts_sb = [], [], [], [], []
    means_gw, stds_gw, medians_gw, skews_gw, kurts_gw = [], [], [], [], []

    num_images = min(len(iSB), len(iGW))

    for idx in range(num_images):
        # Slaty-backed
        wingtip_sb = extract_wingtip_pixels(iSB[idx], sSB[idx], WINGTIP_COLOR)
        stats_sb   = compute_statistics(wingtip_sb)
        means_sb.append(stats_sb['mean'])
        stds_sb.append(stats_sb['std'])
        medians_sb.append(stats_sb['median'])
        skews_sb.append(stats_sb['skew'])
        kurts_sb.append(stats_sb['kurtosis'])

        # Glaucous-winged
        wingtip_gw = extract_wingtip_pixels(iGW[idx], sGW[idx], WINGTIP_COLOR)
        stats_gw   = compute_statistics(wingtip_gw)
        means_gw.append(stats_gw['mean'])
        stds_gw.append(stats_gw['std'])
        medians_gw.append(stats_gw['median'])
        skews_gw.append(stats_gw['skew'])
        kurts_gw.append(stats_gw['kurtosis'])

    # Aggregate
    mean_means_sb = np.nanmean(means_sb)
    mean_stds_sb  = np.nanmean(stds_sb)
    mean_meds_sb  = np.nanmean(medians_sb)
    mean_skew_sb  = np.nanmean(skews_sb)
    mean_kurt_sb  = np.nanmean(kurts_sb)

    mean_means_gw = np.nanmean(means_gw)
    mean_stds_gw  = np.nanmean(stds_gw)
    mean_meds_gw  = np.nanmean(medians_gw)
    mean_skew_gw  = np.nanmean(skews_gw)
    mean_kurt_gw  = np.nanmean(kurts_gw)

    # Plot hist of means
    valid_sb = [v for v in means_sb if not np.isnan(v)]
    valid_gw = [v for v in means_gw if not np.isnan(v)]

    plt.figure()
    if valid_sb:
        plt.hist(valid_sb, bins=10, alpha=0.5, label="SB Means")
    if valid_gw:
        plt.hist(valid_gw, bins=10, alpha=0.5, label="GW Means")
    plt.title("Histogram of Wingtip Mean Intensities")
    plt.xlabel("Mean Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Print text report
    print("==== Slaty-backed Gull (SB) Summary ====")
    print(f"Mean of Means: {mean_means_sb:.2f} ± {error_bar(means_sb):.2f}")
    print(f"Mean of Stds : {mean_stds_sb:.2f} ± {error_bar(stds_sb):.2f}")
    print(f"Mean of Meds : {mean_meds_sb:.2f}")
    print(f"Mean Skew    : {mean_skew_sb:.2f}")
    print(f"Mean Kurtosis: {mean_kurt_sb:.2f}")

    print("\n==== Glaucous-winged Gull (GW) Summary ====")
    print(f"Mean of Means: {mean_means_gw:.2f} ± {error_bar(means_gw):.2f}")
    print(f"Mean of Stds : {mean_stds_gw:.2f} ± {error_bar(stds_gw):.2f}")
    print(f"Mean of Meds : {mean_meds_gw:.2f}")
    print(f"Mean Skew    : {mean_skew_gw:.2f}")
    print(f"Mean Kurtosis: {mean_kurt_gw:.2f}")

    print("\nNext Steps / Discussion:")
    print("- Verify if these intensity-based features differentiate the species.")
    print("- Possibly add texture-based features (GLCM, etc.).")
    print("- Expand segmentation to other regions (head, body) if needed.")
    print("- Scale up analysis to more images, e.g. s=100.")

if __name__ == "__main__":
    main()
