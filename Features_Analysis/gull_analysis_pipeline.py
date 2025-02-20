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
    "head":    (255, 255, 0),    # Yellow in RGB → (0, 255, 255) in BGR
    "body":    (0, 255, 255)   # Sky Blue (e.g., RGB (135,206,235)) → (235,206,135) in BGR
}

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def load_images_and_seg_maps(sb_img_dir, sb_seg_dir, gw_img_dir, gw_seg_dir, s):
    sb_images = sorted(os.listdir(sb_img_dir))[:s]
    sb_segs = sorted(os.listdir(sb_seg_dir))[:s]
    gw_images = sorted(os.listdir(gw_img_dir))[:s]
    gw_segs = sorted(os.listdir(gw_seg_dir))[:s]

    iSB, sSB, iGW, sGW = [], [], [], []

    for img_name, seg_name in zip(sb_images, sb_segs):
        img_path, seg_path = os.path.join(sb_img_dir, img_name), os.path.join(sb_seg_dir, seg_name)
        img, seg = cv2.imread(img_path), cv2.imread(seg_path)

        if seg is not None and seg.shape[-1] == 4:
            seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)

        if img is not None and seg is not None:
            iSB.append(img)
            sSB.append(seg)
        else:
            print(f"[Warning] Could not load: {img_name}, {seg_name}")

    for img_name, seg_name in zip(gw_images, gw_segs):
        img_path, seg_path = os.path.join(gw_img_dir, img_name), os.path.join(gw_seg_dir, seg_name)
        img, seg = cv2.imread(img_path), cv2.imread(seg_path)

        if seg is not None and seg.shape[-1] == 4:
            seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)

        if img is not None and seg is not None:
            iGW.append(img)
            sGW.append(seg)
        else:
            print(f"[Warning] Could not load: {img_name}, {seg_name}")

    return iSB, sSB, iGW, sGW


def extract_region_pixels(image, seg_map, region_color, tolerance=10):
    if image is None or seg_map is None:
        return np.array([])

    lower = np.array([max(c - tolerance, 0) for c in region_color], dtype=np.uint8)
    upper = np.array([min(c + tolerance, 255) for c in region_color], dtype=np.uint8)
    mask = cv2.inRange(seg_map, lower, upper)

    selected_pixels = image[mask > 0]

    if selected_pixels.size == 0:
        print(f"[Info] No pixels found for region color {region_color} (Tolerance: {tolerance})")

    return selected_pixels, mask


def compute_statistics(pixel_values):
    if pixel_values.size == 0:
        return {'mean': np.nan, 'std': np.nan, 'median': np.nan, 'skew': np.nan, 'kurtosis': np.nan}

    if len(pixel_values.shape) == 2 and pixel_values.shape[1] == 3:
        pixel_values = pixel_values.mean(axis=1)

    return {
        'mean': np.mean(pixel_values),
        'std': np.std(pixel_values),
        'median': np.median(pixel_values),
        'skew': skew(pixel_values),
        'kurtosis': kurtosis(pixel_values)
    }


###############################################################################
# MAIN PIPELINE
###############################################################################

def main():
    iSB, sSB, iGW, sGW = load_images_and_seg_maps(
        SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR,
        GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR,
        S
    )

    num_images = min(len(iSB), len(iGW))

    for idx in range(num_images):
        for region, color in REGION_COLORS.items():
            # Process Slaty-backed images
            pixels_sb, mask_sb = extract_region_pixels(iSB[idx], sSB[idx], color)
            stats_sb = compute_statistics(pixels_sb)

            # Process Glaucous-winged images
            pixels_gw, mask_gw = extract_region_pixels(iGW[idx], sGW[idx], color)
            stats_gw = compute_statistics(pixels_gw)

            # Display Selected Regions
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(cv2.cvtColor(iSB[idx], cv2.COLOR_BGR2RGB))
            axs[0].set_title(f"Original SB - {region}")
            axs[0].axis('off')

            axs[1].imshow(mask_sb, cmap='gray')
            axs[1].set_title(f"Mask SB - {region}")
            axs[1].axis('off')

            axs[2].imshow(cv2.cvtColor(iGW[idx], cv2.COLOR_BGR2RGB))
            axs[2].set_title(f"Original GW - {region}")
            axs[2].axis('off')

            plt.show()

            # Histogram Debugging
            plt.figure(figsize=(8, 5))
            if pixels_sb.size > 0:
                plt.hist(pixels_sb.mean(axis=1), bins=30, alpha=0.5, label="SB Pixels")
            if pixels_gw.size > 0:
                plt.hist(pixels_gw.mean(axis=1), bins=30, alpha=0.5, label="GW Pixels")
            plt.title(f"Histogram of {region} Intensity")
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

            # Debugging Log
            print(f"\n[DEBUG] Region: {region}")
            print(f"SB - Pixels Extracted: {pixels_sb.shape[0]}")
            print(f"GW - Pixels Extracted: {pixels_gw.shape[0]}")
            print(f"SB - Intensity Stats: {stats_sb}")
            print(f"GW - Intensity Stats: {stats_gw}")


if __name__ == "__main__":
    main()
