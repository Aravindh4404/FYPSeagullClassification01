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

import matplotlib.pyplot as plt

# Import configuration
from config import (SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR,
                    GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR, S,
                    REGION_COLORS)

# Import necessary functions and modules

from utils import *

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
