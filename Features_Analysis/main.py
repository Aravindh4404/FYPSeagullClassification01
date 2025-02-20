"""
Pipeline for Gull Regional Analysis

This script:
1. Loads images and segmentation maps.
2. Extracts region pixels for each of: wingtip, wing, body, and head.
3. Computes intensity statistics (mean, std, median, skewness, kurtosis) for each region.
4. Visualizes results with clear layouts (original images, masks, histograms).
5. Prints a textual summary report.

Dependencies:
- Python 3.x
- numpy
- opencv-python (cv2)
- matplotlib
- scipy (for skew, kurtosis)
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Import configuration
from config import (SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR,
                   GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR, S,
                   REGION_COLORS)

# Import necessary functions and modules
from utils import (load_images_and_seg_maps, extract_region_pixels,
                  compute_enhanced_statistics, compare_distributions)

###############################################################################
# MAIN PIPELINE
###############################################################################

def main():
    # Load images and segmentation maps
    iSB, sSB, iGW, sGW = load_images_and_seg_maps(
        SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR,
        GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR,
        S
    )

    num_images = min(len(iSB), len(iGW))

    for idx in range(num_images):
        for region, color in REGION_COLORS.items():
            # Extract pixels and masks for Slaty-backed (SB)
            pixels_sb, mask_sb = extract_region_pixels(iSB[idx], sSB[idx], color)
            stats_sb = compute_enhanced_statistics(pixels_sb)

            # Extract pixels and masks for Glaucous-winged (GW)
            pixels_gw, mask_gw = extract_region_pixels(iGW[idx], sGW[idx], color)
            stats_gw = compute_enhanced_statistics(pixels_gw)

            # Distribution comparison between SB and GW
            dist_comparison = compare_distributions(pixels_sb, pixels_gw)

            # Enhanced Visualization: Clear 1x3 layout
            plt.figure(figsize=(15, 5))  # Larger figure for clarity
            plt.suptitle(f"Image {idx + 1} - Region: {region}", fontsize=14)

            # 1. Original Slaty-backed (SB) Image
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(iSB[idx], cv2.COLOR_BGR2RGB))
            plt.title("Original SB")
            plt.axis('off')
            # Overlay mask contour for better understanding
            if mask_sb is not None and np.sum(mask_sb) > 0:
                contours, _ = cv2.findContours(mask_sb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                plt.contour(mask_sb, levels=[0.5], colors='red', linewidths=1)

            # 2. Segmentation Mask for SB (highlighting the region)
            plt.subplot(1, 3, 2)
            if mask_sb is not None:
                plt.imshow(mask_sb, cmap='gray')
                plt.title(f"SB Mask - {region}")
                plt.axis('off')
            else:
                plt.text(0.5, 0.5, "No Mask Data", ha='center', va='center')
                plt.axis('off')

            # 3. Original Glaucous-winged (GW) Image
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(iGW[idx], cv2.COLOR_BGR2RGB))
            plt.title("Original GW")
            plt.axis('off')
            # Overlay mask contour for better understanding
            if mask_gw is not None and np.sum(mask_gw) > 0:
                contours, _ = cv2.findContours(mask_gw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                plt.contour(mask_gw, levels=[0.5], colors='red', linewidths=1)

            plt.tight_layout()
            plt.show()

            # Enhanced Histogram with Distribution Comparison
            plt.figure(figsize=(8, 5))
            plt.title(f"Intensity Distribution - {region}\nKS p-value: {dist_comparison['ks_pvalue']:.3f}")
            plt.xlabel("Intensity (Normalized)")
            plt.ylabel("Frequency")

            if pixels_sb.size > 0:
                plt.hist(pixels_sb.mean(axis=1), bins=30, alpha=0.5, label="SB Pixels", color='blue')
            if pixels_gw.size > 0:
                plt.hist(pixels_gw.mean(axis=1), bins=30, alpha=0.5, label="GW Pixels", color='orange')

            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            # Detailed Logging for Understanding
            print(f"\n=== Analysis for Image {idx + 1}, Region: {region} ===")
            print(f"SB Pixels Extracted: {pixels_sb.shape[0] if pixels_sb.size > 0 else 0}")
            print(f"GW Pixels Extracted: {pixels_gw.shape[0] if pixels_gw.size > 0 else 0}")
            print(f"SB Statistics: {stats_sb}")
            print(f"GW Statistics: {stats_gw}")
            print(f"Distribution Comparison (KS Test): {dist_comparison}")

if __name__ == "__main__":
    main()