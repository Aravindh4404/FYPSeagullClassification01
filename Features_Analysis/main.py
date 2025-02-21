import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import (
    SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR,
    GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR, S,
    REGION_COLORS
)
from utils import (
    load_images_and_seg_maps, extract_region_pixels,
    compute_enhanced_statistics, compare_distributions
)


def display_full_analysis(image_sb, mask_sb, image_gw, mask_gw, region, pixels_sb, pixels_gw, stats_sb, stats_gw,
                          dist_comparison):
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Detailed Analysis for Region: {region}", fontsize=16)

    # 1. Original SB Image with Mask Overlay
    plt.subplot(2, 3, 1)
    image_sb_rgb = cv2.cvtColor(image_sb, cv2.COLOR_BGR2RGB)
    plt.imshow(image_sb_rgb)
    plt.title("Slaty-backed Gull (SB) - Original")
    plt.axis('off')
    if mask_sb is not None and np.sum(mask_sb) > 0:
        contours, _ = cv2.findContours(mask_sb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plt.contour(mask_sb, levels=[0.5], colors='red', linewidths=1)

    # 2. SB Segmented Region Mask
    plt.subplot(2, 3, 2)
    if mask_sb is not None:
        plt.imshow(mask_sb, cmap='gray')
        plt.title(f"SB Segmented Mask - {region}")
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, "No Mask Data", ha='center', va='center')
        plt.axis('off')

    # 3. SB Pixel Intensity Histogram
    plt.subplot(2, 3, 3)
    if pixels_sb.size > 0:
        plt.hist(pixels_sb.mean(axis=1), bins=30, alpha=0.7, color='blue')
        plt.title("SB Pixel Intensity Distribution")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
    else:
        plt.text(0.5, 0.5, "No Data", ha='center')

    # 4. Original GW Image with Mask Overlay
    plt.subplot(2, 3, 4)
    image_gw_rgb = cv2.cvtColor(image_gw, cv2.COLOR_BGR2RGB)
    plt.imshow(image_gw_rgb)
    plt.title("Glaucous-winged Gull (GW) - Original")
    plt.axis('off')
    if mask_gw is not None and np.sum(mask_gw) > 0:
        contours, _ = cv2.findContours(mask_gw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plt.contour(mask_gw, levels=[0.5], colors='red', linewidths=1)

    # 5. GW Segmented Region Mask
    plt.subplot(2, 3, 5)
    if mask_gw is not None:
        plt.imshow(mask_gw, cmap='gray')
        plt.title(f"GW Segmented Mask - {region}")
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, "No Mask Data", ha='center', va='center')
        plt.axis('off')

    # 6. Combined Histogram with KS-Test
    plt.subplot(2, 3, 6)
    plt.title(f"Combined Intensity - {region}\nKS p-value: {dist_comparison['ks_pvalue']:.3f}")
    plt.xlabel("Intensity (Normalized)")
    plt.ylabel("Frequency")
    if pixels_sb.size > 0:
        plt.hist(pixels_sb.mean(axis=1), bins=30, alpha=0.5, label="SB", color='blue')
    if pixels_gw.size > 0:
        plt.hist(pixels_gw.mean(axis=1), bins=30, alpha=0.5, label="GW", color='orange')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print Detailed Stats
    print(f"\n=== Detailed Statistics for Region: {region} ===")
    print("--- Slaty-backed Gull (SB) ---")
    for k, v in stats_sb.items():
        print(f"{k.capitalize()}: {v:.2f}" if not np.isnan(v) else f"{k.capitalize()}: No Data")

    print("\n--- Glaucous-winged Gull (GW) ---")
    for k, v in stats_gw.items():
        print(f"{k.capitalize()}: {v:.2f}" if not np.isnan(v) else f"{k.capitalize()}: No Data")

    print("\n--- KS-Test Comparison ---")
    print(f"KS Statistic: {dist_comparison['ks_stat']:.3f}")
    print(f"p-value: {dist_comparison['ks_pvalue']:.3f}")


def main():
    iSB, sSB, iGW, sGW = load_images_and_seg_maps(
        SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR,
        GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR,
        S
    )

    num_images = min(len(iSB), len(iGW))

    for idx in range(num_images):
        for region, color in REGION_COLORS.items():
            pixels_sb, mask_sb = extract_region_pixels(iSB[idx], sSB[idx], color)
            stats_sb = compute_enhanced_statistics(pixels_sb)
            pixels_gw, mask_gw = extract_region_pixels(iGW[idx], sGW[idx], color)
            stats_gw = compute_enhanced_statistics(pixels_gw)
            dist_comparison = compare_distributions(pixels_sb, pixels_gw)
            display_full_analysis(iSB[idx], mask_sb, iGW[idx], mask_gw, region, pixels_sb, pixels_gw, stats_sb,
                                  stats_gw, dist_comparison)


if __name__ == "__main__":
    main()