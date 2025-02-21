from Features_Analysis.config import *
from Features_Analysis.utils import *


def compute_enhanced_statistics(pixel_values):
    """
    Compute various statistical measures (mean, std, median, skew, kurtosis, min, max, q25, q75)
    for a set of pixel intensities.
    """
    if pixel_values.size == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'skew': np.nan, 'kurtosis': np.nan, 'min': np.nan,
            'max': np.nan, 'q25': np.nan, 'q75': np.nan
        }

    # If the pixel_values are in RGB, convert them to grayscale intensities
    if len(pixel_values.shape) == 2 and pixel_values.shape[1] == 3:
        pixel_values = pixel_values.mean(axis=1)

    return {
        'mean':     float(np.mean(pixel_values)),
        'std':      float(np.std(pixel_values)),
        'median':   float(np.median(pixel_values)),
        'skew':     float(skew(pixel_values)),
        'kurtosis': float(kurtosis(pixel_values)),
        'min':      float(np.min(pixel_values)),
        'max':      float(np.max(pixel_values)),
        'q25':      float(np.percentile(pixel_values, 25)),
        'q75':      float(np.percentile(pixel_values, 75))
    }


def compare_distributions(pixels_sb, pixels_gw):
    """
    Use the Kolmogorov-Smirnov (KS) test to compare two distributions (SB vs. GW).
    Returns a dict with 'ks_stat' and 'ks_pvalue'.
    """
    if pixels_sb.size == 0 or pixels_gw.size == 0:
        return {'ks_stat': np.nan, 'ks_pvalue': np.nan}

    # Flatten to intensities if RGB
    sb_vals = pixels_sb.mean(axis=1) if (len(pixels_sb.shape) > 1) else pixels_sb
    gw_vals = pixels_gw.mean(axis=1) if (len(pixels_gw.shape) > 1) else pixels_gw

    ks_stat, ks_pvalue = ks_2samp(sb_vals, gw_vals)
    return {'ks_stat': ks_stat, 'ks_pvalue': ks_pvalue}


###############################################################################
# 3. MAIN VISUALIZATION HELPER
###############################################################################
def display_full_analysis(
    image_sb, mask_sb,
    image_gw, mask_gw,
    region, pixels_sb, pixels_gw,
    stats_sb, stats_gw,
    dist_comparison
):
    """
    Shows a detailed 6-panel layout:
    - Top Left:  SB original with contour
    - Top Middle: SB mask
    - Top Right:  SB histogram
    - Bottom Left: GW original with contour
    - Bottom Middle: GW mask
    - Bottom Right: Combined histogram + KS p-value
    Also prints stats to the console for deeper inspection.
    """
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Detailed Analysis for Region: {region}", fontsize=16)

    # 1. Slaty-backed Gull (SB) original
    plt.subplot(2, 3, 1)
    sb_rgb = cv2.cvtColor(image_sb, cv2.COLOR_BGR2RGB)
    plt.imshow(sb_rgb)
    plt.title("SB Gull - Original")
    plt.axis('off')
    # Overlay contour
    if mask_sb is not None and np.sum(mask_sb) > 0:
        sb_contours, _ = cv2.findContours(mask_sb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plt.contour(mask_sb, levels=[0.5], colors='red', linewidths=1)

    # 2. SB Mask
    plt.subplot(2, 3, 2)
    if mask_sb is not None:
        plt.imshow(mask_sb, cmap='gray')
        plt.title(f"SB Segmented Mask - {region}")
    else:
        plt.text(0.5, 0.5, "No Mask Data", ha='center', va='center')
    plt.axis('off')

    # 3. SB Histogram
    plt.subplot(2, 3, 3)
    if pixels_sb.size > 0:
        sb_intensities = (pixels_sb.mean(axis=1) if len(pixels_sb.shape) == 2 and pixels_sb.shape[1] == 3 else pixels_sb)
        plt.hist(sb_intensities, bins=30, alpha=0.7, color='blue')
        plt.title("SB Pixel Intensity Distribution")
        plt.xlabel("Intensity (0=dark, 255=bright)")
        plt.ylabel("Frequency (# of pixels)")
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No SB Data", ha='center', va='center')

    # 4. Glaucous-winged Gull (GW) original
    plt.subplot(2, 3, 4)
    gw_rgb = cv2.cvtColor(image_gw, cv2.COLOR_BGR2RGB)
    plt.imshow(gw_rgb)
    plt.title("GW Gull - Original")
    plt.axis('off')
    # Overlay contour
    if mask_gw is not None and np.sum(mask_gw) > 0:
        gw_contours, _ = cv2.findContours(mask_gw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plt.contour(mask_gw, levels=[0.5], colors='red', linewidths=1)

    # 5. GW Mask
    plt.subplot(2, 3, 5)
    if mask_gw is not None:
        plt.imshow(mask_gw, cmap='gray')
        plt.title(f"GW Segmented Mask - {region}")
    else:
        plt.text(0.5, 0.5, "No Mask Data", ha='center', va='center')
    plt.axis('off')

    # 6. Combined Histogram + KS Test
    plt.subplot(2, 3, 6)
    plt.title(f"Combined Intensity - {region}\nKS p-value: {dist_comparison['ks_pvalue']:.3f}")
    plt.xlabel("Intensity (0=dark, 255=bright)")
    plt.ylabel("Frequency (# of pixels)")
    if pixels_sb.size > 0:
        plt.hist(sb_intensities, bins=30, alpha=0.5, label="SB", color='blue')
    if pixels_gw.size > 0:
        gw_intensities = (pixels_gw.mean(axis=1) if len(pixels_gw.shape) == 2 and pixels_gw.shape[1] == 3 else pixels_gw)
        plt.hist(gw_intensities, bins=30, alpha=0.5, label="GW", color='orange')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print Stats to Console
    print(f"\n=== Detailed Statistics for Region: {region} ===")
    print("--- Slaty-backed Gull (SB) ---")
    for k, v in stats_sb.items():
        val_str = f"{v:.2f}" if not np.isnan(v) else "No Data"
        print(f"{k.capitalize()}: {val_str}")

    print("\n--- Glaucous-winged Gull (GW) ---")
    for k, v in stats_gw.items():
        val_str = f"{v:.2f}" if not np.isnan(v) else "No Data"
        print(f"{k.capitalize()}: {val_str}")

    print("\n--- KS-Test Comparison ---")
    print(f"KS Statistic: {dist_comparison['ks_stat']:.3f}")
    print(f"p-value: {dist_comparison['ks_pvalue']:.3f}")
    print("If p-value < 0.05, the two distributions are significantly different.\n")


###############################################################################
# 4. MAIN FUNCTION TO RUN THE ANALYSIS
###############################################################################
def main():
    print("Loading images and segmentation maps...")
    iSB, sSB, iGW, sGW = load_images_and_seg_maps(
        SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR,
        GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR,
        S
    )

    num_images = min(len(iSB), len(iGW))
    print(f"Found {num_images} SB-GW image pairs to analyze.\n")

    # Data structures for aggregated statistics
    all_stats_sb = {'mean': [], 'std': []}  # Extend for other stats as needed.
    all_stats_gw = {'mean': [], 'std': []}

    # Loop through each image pair
    for idx in range(num_images):
        print(f"--- Processing Image Pair #{idx + 1} ---")
        for region_name, bgr_color in REGION_COLORS.items():
            # Extract region pixels for SB and GW
            sb_pixels, sb_mask = extract_region_pixels(iSB[idx], sSB[idx], bgr_color)
            sb_stats = compute_enhanced_statistics(sb_pixels)

            gw_pixels, gw_mask = extract_region_pixels(iGW[idx], sGW[idx], bgr_color)
            gw_stats = compute_enhanced_statistics(gw_pixels)

            # Accumulate statistics
            all_stats_sb['mean'].append(sb_stats['mean'])
            all_stats_sb['std'].append(sb_stats['std'])
            all_stats_gw['mean'].append(gw_stats['mean'])
            all_stats_gw['std'].append(gw_stats['std'])

            # Compare distributions and display per-image analysis
            dist_comp = compare_distributions(sb_pixels, gw_pixels)
            display_full_analysis(
                iSB[idx], sb_mask,
                iGW[idx], gw_mask,
                region_name, sb_pixels, gw_pixels,
                sb_stats, gw_stats,
                dist_comp
            )

    # After processing all images, compute global aggregates:
    global_mean_sb = np.nanmean(all_stats_sb['mean'])
    global_std_sb = np.nanmean(all_stats_sb['std'])
    global_mean_gw = np.nanmean(all_stats_gw['mean'])
    global_std_gw = np.nanmean(all_stats_gw['std'])

    # Compute error bars (could be standard error or std of means)
    error_bar_sb = np.nanstd(all_stats_sb['mean'])
    error_bar_gw = np.nanstd(all_stats_gw['mean'])

    print("\n=== Global Aggregated Statistics ===")
    print(
        f"SB: Mean of means = {global_mean_sb:.2f} (Error Bar: {error_bar_sb:.2f}), Mean of std devs = {global_std_sb:.2f}")
    print(
        f"GW: Mean of means = {global_mean_gw:.2f} (Error Bar: {error_bar_gw:.2f}), Mean of std devs = {global_std_gw:.2f}")

    # Plot aggregated histograms for the means
    plt.figure(figsize=(10, 5))
    plt.hist(all_stats_sb['mean'], bins=10, alpha=0.5, label='SB Means', color='blue')
    plt.hist(all_stats_gw['mean'], bins=10, alpha=0.5, label='GW Means', color='orange')
    plt.xlabel("Intensity Mean")
    plt.ylabel("Frequency")
    plt.title("Aggregated Histogram of Intensity Means")
    plt.legend()
    plt.show()

    # Future work: Implement texture-based features and analyze other regions (e.g., head).


if __name__ == "__main__":
    main()
