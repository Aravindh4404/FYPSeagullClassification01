# import os
# import cv2
# import numpy as np
# from scipy.stats import skew, kurtosis
#
#
# ###############################################################################
# # HELPER FUNCTIONS
# ###############################################################################
#
# def load_images_and_seg_maps(sb_img_dir, sb_seg_dir, gw_img_dir, gw_seg_dir, s):
#     sb_images = sorted(os.listdir(sb_img_dir))[:s]
#     sb_segs = sorted(os.listdir(sb_seg_dir))[:s]
#     gw_images = sorted(os.listdir(gw_img_dir))[:s]
#     gw_segs = sorted(os.listdir(gw_seg_dir))[:s]
#
#     iSB, sSB, iGW, sGW = [], [], [], []
#
#     for img_name, seg_name in zip(sb_images, sb_segs):
#         img_path, seg_path = os.path.join(sb_img_dir, img_name), os.path.join(sb_seg_dir, seg_name)
#         img, seg = cv2.imread(img_path), cv2.imread(seg_path)
#
#         if seg is not None and seg.shape[-1] == 4:
#             seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)
#
#         if img is not None and seg is not None:
#             iSB.append(img)
#             sSB.append(seg)
#         else:
#             print(f"[Warning] Could not load: {img_name}, {seg_name}")
#
#     for img_name, seg_name in zip(gw_images, gw_segs):
#         img_path, seg_path = os.path.join(gw_img_dir, img_name), os.path.join(gw_seg_dir, seg_name)
#         img, seg = cv2.imread(img_path), cv2.imread(seg_path)
#
#         if seg is not None and seg.shape[-1] == 4:
#             seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)
#
#         if img is not None and seg is not None:
#             iGW.append(img)
#             sGW.append(seg)
#         else:
#             print(f"[Warning] Could not load: {img_name}, {seg_name}")
#
#     return iSB, sSB, iGW, sGW
#
#
# def extract_region_pixels(image, seg_map, region_color, tolerance=10):
#     if image is None or seg_map is None:
#         return np.array([])
#
#     lower = np.array([max(c - tolerance, 0) for c in region_color], dtype=np.uint8)
#     upper = np.array([min(c + tolerance, 255) for c in region_color], dtype=np.uint8)
#     mask = cv2.inRange(seg_map, lower, upper)
#
#     selected_pixels = image[mask > 0]
#
#     if selected_pixels.size == 0:
#         print(f"[Info] No pixels found for region color {region_color} (Tolerance: {tolerance})")
#
#     return selected_pixels, mask
#
#
# def compute_statistics(pixel_values):
#     if pixel_values.size == 0:
#         return {'mean': np.nan, 'std': np.nan, 'median': np.nan, 'skew': np.nan, 'kurtosis': np.nan}
#
#     if len(pixel_values.shape) == 2 and pixel_values.shape[1] == 3:
#         pixel_values = pixel_values.mean(axis=1)
#
#     return {
#         'mean': np.mean(pixel_values),
#         'std': np.std(pixel_values),
#         'median': np.median(pixel_values),
#         'skew': skew(pixel_values),
#         'kurtosis': kurtosis(pixel_values)
#     }


import os
import cv2
import numpy as np
from scipy.stats import skew, kurtosis, ks_2samp
import matplotlib.pyplot as plt
from scipy import stats


###############################################################################
# ENHANCED HELPER FUNCTIONS
###############################################################################

def load_images_and_seg_maps(sb_img_dir, sb_seg_dir, gw_img_dir, gw_seg_dir, s):
    """Load images and segmentation maps from directories."""
    sb_images = sorted(os.listdir(sb_img_dir))[:s]
    sb_segs = sorted(os.listdir(sb_seg_dir))[:s]
    gw_images = sorted(os.listdir(gw_img_dir))[:s]
    gw_segs = sorted(os.listdir(gw_seg_dir))[:s]

    iSB, sSB, iGW, sGW = [], [], [], []

    for img_name, seg_name in zip(sb_images, sb_segs):
        img_path = os.path.join(sb_img_dir, img_name)
        seg_path = os.path.join(sb_seg_dir, seg_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)

        if seg is not None and seg.shape[-1] == 4:
            seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)

        if img is not None and seg is not None:
            iSB.append(img)
            sSB.append(seg)

    for img_name, seg_name in zip(gw_images, gw_segs):
        img_path = os.path.join(gw_img_dir, img_name)
        seg_path = os.path.join(gw_seg_dir, seg_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)

        if seg is not None and seg.shape[-1] == 4:
            seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)

        if img is not None and seg is not None:
            iGW.append(img)
            sGW.append(seg)

    return iSB, sSB, iGW, sGW


def extract_region_pixels(image, seg_map, region_color, tolerance=10):
    """Extract pixels from image based on segmentation map region color."""
    if image is None or seg_map is None:
        return np.array([]), None

    lower = np.array([max(c - tolerance, 0) for c in region_color], dtype=np.uint8)
    upper = np.array([min(c + tolerance, 255) for c in region_color], dtype=np.uint8)
    mask = cv2.inRange(seg_map, lower, upper)
    selected_pixels = image[mask > 0]

    return selected_pixels, mask


def compute_enhanced_statistics(pixel_values):
    """Compute enhanced statistical measures for pixel values."""
    if pixel_values.size == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'median': np.nan,
            'skew': np.nan, 'kurtosis': np.nan, 'min': np.nan,
            'max': np.nan, 'q25': np.nan, 'q75': np.nan
        }

    if len(pixel_values.shape) == 2 and pixel_values.shape[1] == 3:
        pixel_values = pixel_values.mean(axis=1)

    return {
        'mean': np.mean(pixel_values),
        'std': np.std(pixel_values),
        'median': np.median(pixel_values),
        'skew': skew(pixel_values),
        'kurtosis': kurtosis(pixel_values),
        'min': np.min(pixel_values),
        'max': np.max(pixel_values),
        'q25': np.percentile(pixel_values, 25),
        'q75': np.percentile(pixel_values, 75)
    }


def compare_distributions(pixels_sb, pixels_gw):
    """Compare distributions between SB and GW using statistical tests."""
    if pixels_sb.size == 0 or pixels_gw.size == 0:
        return {'ks_stat': np.nan, 'ks_pvalue': np.nan}

    sb_vals = pixels_sb.mean(axis=1) if len(pixels_sb.shape) > 1 else pixels_sb
    gw_vals = pixels_gw.mean(axis=1) if len(pixels_gw.shape) > 1 else pixels_gw

    ks_stat, ks_pvalue = ks_2samp(sb_vals, gw_vals)
    return {'ks_stat': ks_stat, 'ks_pvalue': ks_pvalue}


###############################################################################
# ENHANCED ANALYSIS PIPELINE
###############################################################################

def analyze_regions(iSB, sSB, iGW, sGW, region_colors):
    """Enhanced analysis pipeline with improved comparisons."""
    num_images = min(len(iSB), len(iGW))
    results = {region: {'SB': [], 'GW': []} for region in region_colors.keys()}

    for idx in range(num_images):
        plt.figure(figsize=(15, 10))

        for region_idx, (region, color) in enumerate(region_colors.items()):
            # Extract pixels and masks
            pixels_sb, mask_sb = extract_region_pixels(iSB[idx], sSB[idx], color)
            pixels_gw, mask_gw = extract_region_pixels(iGW[idx], sGW[idx], color)

            # Compute statistics
            stats_sb = compute_enhanced_statistics(pixels_sb)
            stats_gw = compute_enhanced_statistics(pixels_gw)
            dist_comparison = compare_distributions(pixels_sb, pixels_gw)

            # Store results
            results[region]['SB'].append(stats_sb)
            results[region]['GW'].append(stats_gw)

            # Visualization
            plt.subplot(len(region_colors), 4, region_idx * 4 + 1)
            plt.imshow(cv2.cvtColor(iSB[idx], cv2.COLOR_BGR2RGB))
            plt.title(f"SB {region}")
            plt.axis('off')

            plt.subplot(len(region_colors), 4, region_idx * 4 + 2)
            plt.imshow(mask_sb, cmap='gray')
            plt.title(f"SB Mask")
            plt.axis('off')

            plt.subplot(len(region_colors), 4, region_idx * 4 + 3)
            plt.imshow(cv2.cvtColor(iGW[idx], cv2.COLOR_BGR2RGB))
            plt.title(f"GW {region}")
            plt.axis('off')

            # Enhanced histogram with statistical comparison
            plt.subplot(len(region_colors), 4, region_idx * 4 + 4)
            if pixels_sb.size > 0:
                plt.hist(pixels_sb.mean(axis=1), bins=30, alpha=0.5, label='SB')
            if pixels_gw.size > 0:
                plt.hist(pixels_gw.mean(axis=1), bins=30, alpha=0.5, label='GW')
            plt.title(f"{region} Histogram\nKS p={dist_comparison['ks_pvalue']:.3f}")
            plt.legend()

        plt.tight_layout()
        plt.show()

        # Print detailed comparison
        print(f"\nImage {idx + 1} Analysis:")
        for region in region_colors.keys():
            print(f"\n{region}:")
            print(f"SB Stats: {results[region]['SB'][-1]}")
            print(f"GW Stats: {results[region]['GW'][-1]}")
            print(f"Distribution Comparison: {compare_distributions(pixels_sb, pixels_gw)}")

    return results
