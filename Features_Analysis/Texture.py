from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import chisquare

from Features_Analysis.config import *
from Features_Analysis.utils import *


###############################################################################
# 2. TEXTURE FEATURE EXTRACTION (GLCM & LBP)
###############################################################################
def compute_glcm_features(gray_region, distances=[1], angles=[0], levels=256):
    """
    Compute GLCM-based texture features (contrast, correlation, energy, homogeneity).
    """
    glcm = graycomatrix(
        gray_region,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    features = {
        'contrast':    float(graycoprops(glcm, 'contrast').mean()),
        'correlation': float(graycoprops(glcm, 'correlation').mean()),
        'energy':      float(graycoprops(glcm, 'energy').mean()),
        'homogeneity': float(graycoprops(glcm, 'homogeneity').mean())
    }
    return features


def compute_lbp_histogram(gray_region, radius=1, n_points=8, method='uniform'):
    """
    Compute a normalized LBP (Local Binary Pattern) histogram for the grayscale region.
    """
    lbp = local_binary_pattern(gray_region, n_points, radius, method)
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def compute_texture_features(image, mask):
    """
    Given a color image and a binary mask, compute GLCM and LBP texture features for that region.
    Returns a dictionary with 'glcm' and 'lbp' keys.
    """
    if mask is None or np.sum(mask) == 0:
        return {'glcm': None, 'lbp': None}

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Crop the region for GLCM / LBP
    y_indices, x_indices = np.where(mask > 0)
    min_y, max_y = y_indices.min(), y_indices.max()
    min_x, max_x = x_indices.min(), x_indices.max()
    region_2d = gray_img[min_y:max_y+1, min_x:max_x+1]

    # Compute features
    glcm_feats = compute_glcm_features(region_2d)
    lbp_hist   = compute_lbp_histogram(region_2d)

    return {
        'glcm': glcm_feats,
        'lbp': lbp_hist
    }


###############################################################################
# 3. OPTIONAL DISTRIBUTION COMPARISON (LBP)
###############################################################################
def compare_lbp_distributions(lbp_sb, lbp_gw):
    """
    Example: use a chi-square test to compare two LBP histograms.
    Returns a dict with chi-square statistic and p-value.
    """
    if lbp_sb is None or lbp_gw is None:
        return {'chi2': np.nan, 'p_value': np.nan}

    # hist * total_pixels = counts, but we can just pass the relative frequencies
    chi2_stat, p_val = chisquare(lbp_sb, f_exp=lbp_gw)
    return {'chi2': chi2_stat, 'p_value': p_val}


###############################################################################
# 4. VISUALIZATION
###############################################################################
def display_texture_analysis(
    image_sb, mask_sb,
    image_gw, mask_gw,
    region_name, lbp_sb, lbp_gw,
    glcm_sb, glcm_gw,
    lbp_comparison
):
    """
    Shows a 6-panel layout, similar to the intensity analysis:
    1. SB original
    2. SB mask
    3. SB LBP histogram
    4. GW original
    5. GW mask
    6. Combined LBP histogram + chi-square p-value

    Also prints GLCM stats to the console.
    """
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Detailed Texture Analysis for Region: {region_name}", fontsize=16)

    # 1. Slaty-backed Gull (SB) original
    plt.subplot(2, 3, 1)
    sb_rgb = cv2.cvtColor(image_sb, cv2.COLOR_BGR2RGB)
    plt.imshow(sb_rgb)
    plt.title("SB Gull - Original")
    plt.axis('off')
    # Overlay contour
    if mask_sb is not None and np.sum(mask_sb) > 0:
        plt.contour(mask_sb, levels=[0.5], colors='red', linewidths=1)

    # 2. SB Mask
    plt.subplot(2, 3, 2)
    if mask_sb is not None:
        plt.imshow(mask_sb, cmap='gray')
        plt.title(f"SB Segmented Mask - {region_name}")
    else:
        plt.text(0.5, 0.5, "No Mask Data", ha='center', va='center')
    plt.axis('off')

    # 3. SB LBP Histogram
    plt.subplot(2, 3, 3)
    if lbp_sb is not None:
        plt.bar(range(len(lbp_sb)), lbp_sb, alpha=0.7, color='blue')
        plt.title("SB LBP Histogram")
        plt.xlabel("LBP Bins")
        plt.ylabel("Frequency")
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
        plt.contour(mask_gw, levels=[0.5], colors='red', linewidths=1)

    # 5. GW Mask
    plt.subplot(2, 3, 5)
    if mask_gw is not None:
        plt.imshow(mask_gw, cmap='gray')
        plt.title(f"GW Segmented Mask - {region_name}")
    else:
        plt.text(0.5, 0.5, "No Mask Data", ha='center', va='center')
    plt.axis('off')

    # 6. Combined LBP Histogram + chi-square p-value
    plt.subplot(2, 3, 6)
    plt.title(f"Combined LBP - {region_name}\nChi-square p-value: {lbp_comparison['p_value']:.3f}")
    plt.xlabel("LBP Bins")
    plt.ylabel("Frequency")
    if lbp_sb is not None:
        plt.bar(range(len(lbp_sb)), lbp_sb, alpha=0.5, label="SB", color='blue')
    if lbp_gw is not None:
        plt.bar(range(len(lbp_gw)), lbp_gw, alpha=0.5, label="GW", color='orange')
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Print GLCM Stats to Console
    print(f"\n=== GLCM Statistics for Region: {region_name} ===")
    print("--- Slaty-backed Gull (SB) ---")
    if glcm_sb is not None:
        for k, v in glcm_sb.items():
            print(f"{k.capitalize()}: {v:.2f}")
    else:
        print("No GLCM data for SB.")

    print("\n--- Glaucous-winged Gull (GW) ---")
    if glcm_gw is not None:
        for k, v in glcm_gw.items():
            print(f"{k.capitalize()}: {v:.2f}")
    else:
        print("No GLCM data for GW.")

    print("\n--- LBP Chi-square Comparison ---")
    print(f"Chi2 Stat: {lbp_comparison['chi2']:.3f}")
    print(f"p-value: {lbp_comparison['p_value']:.3f}")
    print("If p-value < 0.05, the LBP distributions are significantly different.\n")


###############################################################################
# 5. MAIN FUNCTION
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

    # Loop through each image pair
    for idx in range(num_images):
        print(f"--- Processing Image Pair #{idx + 1} ---")

        for region_name, bgr_color in REGION_COLORS.items():

            # 1. Extract region mask & pixels for Slaty-backed (SB)
            sb_pixels, sb_mask = extract_region_pixels(iSB[idx], sSB[idx], bgr_color)
            sb_texture = compute_texture_features(iSB[idx], sb_mask)
            glcm_sb = sb_texture['glcm']
            lbp_sb  = sb_texture['lbp']

            # 2. Extract region mask & pixels for Glaucous-winged (GW)
            gw_pixels, gw_mask = extract_region_pixels(iGW[idx], sGW[idx], bgr_color)
            gw_texture = compute_texture_features(iGW[idx], gw_mask)
            glcm_gw = gw_texture['glcm']
            lbp_gw  = gw_texture['lbp']

            # 3. Compare LBP distributions with chi-square (optional)
            lbp_comp = compare_lbp_distributions(lbp_sb, lbp_gw)

            # 4. Visualize (similar to intensity analysis, but for texture)
            display_texture_analysis(
                iSB[idx], sb_mask,
                iGW[idx], gw_mask,
                region_name, lbp_sb, lbp_gw,
                glcm_sb, glcm_gw,
                lbp_comp
            )

if __name__ == "__main__":
    main()
