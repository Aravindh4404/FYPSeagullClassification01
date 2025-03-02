import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem
from Features_Analysis.config import *
from Features_Analysis.image_normalization import (
    to_grayscale,
    minmax_normalize,
    zscore_normalize,
    hist_equalize,
    clahe_normalize,
    compute_global_minmax,
    global_minmax_normalize,
    load_images
)


# Create a function to apply a normalization method to an image
def apply_normalization(image, method, global_min=None, global_max=None):
    """
    Apply a specific normalization method to an image.

    Parameters:
        image (numpy.ndarray): Input image
        method (str): Name of normalization method
        global_min (float, optional): Global minimum for global min-max normalization
        global_max (float, optional): Global maximum for global min-max normalization

    Returns:
        numpy.ndarray: Normalized image
    """
    if method == "None":
        return image.copy()
    elif method == "Grayscale":
        # Convert to grayscale, then back to 3 channels for consistency
        gray = to_grayscale(image)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif method == "MinMax":
        gray = minmax_normalize(image)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif method == "ZScore":
        gray = zscore_normalize(image)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif method == "HistEq":
        gray = hist_equalize(image)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif method == "CLAHE":
        gray = clahe_normalize(image)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif method == "GlobalMinMax":
        if global_min is None or global_max is None:
            raise ValueError("Global min and max must be provided for GlobalMinMax normalization")
        gray = global_minmax_normalize(image, global_min, global_max)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# Function to run analysis with different normalization techniques
def run_normalization_comparison():
    print("\n" + "=" * 50)
    print("NORMALIZATION TECHNIQUES COMPARISON")
    print("=" * 50)

    # Create output directories
    os.makedirs("Outputs/Normalization_Comparison", exist_ok=True)
    os.makedirs("Outputs/Normalization_Comparison/data", exist_ok=True)
    os.makedirs("Outputs/Normalization_Comparison/visualizations", exist_ok=True)

    # Load all images
    print("Loading images...")
    sb_images = load_images(SLATY_BACKED_IMG_DIR)[:S]
    gw_images = load_images(GLAUCOUS_WINGED_IMG_DIR)[:S]

    # Load segmentation maps
    print("Loading segmentation maps...")
    sb_segs = []
    for fname, _ in sb_images:
        seg_path = os.path.join(SLATY_BACKED_SEG_DIR, fname)
        seg = cv2.imread(seg_path)
        if seg is not None:
            sb_segs.append((fname, seg))

    gw_segs = []
    for fname, _ in gw_images:
        seg_path = os.path.join(GLAUCOUS_WINGED_SEG_DIR, fname)
        seg = cv2.imread(seg_path)
        if seg is not None:
            gw_segs.append((fname, seg))

    # Match images with their segmentation maps
    sb_pairs = []
    for img_fname, img in sb_images:
        for seg_fname, seg in sb_segs:
            if img_fname == seg_fname:
                sb_pairs.append((img, seg))
                break

    gw_pairs = []
    for img_fname, img in gw_images:
        for seg_fname, seg in gw_segs:
            if img_fname == seg_fname:
                gw_pairs.append((img, seg))
                break

    # Compute global min and max across all images for global min-max normalization
    all_images = [img for _, img in sb_images + gw_images]
    global_min, global_max = compute_global_minmax([(None, img) for img in all_images])
    print(f"Global min: {global_min}, Global max: {global_max}")

    # Define normalization methods to compare
    normalization_methods = [
        "None",  # Original images, no normalization
        "Grayscale",  # Simple grayscale conversion
        "MinMax",  # Per-image min-max normalization
        "ZScore",  # Z-score normalization
        "HistEq",  # Histogram equalization
        "CLAHE",  # Contrast Limited Adaptive Histogram Equalization
        "GlobalMinMax"  # Global min-max normalization
    ]

    # Collect statistics for each normalization method
    stats_by_method = {method: {"SB": [], "GW": []} for method in normalization_methods}

    # Process each normalization method
    for method in normalization_methods:
        print(f"\nProcessing normalization method: {method}")

        # A new folder for each normalization method's results
        method_dir = f"Outputs/Normalization_Comparison/{method}"
        os.makedirs(method_dir, exist_ok=True)

        # Create a subplot for visual comparison of all images with this normalization
        fig, axes = plt.subplots(2, max(len(sb_pairs), len(gw_pairs)), figsize=(15, 6))
        plt.suptitle(f"Normalization Method: {method}", fontsize=16)

        # Process Slaty-backed Gull images
        for i, (img, seg) in enumerate(sb_pairs):
            # Apply normalization
            if method == "GlobalMinMax":
                norm_img = apply_normalization(img, method, global_min, global_max)
            else:
                norm_img = apply_normalization(img, method)

            # Extract the wingtip region using the segmentation mask
            mask = cv2.inRange(seg,
                               np.array([max(c - 10, 0) for c in REGION_COLORS["wingtip"]], dtype=np.uint8),
                               np.array([min(c + 10, 255) for c in REGION_COLORS["wingtip"]], dtype=np.uint8))

            # Calculate statistics for the wingtip region
            wingtip_pixels = norm_img[mask > 0]
            if len(wingtip_pixels) > 0:
                if len(wingtip_pixels.shape) > 1:
                    # For color images, average the channels
                    wingtip_intensities = wingtip_pixels.mean(axis=1)
                else:
                    wingtip_intensities = wingtip_pixels

                mean_val = np.mean(wingtip_intensities)
                std_val = np.std(wingtip_intensities)
                stats_by_method[method]["SB"].append({
                    "mean": mean_val,
                    "std": std_val,
                    "min": np.min(wingtip_intensities),
                    "max": np.max(wingtip_intensities)
                })

            # Display the normalized image in the subplot
            if i < axes[0].shape[0]:
                axes[0, i].imshow(cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB))
                axes[0, i].set_title(f"SB {i + 1}")
                axes[0, i].axis('off')

        # Process Glaucous-winged Gull images
        for i, (img, seg) in enumerate(gw_pairs):
            # Apply normalization
            if method == "GlobalMinMax":
                norm_img = apply_normalization(img, method, global_min, global_max)
            else:
                norm_img = apply_normalization(img, method)

            # Extract the wingtip region using the segmentation mask
            mask = cv2.inRange(seg,
                               np.array([max(c - 10, 0) for c in REGION_COLORS["wingtip"]], dtype=np.uint8),
                               np.array([min(c + 10, 255) for c in REGION_COLORS["wingtip"]], dtype=np.uint8))

            # Calculate statistics for the wingtip region
            wingtip_pixels = norm_img[mask > 0]
            if len(wingtip_pixels) > 0:
                if len(wingtip_pixels.shape) > 1:
                    # For color images, average the channels
                    wingtip_intensities = wingtip_pixels.mean(axis=1)
                else:
                    wingtip_intensities = wingtip_pixels

                mean_val = np.mean(wingtip_intensities)
                std_val = np.std(wingtip_intensities)
                stats_by_method[method]["GW"].append({
                    "mean": mean_val,
                    "std": std_val,
                    "min": np.min(wingtip_intensities),
                    "max": np.max(wingtip_intensities)
                })

            # Display the normalized image in the subplot
            if i < axes[1].shape[0]:
                axes[1, i].imshow(cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB))
                axes[1, i].set_title(f"GW {i + 1}")
                axes[1, i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{method_dir}/all_images_comparison.png", dpi=150)
        plt.close()

    # Create summary statistics and comparative visualization
    method_summary = []

    for method in normalization_methods:
        # Calculate mean statistics across all images for each species
        sb_means = [stats["mean"] for stats in stats_by_method[method]["SB"]]
        gw_means = [stats["mean"] for stats in stats_by_method[method]["GW"]]

        sb_mean = np.mean(sb_means) if sb_means else 0
        sb_std = np.std(sb_means) if sb_means else 0
        gw_mean = np.mean(gw_means) if gw_means else 0
        gw_std = np.std(gw_means) if gw_means else 0

        # Calculate mean of standard deviations (average local variation)
        sb_local_std = np.mean([stats["std"] for stats in stats_by_method[method]["SB"]]) if stats_by_method[method][
            "SB"] else 0
        gw_local_std = np.mean([stats["std"] for stats in stats_by_method[method]["GW"]]) if stats_by_method[method][
            "GW"] else 0

        # Calculate range of intensities
        sb_min = min([stats["min"] for stats in stats_by_method[method]["SB"]]) if stats_by_method[method]["SB"] else 0
        sb_max = max([stats["max"] for stats in stats_by_method[method]["SB"]]) if stats_by_method[method]["SB"] else 0
        gw_min = min([stats["min"] for stats in stats_by_method[method]["GW"]]) if stats_by_method[method]["GW"] else 0
        gw_max = max([stats["max"] for stats in stats_by_method[method]["GW"]]) if stats_by_method[method]["GW"] else 0

        # Calculate separation (difference between means)
        separation = abs(sb_mean - gw_mean)

        # Calculate coefficient of variation (CV) for each species
        sb_cv = sb_std / sb_mean if sb_mean != 0 else 0
        gw_cv = gw_std / gw_mean if gw_mean != 0 else 0

        # Add to summary
        method_summary.append({
            "Method": method,
            "SB_Mean": sb_mean,
            "SB_StdAcrossImages": sb_std,
            "SB_CV": sb_cv,
            "SB_LocalStd": sb_local_std,
            "SB_Range": sb_max - sb_min,
            "GW_Mean": gw_mean,
            "GW_StdAcrossImages": gw_std,
            "GW_CV": gw_cv,
            "GW_LocalStd": gw_local_std,
            "GW_Range": gw_max - gw_min,
            "Separation": separation,
            "CombinedCV": (sb_cv + gw_cv) / 2,  # Average CV across species
        })

    # Convert to DataFrame and save
    summary_df = pd.DataFrame(method_summary)
    summary_df.to_csv("Outputs/Normalization_Comparison/normalization_comparison_stats.csv", index=False)

    # Create a bar plot comparing mean intensities
    plt.figure(figsize=(12, 8))
    x = np.arange(len(normalization_methods))
    width = 0.35

    plt.bar(x - width / 2, summary_df["SB_Mean"], width, yerr=summary_df["SB_StdAcrossImages"],
            label="Slaty-backed", capsize=5, color="blue", alpha=0.7)
    plt.bar(x + width / 2, summary_df["GW_Mean"], width, yerr=summary_df["GW_StdAcrossImages"],
            label="Glaucous-winged", capsize=5, color="green", alpha=0.7)

    plt.xlabel("Normalization Method")
    plt.ylabel("Mean Intensity")
    plt.title("Comparison of Normalization Methods: Mean Intensity in Wingtip Region")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("Outputs/Normalization_Comparison/mean_intensity_comparison.png", dpi=150)
    plt.close()

    # Create a bar plot comparing CV (coefficient of variation)
    plt.figure(figsize=(12, 8))

    plt.bar(x - width / 2, summary_df["SB_CV"], width, label="Slaty-backed CV",
            color="blue", alpha=0.7)
    plt.bar(x + width / 2, summary_df["GW_CV"], width, label="Glaucous-winged CV",
            color="green", alpha=0.7)

    plt.xlabel("Normalization Method")
    plt.ylabel("Coefficient of Variation")
    plt.title("Comparison of Normalization Methods: Coefficient of Variation in Wingtip Region")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("Outputs/Normalization_Comparison/cv_comparison.png", dpi=150)
    plt.close()

    # Create a bar plot comparing separation between species
    plt.figure(figsize=(12, 8))

    plt.bar(x, summary_df["Separation"], width, color="purple", alpha=0.7)

    plt.xlabel("Normalization Method")
    plt.ylabel("Separation (|SB_Mean - GW_Mean|)")
    plt.title("Comparison of Normalization Methods: Separation Between Species")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig("Outputs/Normalization_Comparison/separation_comparison.png", dpi=150)
    plt.close()

    # Create a comprehensive multi-metric comparison figure
    plt.figure(figsize=(14, 10))

    # Subplot 1: Mean Intensities
    plt.subplot(2, 2, 1)
    plt.bar(x - width / 2, summary_df["SB_Mean"], width, yerr=summary_df["SB_StdAcrossImages"],
            label="Slaty-backed", capsize=5, color="blue", alpha=0.7)
    plt.bar(x + width / 2, summary_df["GW_Mean"], width, yerr=summary_df["GW_StdAcrossImages"],
            label="Glaucous-winged", capsize=5, color="green", alpha=0.7)
    plt.xlabel("Normalization Method")
    plt.ylabel("Mean Intensity")
    plt.title("Mean Intensity in Wingtip Region")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)

    # Subplot 2: Local Standard Deviation
    plt.subplot(2, 2, 2)
    plt.bar(x - width / 2, summary_df["SB_LocalStd"], width, label="Slaty-backed",
            color="blue", alpha=0.7)
    plt.bar(x + width / 2, summary_df["GW_LocalStd"], width, label="Glaucous-winged",
            color="green", alpha=0.7)
    plt.xlabel("Normalization Method")
    plt.ylabel("Average Local Std Dev")
    plt.title("Local Variation (Avg Std Dev) in Wingtip Region")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.grid(alpha=0.3)

    # Subplot 3: Coefficient of Variation
    plt.subplot(2, 2, 3)
    plt.bar(x - width / 2, summary_df["SB_CV"], width, label="Slaty-backed",
            color="blue", alpha=0.7)
    plt.bar(x + width / 2, summary_df["GW_CV"], width, label="Glaucous-winged",
            color="green", alpha=0.7)
    plt.xlabel("Normalization Method")
    plt.ylabel("Coefficient of Variation")
    plt.title("Coefficient of Variation in Wingtip Region")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.grid(alpha=0.3)

    # Subplot 4: Separation & Combined CV
    plt.subplot(2, 2, 4)
    plt.bar(x - width / 2, summary_df["Separation"], width, label="Separation",
            color="purple", alpha=0.7)
    plt.bar(x + width / 2, summary_df["CombinedCV"], width, label="Combined CV",
            color="orange", alpha=0.7)
    plt.xlabel("Normalization Method")
    plt.ylabel("Metric Value")
    plt.title("Separation Between Species & Combined CV")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Outputs/Normalization_Comparison/comprehensive_comparison.png", dpi=150)
    plt.close()

    # Print summary table
    print("\n" + "=" * 100)
    print("NORMALIZATION COMPARISON SUMMARY")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 100)

    # Determine the "best" normalization method based on different criteria
    best_separation = summary_df.loc[summary_df["Separation"].idxmax()]
    best_combined_cv = summary_df.loc[summary_df["CombinedCV"].idxmin()]

    print(
        f"\nBest method for SEPARATION: {best_separation['Method']} (Separation = {best_separation['Separation']:.2f})")
    print(
        f"Best method for CONSISTENCY (lowest CV): {best_combined_cv['Method']} (Combined CV = {best_combined_cv['CombinedCV']:.2f})")

    print("\nNormalization comparison complete! Results saved to 'Outputs/Normalization_Comparison/'")


# Call this function from your main script to run the comparison
if __name__ == "__main__":
    run_normalization_comparison()