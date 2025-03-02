import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem

# If these imports refer to your own config and image_normalization, adjust as needed
from Features_Analysis.config import (
    SLATY_BACKED_IMG_DIR,
    GLAUCOUS_WINGED_IMG_DIR,
    S
)
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


def apply_normalization(image, method, global_min=None, global_max=None):
    """
    Apply a specific normalization method to an image.

    Parameters:
        image (numpy.ndarray): Input image
        method (str): Name of normalization method
        global_min (float, optional): Global minimum for global min-max normalization
        global_max (float, optional): Global maximum for global min-max normalization

    Returns:
        numpy.ndarray: Normalized image (3-channel BGR)
    """
    if method == "None":
        return image.copy()
    elif method == "Grayscale":
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


def run_normalization_comparison():
    """
    Compares multiple normalization methods using entire-image statistics.
    """
    print("\n" + "=" * 50)
    print("NORMALIZATION TECHNIQUES COMPARISON (Entire Image)")
    print("=" * 50)

    # Create output directories
    os.makedirs("Outputs/Normalization_Comparison", exist_ok=True)
    os.makedirs("Outputs/Normalization_Comparison/data", exist_ok=True)
    os.makedirs("Outputs/Normalization_Comparison/visualizations", exist_ok=True)

    # Load all images (subset size controlled by S)
    print("Loading images...")
    sb_images = load_images(SLATY_BACKED_IMG_DIR)[:S]
    gw_images = load_images(GLAUCOUS_WINGED_IMG_DIR)[:S]

    # We no longer need segmentation maps since we're analyzing the entire image

    # Combine images for global min-max
    all_images = [img for _, img in sb_images + gw_images]
    global_min, global_max = compute_global_minmax([(None, img) for img in all_images])
    print(f"Global min: {global_min}, Global max: {global_max}")

    # Define normalization methods
    normalization_methods = [
        "None",
        "Grayscale",
        "MinMax",
        "ZScore",
        "HistEq",
        "CLAHE",
        "GlobalMinMax"
    ]

    # Data structure to hold statistics
    stats_by_method = {method: {"SB": [], "GW": []} for method in normalization_methods}

    # Process each normalization method
    for method in normalization_methods:
        print(f"\nProcessing normalization method: {method}")

        # Create a folder for each method's results
        method_dir = f"Outputs/Normalization_Comparison/{method}"
        os.makedirs(method_dir, exist_ok=True)

        # Prepare a subplot for visual comparison of images
        fig, axes = plt.subplots(2, max(len(sb_images), len(gw_images)), figsize=(15, 6))
        fig.suptitle(f"Normalization Method: {method}", fontsize=16)

        # Process Slaty-backed Gull images
        for i, (fname, img) in enumerate(sb_images):
            # Apply normalization
            if method == "GlobalMinMax":
                norm_img = apply_normalization(img, method, global_min, global_max)
            else:
                norm_img = apply_normalization(img, method)

            # Convert normalized image to intensities (flattened)
            if len(norm_img.shape) == 3:
                # For color images, average the channels
                intensities = norm_img.mean(axis=2).flatten()
            else:
                intensities = norm_img.flatten()

            # Calculate statistics for the entire image
            mean_val = np.mean(intensities)
            std_val = np.std(intensities)

            stats_by_method[method]["SB"].append({
                "mean": mean_val,
                "std": std_val,
                "min": np.min(intensities),
                "max": np.max(intensities)
            })

            # Display the normalized image
            if i < axes[0].shape[0]:
                axes[0, i].imshow(cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB))
                axes[0, i].set_title(f"SB {i + 1}")
                axes[0, i].axis('off')

        # Process Glaucous-winged Gull images
        for i, (fname, img) in enumerate(gw_images):
            # Apply normalization
            if method == "GlobalMinMax":
                norm_img = apply_normalization(img, method, global_min, global_max)
            else:
                norm_img = apply_normalization(img, method)

            # Convert normalized image to intensities (flattened)
            if len(norm_img.shape) == 3:
                intensities = norm_img.mean(axis=2).flatten()
            else:
                intensities = norm_img.flatten()

            # Calculate statistics for the entire image
            mean_val = np.mean(intensities)
            std_val = np.std(intensities)

            stats_by_method[method]["GW"].append({
                "mean": mean_val,
                "std": std_val,
                "min": np.min(intensities),
                "max": np.max(intensities)
            })

            # Display the normalized image
            if i < axes[1].shape[0]:
                axes[1, i].imshow(cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB))
                axes[1, i].set_title(f"GW {i + 1}")
                axes[1, i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{method_dir}/all_images_comparison.png", dpi=150)
        plt.close()

    # Summarize results
    method_summary = []

    for method in normalization_methods:
        sb_means = [stats["mean"] for stats in stats_by_method[method]["SB"]]
        gw_means = [stats["mean"] for stats in stats_by_method[method]["GW"]]

        sb_mean = np.mean(sb_means) if sb_means else 0
        sb_std = np.std(sb_means) if sb_means else 0
        gw_mean = np.mean(gw_means) if gw_means else 0
        gw_std = np.std(gw_means) if gw_means else 0

        # Mean of standard deviations
        sb_local_std = np.mean([stats["std"] for stats in stats_by_method[method]["SB"]]) \
            if stats_by_method[method]["SB"] else 0
        gw_local_std = np.mean([stats["std"] for stats in stats_by_method[method]["GW"]]) \
            if stats_by_method[method]["GW"] else 0

        # Ranges
        sb_min = min([stats["min"] for stats in stats_by_method[method]["SB"]]) \
            if stats_by_method[method]["SB"] else 0
        sb_max = max([stats["max"] for stats in stats_by_method[method]["SB"]]) \
            if stats_by_method[method]["SB"] else 0
        gw_min = min([stats["min"] for stats in stats_by_method[method]["GW"]]) \
            if stats_by_method[method]["GW"] else 0
        gw_max = max([stats["max"] for stats in stats_by_method[method]["GW"]]) \
            if stats_by_method[method]["GW"] else 0

        # Separation between species
        separation = abs(sb_mean - gw_mean)

        # Coefficient of Variation for each species
        sb_cv = sb_std / sb_mean if sb_mean != 0 else 0
        gw_cv = gw_std / gw_mean if gw_mean != 0 else 0

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
            "CombinedCV": (sb_cv + gw_cv) / 2,
        })

    # Save summary to CSV
    summary_df = pd.DataFrame(method_summary)
    summary_df.to_csv("Outputs/Normalization_Comparison/normalization_comparison_stats.csv", index=False)

    # Plotting
    x = np.arange(len(normalization_methods))
    width = 0.35

    # 1) Mean Intensity Comparison
    plt.figure(figsize=(12, 8))
    plt.bar(x - width / 2, summary_df["SB_Mean"], width, yerr=summary_df["SB_StdAcrossImages"],
            label="Slaty-backed", capsize=5, color="blue", alpha=0.7)
    plt.bar(x + width / 2, summary_df["GW_Mean"], width, yerr=summary_df["GW_StdAcrossImages"],
            label="Glaucous-winged", capsize=5, color="green", alpha=0.7)
    plt.xlabel("Normalization Method")
    plt.ylabel("Mean Intensity (Entire Image)")
    plt.title("Comparison of Normalization Methods: Mean Intensity (Entire Image)")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Outputs/Normalization_Comparison/mean_intensity_comparison.png", dpi=150)
    plt.close()

    # 2) Coefficient of Variation (CV)
    plt.figure(figsize=(12, 8))
    plt.bar(x - width / 2, summary_df["SB_CV"], width, label="Slaty-backed CV",
            color="blue", alpha=0.7)
    plt.bar(x + width / 2, summary_df["GW_CV"], width, label="Glaucous-winged CV",
            color="green", alpha=0.7)
    plt.xlabel("Normalization Method")
    plt.ylabel("Coefficient of Variation (Entire Image)")
    plt.title("Comparison of Normalization Methods: CV (Entire Image)")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Outputs/Normalization_Comparison/cv_comparison.png", dpi=150)
    plt.close()

    # 3) Separation Between Species
    plt.figure(figsize=(12, 8))
    plt.bar(x, summary_df["Separation"], width, color="purple", alpha=0.7)
    plt.xlabel("Normalization Method")
    plt.ylabel("Separation (|SB_Mean - GW_Mean|)")
    plt.title("Comparison of Normalization Methods: Separation Between Species (Entire Image)")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Outputs/Normalization_Comparison/separation_comparison.png", dpi=150)
    plt.close()

    # 4) Comprehensive Comparison
    plt.figure(figsize=(14, 10))

    # Subplot 1: Mean Intensities
    plt.subplot(2, 2, 1)
    plt.bar(x - width / 2, summary_df["SB_Mean"], width, yerr=summary_df["SB_StdAcrossImages"],
            label="Slaty-backed", capsize=5, color="blue", alpha=0.7)
    plt.bar(x + width / 2, summary_df["GW_Mean"], width, yerr=summary_df["GW_StdAcrossImages"],
            label="Glaucous-winged", capsize=5, color="green", alpha=0.7)
    plt.xlabel("Normalization Method")
    plt.ylabel("Mean Intensity (Entire Image)")
    plt.title("Mean Intensity (Entire Image)")
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
    plt.ylabel("Average Std Dev (Entire Image)")
    plt.title("Local Variation (Entire Image)")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.grid(alpha=0.3)

    # Subplot 3: Coefficient of Variation
    plt.subplot(2, 2, 3)
    plt.bar(x - width / 2, summary_df["SB_CV"], width, label="Slaty-backed",
            color="blue", alpha=0.7)
    plt.bar(x + width / 2, summary_df["GW_CV"], width, label="Glaucous-winged",
            color="green", alpha=0.7)
    plt.xlabel("Normalization Method")
    plt.ylabel("Coefficient of Variation (Entire Image)")
    plt.title("Coefficient of Variation (Entire Image)")
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
    plt.title("Separation & Combined CV (Entire Image)")
    plt.xticks(x, normalization_methods, rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Outputs/Normalization_Comparison/comprehensive_comparison.png", dpi=150)
    plt.close()

    # Print summary table
    print("\n" + "=" * 100)
    print("NORMALIZATION COMPARISON SUMMARY (Entire Image)")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("\n" + "=" * 100)

    # Determine best method for separation & consistency
    best_separation = summary_df.loc[summary_df["Separation"].idxmax()]
    best_combined_cv = summary_df.loc[summary_df["CombinedCV"].idxmin()]

    print(f"\nBest method for SEPARATION: {best_separation['Method']} "
          f"(Separation = {best_separation['Separation']:.2f})")
    print(f"Best method for CONSISTENCY (lowest CV): {best_combined_cv['Method']} "
          f"(Combined CV = {best_combined_cv['CombinedCV']:.2f})")

    print("\nNormalization comparison complete! Results saved to 'Outputs/Normalization_Comparison/'")


if __name__ == "__main__":
    run_normalization_comparison()
