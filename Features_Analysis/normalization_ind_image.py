import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import sem

from Features_Analysis.config import (
    SLATY_BACKED_IMG_DIR,
    SLATY_BACKED_SEG_DIR,
    GLAUCOUS_WINGED_IMG_DIR,
    GLAUCOUS_WINGED_SEG_DIR,
    extract_region,
    S
)
from Features_Analysis.image_normalization import (
    to_grayscale,
    minmax_normalize,
    zscore_normalize,
    hist_equalize,
    clahe_normalize,
    compute_global_minmax,
    global_minmax_normalize
)


def load_images_with_segmentation(img_dir, seg_dir):
    """
    Load both original and segmentation images from the given directories.

    Parameters:
        img_dir (str or Path): Directory containing original images
        seg_dir (str or Path): Directory containing segmentation images

    Returns:
        list: List of tuples (filename, original_image, segmentation_image)
    """
    images = []
    img_dir = str(img_dir)
    seg_dir = str(seg_dir)

    # Get all image filenames from the original directory
    img_files = sorted(os.listdir(img_dir))

    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        seg_path = os.path.join(seg_dir, fname)

        # Check if both original and segmentation images exist
        if os.path.isfile(img_path) and os.path.isfile(seg_path):
            original_img = cv2.imread(img_path)
            seg_img = cv2.imread(seg_path)

            if original_img is not None and seg_img is not None:
                images.append((fname, original_img, seg_img))

    return images


def apply_normalization_to_region(original_img, seg_img, region_name, method, global_min=None, global_max=None):
    """
    Apply a specific normalization method to a specific region of an image.

    Parameters:
        original_img (numpy.ndarray): Original image
        seg_img (numpy.ndarray): Segmentation image
        region_name (str): Name of the region to analyze
        method (str): Name of normalization method
        global_min (float, optional): Global minimum for global min-max normalization
        global_max (float, optional): Global maximum for global min-max normalization

    Returns:
        tuple: (normalized_region, mask)
    """
    # Extract the region
    region_img, mask = extract_region(original_img, seg_img, region_name)

    # Skip if region is empty
    if np.sum(mask) == 0:
        return region_img, mask

    # Apply normalization
    if method == "None":
        return region_img, mask
    elif method == "Grayscale":
        gray = to_grayscale(region_img)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), mask
    elif method == "MinMax":
        gray = to_grayscale(region_img)
        # Apply normalization only to the masked region
        normalized = np.zeros_like(gray)
        masked_pixels = gray[mask > 0]

        if len(masked_pixels) > 0:
            min_val = np.min(masked_pixels)
            max_val = np.max(masked_pixels)

            # Normalize only the masked region
            normalized_pixels = (masked_pixels - min_val) / (max_val - min_val + 1e-8) * 255
            normalized[mask > 0] = normalized_pixels.astype(np.uint8)

        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR), mask
    elif method == "ZScore":
        gray = to_grayscale(region_img)
        normalized = np.zeros_like(gray)
        masked_pixels = gray[mask > 0]

        if len(masked_pixels) > 0:
            mean = np.mean(masked_pixels)
            std = np.std(masked_pixels)

            # Z-score normalize only the masked region
            zscored = (masked_pixels - mean) / (std + 1e-8)
            z_min, z_max = np.min(zscored), np.max(zscored)
            normalized_pixels = (zscored - z_min) / (z_max - z_min + 1e-8) * 255
            normalized[mask > 0] = normalized_pixels.astype(np.uint8)

        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR), mask
    elif method == "HistEq":
        gray = to_grayscale(region_img)
        # For histogram equalization, we need to handle the region separately
        normalized = np.zeros_like(gray)
        masked_pixels = gray[mask > 0]

        if len(masked_pixels) > 0 and np.max(masked_pixels) > np.min(masked_pixels):
            # Create a temporary image with just the region
            temp = np.zeros_like(gray)
            temp[mask > 0] = masked_pixels

            # Apply histogram equalization
            hist_eq = cv2.equalizeHist(temp)
            normalized[mask > 0] = hist_eq[mask > 0]

        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR), mask
    elif method == "CLAHE":
        gray = to_grayscale(region_img)
        normalized = np.zeros_like(gray)

        # For CLAHE, create a temp image and apply mask after
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_applied = clahe.apply(gray)
        normalized[mask > 0] = clahe_applied[mask > 0]

        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR), mask
    elif method == "GlobalMinMax":
        if global_min is None or global_max is None:
            raise ValueError("Global min and max must be provided for GlobalMinMax normalization")

        gray = to_grayscale(region_img)
        normalized = np.zeros_like(gray)
        masked_pixels = gray[mask > 0]

        if len(masked_pixels) > 0:
            # Apply global min-max normalization only to the masked region
            normalized_pixels = (masked_pixels - global_min) / (global_max - global_min + 1e-8) * 255
            normalized[mask > 0] = normalized_pixels.astype(np.uint8)

        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR), mask
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_global_minmax(image_list, region_name):
    """
    Compute global min and max for a specific region across all images.

    Parameters:
        image_list (list): List of tuples (filename, original_image, segmentation_image)
        region_name (str): Name of the region to analyze

    Returns:
        tuple: (global_min, global_max)
    """
    global_min = float('inf')
    global_max = float('-inf')

    for fname, orig_img, seg_img in image_list:
        region_img, mask = extract_region(orig_img, seg_img, region_name)

        if np.sum(mask) > 0:  # If the region exists in this image
            gray = to_grayscale(region_img)
            masked_pixels = gray[mask > 0]

            if len(masked_pixels) > 0:
                local_min = np.min(masked_pixels)
                local_max = np.max(masked_pixels)

                if local_min < global_min:
                    global_min = local_min
                if local_max > global_max:
                    global_max = local_max

    return global_min, global_max


def compare_normalization_techniques_per_image(img_dir, seg_dir, region_name="wingtip", limit=5,
                                               species_name="Unknown"):
    """
    Compare multiple normalization methods for a specific region on each individual image.

    Parameters:
        img_dir (str): Directory with original images
        seg_dir (str): Directory with segmentation images
        region_name (str): Name of the region to analyze
        limit (int): Maximum number of images to process
        species_name (str): Name of the species for output labeling
    """
    print(f"\nComparing normalization techniques for each {species_name} image (region: {region_name})")

    # Create output directory
    output_dir = f"NormalisationPerImage/Per_Image_Comparison/{region_name}/{species_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    images = load_images_with_segmentation(img_dir, seg_dir)[:limit]

    if not images:
        print("No images found to process.")
        return

    # Define normalization methods
    normalization_methods = [
        "None", "Grayscale", "MinMax", "ZScore", "HistEq", "CLAHE", "GlobalMinMax"
    ]

    # Calculate global min/max if needed for GlobalMinMax
    global_min, global_max = compute_global_minmax(images, region_name)
    print(f"Global min: {global_min}, Global max: {global_max}")

    # Process each image
    for idx, (fname, orig_img, seg_img) in enumerate(images):
        print(f"Processing image {idx + 1}/{len(images)}: {fname}")

        # Create figure for this image
        fig, axes = plt.subplots(1, len(normalization_methods), figsize=(20, 5))
        fig.suptitle(f"{species_name} - Image: {fname} (Region: {region_name})", fontsize=16)

        # Apply each normalization method to this image
        for method_idx, method in enumerate(normalization_methods):
            # Apply normalization
            if method == "GlobalMinMax":
                norm_region, mask = apply_normalization_to_region(
                    orig_img, seg_img, region_name, method, global_min, global_max
                )
            else:
                norm_region, mask = apply_normalization_to_region(
                    orig_img, seg_img, region_name, method
                )

            # Skip if region not found
            if np.sum(mask) == 0:
                axes[method_idx].text(0.5, 0.5, "Region not found",
                                      ha='center', va='center')
                axes[method_idx].set_title(method)
                axes[method_idx].axis('off')
                continue

            # Display normalized region
            axes[method_idx].imshow(cv2.cvtColor(norm_region, cv2.COLOR_BGR2RGB))
            axes[method_idx].set_title(method)
            axes[method_idx].axis('off')

            # Calculate metrics for display
            region_pixels = to_grayscale(norm_region)[mask > 0]
            if len(region_pixels) > 0:
                mean_val = np.mean(region_pixels)
                std_val = np.std(region_pixels)
                cv_val = (std_val / mean_val * 100) if mean_val > 0 else 0

                # Add metrics text to the plot
                axes[method_idx].set_xlabel(f"Mean: {mean_val:.1f}\nStd: {std_val:.1f}\nCV: {cv_val:.1f}%")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fname.split('.')[0]}_comparison.png", dpi=150)
        plt.close()

        # Also save individual normalized images
        for method in normalization_methods:
            method_dir = f"{output_dir}/{fname.split('.')[0]}"
            os.makedirs(method_dir, exist_ok=True)

            if method == "GlobalMinMax":
                norm_region, mask = apply_normalization_to_region(
                    orig_img, seg_img, region_name, method, global_min, global_max
                )
            else:
                norm_region, mask = apply_normalization_to_region(
                    orig_img, seg_img, region_name, method
                )

            if np.sum(mask) > 0:
                cv2.imwrite(f"{method_dir}/{method}.png", norm_region)

    print(f"Comparison complete! Results saved to '{output_dir}/'")


def run_region_normalization_comparison(region_name="wingtip"):
    """
    Compare multiple normalization methods for a specific region.

    Parameters:
        region_name (str): Name of the region to analyze
    """
    print("\n" + "=" * 50)
    print(f"NORMALIZATION TECHNIQUES COMPARISON (REGION: {region_name})")
    print("=" * 50)

    # Create output directories
    output_base = f"Outputs/Region_Normalization/{region_name}"
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(f"{output_base}/data", exist_ok=True)
    os.makedirs(f"{output_base}/visualizations", exist_ok=True)

    # Load all images with segmentation
    print("Loading images...")
    sb_images = load_images_with_segmentation(SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR)[:S]
    gw_images = load_images_with_segmentation(GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR)[:S]

    # Compute global min-max for the region across all images
    all_images = sb_images + gw_images
    global_min, global_max = compute_global_minmax(all_images, region_name)
    print(f"Global min for {region_name}: {global_min}, Global max: {global_max}")

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
        method_dir = f"{output_base}/{method}"
        os.makedirs(method_dir, exist_ok=True)

        # Prepare a subplot for visual comparison of images
        fig, axes = plt.subplots(2, max(len(sb_images), len(gw_images)), figsize=(15, 6))
        fig.suptitle(f"Normalization Method: {method} (Region: {region_name})", fontsize=16)

        # Process Slaty-backed Gull images
        for i, (fname, orig_img, seg_img) in enumerate(sb_images):
            # Apply normalization to the specific region
            if method == "GlobalMinMax":
                norm_region, mask = apply_normalization_to_region(orig_img, seg_img, region_name, method, global_min,
                                                                  global_max)
            else:
                norm_region, mask = apply_normalization_to_region(orig_img, seg_img, region_name, method)

            # Skip if region not found in this image
            if np.sum(mask) == 0:
                continue

            # Extract only the region pixels for statistics
            region_pixels = to_grayscale(norm_region)[mask > 0]

            # Calculate statistics for the region
            if len(region_pixels) > 0:
                mean_val = np.mean(region_pixels)
                std_val = np.std(region_pixels)

                stats_by_method[method]["SB"].append({
                    "filename": fname,
                    "mean": mean_val,
                    "std": std_val,
                    "min": np.min(region_pixels),
                    "max": np.max(region_pixels)
                })

                # Display the normalized region
                if i < axes[0].shape[0]:
                    axes[0, i].imshow(cv2.cvtColor(norm_region, cv2.COLOR_BGR2RGB))
                    axes[0, i].set_title(f"SB {i + 1}")
                    axes[0, i].axis('off')

                    # Save individual normalized regions
                    cv2.imwrite(f"{method_dir}/SB_{i + 1}_{fname}", norm_region)

        # Process Glaucous-winged Gull images
        for i, (fname, orig_img, seg_img) in enumerate(gw_images):
            # Apply normalization to the specific region
            if method == "GlobalMinMax":
                norm_region, mask = apply_normalization_to_region(orig_img, seg_img, region_name, method, global_min,
                                                                  global_max)
            else:
                norm_region, mask = apply_normalization_to_region(orig_img, seg_img, region_name, method)

            # Skip if region not found in this image
            if np.sum(mask) == 0:
                continue

            # Extract only the region pixels for statistics
            region_pixels = to_grayscale(norm_region)[mask > 0]

            # Calculate statistics for the region
            if len(region_pixels) > 0:
                mean_val = np.mean(region_pixels)
                std_val = np.std(region_pixels)

                stats_by_method[method]["GW"].append({
                    "filename": fname,
                    "mean": mean_val,
                    "std": std_val,
                    "min": np.min(region_pixels),
                    "max": np.max(region_pixels)
                })

                # Display the normalized region
                if i < axes[1].shape[0]:
                    axes[1, i].imshow(cv2.cvtColor(norm_region, cv2.COLOR_BGR2RGB))
                    axes[1, i].set_title(f"GW {i + 1}")
                    axes[1, i].axis('off')

                    # Save individual normalized regions
                    cv2.imwrite(f"{method_dir}/GW_{i + 1}_{fname}", norm_region)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{method_dir}/all_images_comparison.png", dpi=150)
        plt.close()

    # Summarize results
    method_summary = []

    for method in normalization_methods:
        if not stats_by_method[method]["SB"] or not stats_by_method[method]["GW"]:
            print(f"Skipping summary for {method} - missing data for one or both species")
            continue

        sb_means = [stats["mean"] for stats in stats_by_method[method]["SB"]]
        gw_means = [stats["mean"] for stats in stats_by_method[method]["GW"]]

        sb_mean = np.mean(sb_means) if sb_means else 0
        sb_std = np.std(sb_means) if sb_means else 0
        gw_mean = np.mean(gw_means) if gw_means else 0
        gw_std = np.std(gw_means) if gw_means else 0

        # Mean of standard deviations (how consistent is each image internally)
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

        # Within-species consistency (lower is better)
        sb_cv = (sb_std / sb_mean) * 100 if sb_mean != 0 else 0  # Coefficient of Variation as percentage
        gw_cv = (gw_std / gw_mean) * 100 if gw_mean != 0 else 0

        # Between-species separation (higher is better)
        separation = abs(sb_mean - gw_mean)

        # Combined metric: within-species consistency
        avg_within_species_cv = (sb_cv + gw_cv) / 2

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
            "AvgWithinSpeciesCV": avg_within_species_cv,
        })

    # Save summary to CSV
    summary_df = pd.DataFrame(method_summary)
    summary_df.to_csv(f"{output_base}/normalization_comparison_stats.csv", index=False)

    # Create visualization plots
    if len(method_summary) > 0:
        x = np.arange(len(method_summary))
        methods = [m["Method"] for m in method_summary]
        width = 0.35

        # 1) Within-species Consistency (CV) - LOWER IS BETTER
        plt.figure(figsize=(12, 8))
        plt.bar(x - width / 2, [m["SB_CV"] for m in method_summary], width,
                label="Slaty-backed CV", color="blue", alpha=0.7)
        plt.bar(x + width / 2, [m["GW_CV"] for m in method_summary], width,
                label="Glaucous-winged CV", color="green", alpha=0.7)
        plt.xlabel("Normalization Method")
        plt.ylabel("Coefficient of Variation (%) - Lower is Better")
        plt.title(f"Within-Species Consistency for {region_name} Region")
        plt.xticks(x, methods, rotation=45)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_base}/within_species_consistency.png", dpi=150)
        plt.close()

        # 2) Average Within-Species CV - LOWER IS BETTER
        plt.figure(figsize=(12, 8))
        cv_bars = plt.bar(x, [m["AvgWithinSpeciesCV"] for m in method_summary], width,
                          color="purple", alpha=0.7)
        plt.xlabel("Normalization Method")
        plt.ylabel("Average CV (%) - Lower is Better")
        plt.title(f"Average Within-Species Consistency for {region_name} Region")
        plt.xticks(x, methods, rotation=45)

        # Add value labels on bars
        for bar in cv_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{height:.1f}%', ha='center', va='bottom')

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_base}/avg_within_species_cv.png", dpi=150)
        plt.close()

        # 3) Comprehensive Comparison
        plt.figure(figsize=(14, 10))

        # Subplot 1: Mean Intensity by species
        plt.subplot(2, 2, 1)
        plt.bar(x - width / 2, [m["SB_Mean"] for m in method_summary], width,
                yerr=[m["SB_StdAcrossImages"] for m in method_summary],
                label="Slaty-backed", capsize=5, color="blue", alpha=0.7)
        plt.bar(x + width / 2, [m["GW_Mean"] for m in method_summary], width,
                yerr=[m["GW_StdAcrossImages"] for m in method_summary],
                label="Glaucous-winged", capsize=5, color="green", alpha=0.7)
        plt.xlabel("Normalization Method")
        plt.ylabel("Mean Intensity")
        plt.title(f"Mean Intensity for {region_name}")
        plt.xticks(x, methods, rotation=45)
        plt.legend()
        plt.grid(alpha=0.3)

        # Subplot 2: Within-Species Consistency
        plt.subplot(2, 2, 2)
        plt.bar(x - width / 2, [m["SB_CV"] for m in method_summary], width,
                label="Slaty-backed", color="blue", alpha=0.7)
        plt.bar(x + width / 2, [m["GW_CV"] for m in method_summary], width,
                label="Glaucous-winged", color="green", alpha=0.7)
        plt.xlabel("Normalization Method")
        plt.ylabel("CV (%) - Lower is Better")
        plt.title(f"Within-Species Consistency for {region_name}")
        plt.xticks(x, methods, rotation=45)
        plt.legend()
        plt.grid(alpha=0.3)

        # Subplot 3: Local Std Dev (internal consistency)
        plt.subplot(2, 2, 3)
        plt.bar(x - width / 2, [m["SB_LocalStd"] for m in method_summary], width,
                label="Slaty-backed", color="blue", alpha=0.7)
        plt.bar(x + width / 2, [m["GW_LocalStd"] for m in method_summary], width,
                label="Glaucous-winged", color="green", alpha=0.7)
        plt.xlabel("Normalization Method")
        plt.ylabel("Local Std Dev")
        plt.title(f"Local Standard Deviation for {region_name}")
        plt.xticks(x, methods, rotation=45)
        plt.legend()
        plt.grid(alpha=0.3)

        # Subplot 4: Average CV vs Separation
        plt.subplot(2, 2, 4)
        plt.bar(x - width / 2, [m["AvgWithinSpeciesCV"] for m in method_summary], width,
                label="Avg Within-Species CV", color="purple", alpha=0.7)
        plt.bar(x + width / 2, [m["Separation"] for m in method_summary], width,
                label="Between-Species Separation", color="orange", alpha=0.7)
        plt.xlabel("Normalization Method")
        plt.ylabel("Metric Value")
        plt.title(f"Consistency vs. Separation for {region_name}")
        plt.xticks(x, methods, rotation=45)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_base}/comprehensive_comparison.png", dpi=150)
        plt.close()

        # Print summary table
        print("\n" + "=" * 100)
        print(f"NORMALIZATION COMPARISON SUMMARY (REGION: {region_name})")
        print("=" * 100)
        print(summary_df.to_string(index=False))
        print("\n" + "=" * 100)

        # Determine best method for within-species consistency
        best_consistency = summary_df.loc[summary_df["AvgWithinSpeciesCV"].idxmin()]
        print(f"\nBest method for WITHIN-SPECIES CONSISTENCY: {best_consistency['Method']} "
              f"(Average CV = {best_consistency['AvgWithinSpeciesCV']:.2f}%)")

        # Rank methods by consistency (lower CV is better)
        consistency_ranked = summary_df.sort_values("AvgWithinSpeciesCV")
        print("\nMethods ranked by within-species consistency (lower CV is better):")
        for i, (_, row) in enumerate(consistency_ranked.iterrows(), 1):
            print(f"{i}. {row['Method']}: CV = {row['AvgWithinSpeciesCV']:.2f}%")

    print(f"\nRegion normalization comparison for '{region_name}' complete! "
          f"Results saved to '{output_base}/'")


if __name__ == "__main__":
    # Set which region to analyze
    region_name = "entire_bird"  # Change to: "wingtip", "wing", "head", "body", or "entire_bird"

    # Set the number of images to process
    limit = 5  # Change this to process more or fewer images

    # Compare techniques for each individual image
    compare_normalization_techniques_per_image(
        SLATY_BACKED_IMG_DIR,
        SLATY_BACKED_SEG_DIR,
        region_name=region_name,
        limit=limit,
        species_name="Slaty-backed"
    )

    compare_normalization_techniques_per_image(
        GLAUCOUS_WINGED_IMG_DIR,
        GLAUCOUS_WINGED_SEG_DIR,
        region_name=region_name,
        limit=limit,
        species_name="Glaucous-winged"
    )

    # Run the full comparison if needed
    # Uncomment to run the full analysis across all images
    # run_region_normalization_comparison(region_name)