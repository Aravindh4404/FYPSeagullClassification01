"""
Features_Analysis_Intensity_Only.py

This version removes LBP-based texture analysis. It only calculates intensity
statistics (mean, std, skew, kurtosis, etc.) for each region.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from Features_Analysis.config import *

###############################################################################
# 1. SINGLE-IMAGE ANALYSIS (Intensity Only)
###############################################################################
def analyze_single_image(
    image, seg_map, species_name, region_colors, image_idx, save_visualization=True
):
    """
    Analyzes a single bird image for intensity-based statistics only.
    Returns a list of dictionaries (region_stats) with stats for each region.
    """

    # OPTIONAL: Create a figure for visualization
    if save_visualization:
        fig = plt.figure(figsize=(18, 15))
        gs = plt.GridSpec(3, 2, figure=fig)

        # (Row 0, spanning 2 cols): Original image
        ax_img = fig.add_subplot(gs[0, :])
        ax_img.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_img.set_title(f"{species_name} - Original Image {image_idx + 1}")
        ax_img.axis("off")

    # Prepare data structures
    region_stats = []
    all_intensities = []

    # For each region, create a mask & compute intensity statistics
    for region_name, bgr_color in region_colors.items():
        tolerance = 10
        lower = np.array([max(c - tolerance, 0) for c in bgr_color], dtype=np.uint8)
        upper = np.array([min(c + tolerance, 255) for c in bgr_color], dtype=np.uint8)
        mask = cv2.inRange(seg_map, lower, upper)

        # Draw contour if needed
        if save_visualization and np.sum(mask) > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ax_img.contour(mask, levels=[0.5], colors="red", linewidths=1)

        selected_pixels = image[mask > 0]  # shape: (N, 3) if color

        if len(selected_pixels) > 0:
            # Convert to grayscale intensity by averaging the RGB channels
            intensities = selected_pixels.mean(axis=1)

            all_intensities.append(intensities)

            # Basic stats
            mean_val = np.mean(intensities)
            std_val = np.std(intensities)
            stats_dict = {
                "region": region_name,
                "mean": mean_val,
                "std": std_val,
                "variance": np.var(intensities),
                "median": np.median(intensities),
                "skew": skew(intensities),
                "kurt": kurtosis(intensities),
                "min": np.min(intensities),
                "max": np.max(intensities),
                "range": np.ptp(intensities),
                "cv": std_val / mean_val if mean_val != 0 else 0,
                "species": species_name,
                "image_idx": image_idx,
            }

            # Local variability (optional)
            if len(intensities) > 1:
                local_diffs = np.abs(np.diff(intensities))
                stats_dict.update(
                    {
                        "local_var_mean": np.mean(local_diffs),
                        "local_var_std": np.std(local_diffs),
                        "local_var_max": np.max(local_diffs),
                    }
                )
            else:
                stats_dict.update(
                    {
                        "local_var_mean": 0,
                        "local_var_std": 0,
                        "local_var_max": 0,
                    }
                )

            region_stats.append(stats_dict)

    # If visualization, create subplots for single-image stats
    if save_visualization and len(region_stats) > 0:
        # Row 1, col 0: Histogram
        ax_hist = fig.add_subplot(gs[1, 0])
        for i, intens in enumerate(all_intensities):
            ax_hist.hist(intens, bins=30, alpha=0.7, label=f"{region_stats[i]['region']}")
        ax_hist.set_title("Intensity Distribution")
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        # Row 1, col 1: Mean ± Std bar
        ax_bar = fig.add_subplot(gs[1, 1])
        regions = [r["region"] for r in region_stats]
        means = [r["mean"] for r in region_stats]
        stds = [r["std"] for r in region_stats]
        ax_bar.bar(regions, means, yerr=stds, capsize=5)
        ax_bar.set_title("Mean Intensity (± Std)")
        ax_bar.grid(True, alpha=0.3)

        # Row 2, col 0: Coefficient of Variation
        ax_cv = fig.add_subplot(gs[2, 0])
        cv_values = [r["cv"] for r in region_stats]
        ax_cv.bar(regions, cv_values)
        ax_cv.set_title("Coefficient of Variation")
        ax_cv.grid(True, alpha=0.3)

        # Row 2, col 1: Table of final stats
        ax_table = fig.add_subplot(gs[2, 1])
        ax_table.axis("off")
        table_data = []
        headers = ["Region", "Mean", "Std", "Median", "CV", "Skew", "Kurt"]
        table_data.append(headers)

        for r in region_stats:
            row = [
                r["region"],
                f"{r['mean']:.2f}",
                f"{r['std']:.2f}",
                f"{r['median']:.2f}",
                f"{r['cv']:.2f}",
                f"{r['skew']:.2f}",
                f"{r['kurt']:.2f}",
            ]
            table_data.append(row)

        table = ax_table.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        plt.suptitle(
            f"Detailed Statistical Analysis - {species_name} Image {image_idx + 1}",
            fontsize=16,
            y=0.95,
        )
        plt.tight_layout()

        os.makedirs("../Outputs/Old_Intensity/visualizations", exist_ok=True)
        figpath = f"Intensity_Results/visualizations/{species_name.replace(' ', '_')}_image_{image_idx + 1}.png"
        plt.savefig(figpath)
        plt.close()

    return region_stats


###############################################################################
# 2. SINGLE-SPECIES ANALYSIS (Intensity Only)
###############################################################################
def analyze_single_species(images, seg_maps, species_name, region_colors):
    """
    Analyze all images for a single species (intensity only).
    Returns summary_stats (dict) and region_df (DataFrame).
    """

    all_region_stats = []

    print(f"\n{'='*50}")
    print(f"Analysis for {species_name}")
    print(f"{'='*50}")

    # Loop over each image
    for idx, (img, seg) in enumerate(zip(images, seg_maps)):
        print(f"  Processing image {idx+1}/{len(images)}")
        r_stats = analyze_single_image(
            img, seg, species_name, region_colors, idx, save_visualization=True
        )
        if r_stats:
            all_region_stats.extend(r_stats)

    # Convert to DataFrame
    if len(all_region_stats) > 0:
        region_df = pd.DataFrame(all_region_stats)

        os.makedirs("../Outputs/Old_Intensity/data", exist_ok=True)
        region_csv = f"Intensity_Results/data/{species_name.replace(' ', '_')}_region_stats.csv"
        region_df.to_csv(region_csv, index=False)
        print(f"Saved region-level stats to {region_csv}")

        # Summary for final comparison
        summary_stats = {
            "means": region_df.groupby("region")["mean"].mean().values,
            "stds": region_df.groupby("region")["std"].mean().values,
            "variances": region_df.groupby("region")["variance"].mean().values,
            "cv": region_df.groupby("region")["cv"].mean().values,
            "skewness": region_df.groupby("region")["skew"].mean().values,
            "kurtosis": region_df.groupby("region")["kurt"].mean().values,
            "medians": region_df.groupby("region")["median"].mean().values,
            "region_names": region_df["region"].unique(),
        }

        return summary_stats, region_df

    return None, None


###############################################################################
# 3. EXTENDED REGION COMPARISON PLOT (MULTIPLE SUBPLOTS)
###############################################################################
def plot_comparative_statistics_all(sb_df, gw_df, output_dir="Intensity_Results/comparisons"):
    """
    Creates a single figure with multiple subplots comparing Slaty-backed vs Glaucous-winged
    across regions for the following metrics:
      1) Mean ± std
      2) Variance
      3) Median
      4) Coefficient of Variation (CV)
      5) Histograms of the 'mean' distribution across regions
      6) Boxplot of means
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # Group each species by region, computing "mean of mean" & "std of mean", etc.
    sb_grouped = sb_df.groupby("region").agg(
        mean_mean=("mean", "mean"),
        mean_std=("mean", "std"),
        var_mean=("variance", "mean"),
        var_std=("variance", "std"),
        median_mean=("median", "mean"),
        median_std=("median", "std"),
        cv_mean=("cv", "mean"),
        cv_std=("cv", "std"),
    ).reset_index()

    gw_grouped = gw_df.groupby("region").agg(
        mean_mean=("mean", "mean"),
        mean_std=("mean", "std"),
        var_mean=("variance", "mean"),
        var_std=("variance", "std"),
        median_mean=("median", "mean"),
        median_std=("median", "std"),
        cv_mean=("cv", "mean"),
        cv_std=("cv", "std"),
    ).reset_index()

    # Combine region names
    all_regions = sorted(set(sb_grouped["region"]).union(gw_grouped["region"]))

    # Convert to dictionary lookups
    sb_dict = sb_grouped.set_index("region").to_dict(orient="index")
    gw_dict = gw_grouped.set_index("region").to_dict(orient="index")

    # For each region, build lists for side-by-side bars
    sb_means, sb_means_std = [], []
    sb_vars, sb_vars_std = [], []
    sb_meds, sb_meds_std = [], []
    sb_cvs, sb_cvs_std = [], []

    gw_means, gw_means_std = [], []
    gw_vars, gw_vars_std = [], []
    gw_meds, gw_meds_std = [], []
    gw_cvs, gw_cvs_std = [], []

    for region in all_regions:
        # Slaty-backed
        if region in sb_dict:
            sb_means.append(sb_dict[region]["mean_mean"])
            sb_means_std.append(sb_dict[region]["mean_std"] if not np.isnan(sb_dict[region]["mean_std"]) else 0)
            sb_vars.append(sb_dict[region]["var_mean"])
            sb_vars_std.append(sb_dict[region]["var_std"] if not np.isnan(sb_dict[region]["var_std"]) else 0)
            sb_meds.append(sb_dict[region]["median_mean"])
            sb_meds_std.append(sb_dict[region]["median_std"] if not np.isnan(sb_dict[region]["median_std"]) else 0)
            sb_cvs.append(sb_dict[region]["cv_mean"])
            sb_cvs_std.append(sb_dict[region]["cv_std"] if not np.isnan(sb_dict[region]["cv_std"]) else 0)
        else:
            sb_means.append(0); sb_means_std.append(0)
            sb_vars.append(0); sb_vars_std.append(0)
            sb_meds.append(0); sb_meds_std.append(0)
            sb_cvs.append(0); sb_cvs_std.append(0)

        # Glaucous-winged
        if region in gw_dict:
            gw_means.append(gw_dict[region]["mean_mean"])
            gw_means_std.append(gw_dict[region]["mean_std"] if not np.isnan(gw_dict[region]["mean_std"]) else 0)
            gw_vars.append(gw_dict[region]["var_mean"])
            gw_vars_std.append(gw_dict[region]["var_std"] if not np.isnan(gw_dict[region]["var_std"]) else 0)
            gw_meds.append(gw_dict[region]["median_mean"])
            gw_meds_std.append(gw_dict[region]["median_std"] if not np.isnan(gw_dict[region]["median_std"]) else 0)
            gw_cvs.append(gw_dict[region]["cv_mean"])
            gw_cvs_std.append(gw_dict[region]["cv_std"] if not np.isnan(gw_dict[region]["cv_std"]) else 0)
        else:
            gw_means.append(0); gw_means_std.append(0)
            gw_vars.append(0); gw_vars_std.append(0)
            gw_meds.append(0); gw_meds_std.append(0)
            gw_cvs.append(0); gw_cvs_std.append(0)

    # Create a 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    x = np.arange(len(all_regions))
    width = 0.35

    # Subplot 0: Mean ± std
    ax0 = axes[0]
    ax0.bar(x - width / 2, sb_means, width,
            yerr=sb_means_std, capsize=5, label="Slaty-backed")
    ax0.bar(x + width / 2, gw_means, width,
            yerr=gw_means_std, capsize=5, label="Glaucous-winged")
    ax0.set_title("Mean Intensity (± std)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(all_regions, rotation=45, ha="right")
    ax0.grid(alpha=0.3)
    ax0.legend()

    # Subplot 1: Variance
    ax1 = axes[1]
    ax1.bar(x - width / 2, sb_vars, width,
            yerr=sb_vars_std, capsize=5, label="Slaty-backed")
    ax1.bar(x + width / 2, gw_vars, width,
            yerr=gw_vars_std, capsize=5, label="Glaucous-winged")
    ax1.set_title("Variance (± std)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_regions, rotation=45, ha="right")
    ax1.grid(alpha=0.3)

    # Subplot 2: Median
    ax2 = axes[2]
    ax2.bar(x - width / 2, sb_meds, width,
            yerr=sb_meds_std, capsize=5, label="Slaty-backed")
    ax2.bar(x + width / 2, gw_meds, width,
            yerr=gw_meds_std, capsize=5, label="Glaucous-winged")
    ax2.set_title("Median")
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_regions, rotation=45, ha="right")
    ax2.grid(alpha=0.3)

    # Subplot 3: Coefficient of Variation
    ax3 = axes[3]
    ax3.bar(x - width / 2, sb_cvs, width,
            yerr=sb_cvs_std, capsize=5, label="Slaty-backed")
    ax3.bar(x + width / 2, gw_cvs, width,
            yerr=gw_cvs_std, capsize=5, label="Glaucous-winged")
    ax3.set_title("Coefficient of Variation")
    ax3.set_xticks(x)
    ax3.set_xticklabels(all_regions, rotation=45, ha="right")
    ax3.grid(alpha=0.3)

    # Subplot 4: Histograms of Mean
    ax4 = axes[4]
    ax4.hist(sb_df["mean"], bins=15, alpha=0.6, label="SB Means")
    ax4.hist(gw_df["mean"], bins=15, alpha=0.6, label="GW Means")
    ax4.set_title("Histogram of 'mean' Values (All Regions & Images)")
    ax4.set_xlabel("Mean Intensity")
    ax4.set_ylabel("Frequency")
    ax4.grid(alpha=0.3)
    ax4.legend()

    # Subplot 5: Boxplot of Means
    ax5 = axes[5]
    ax5.boxplot([sb_df["mean"], gw_df["mean"]], labels=["Slaty-backed", "Glaucous-winged"])
    ax5.set_title("Boxplot of Means")
    ax5.grid(alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(output_dir, "region_comparison_all_metrics.png")
    plt.savefig(outpath)
    plt.close()

    print(f"Saved extended comparison figure to: {outpath}")


###############################################################################
# 4. MAIN FUNCTION (Intensity Only)
###############################################################################
def main():
    # Example config. Adjust as needed or import from your config file.


    os.makedirs("../Outputs/Old_Intensity/data", exist_ok=True)
    os.makedirs("../Outputs/Old_Intensity/visualizations", exist_ok=True)
    os.makedirs("../Outputs/Old_Intensity/comparisons", exist_ok=True)

    print("Loading images and segmentation maps...")

    # A) Load Slaty-backed
    sb_images, sb_segs = [], []
    if os.path.exists(SLATY_BACKED_IMG_DIR):
        sb_filenames = sorted(os.listdir(SLATY_BACKED_IMG_DIR))[:S]
        for img_name in sb_filenames:
            img_path = os.path.join(SLATY_BACKED_IMG_DIR, img_name)
            seg_path = os.path.join(SLATY_BACKED_SEG_DIR, img_name)
            img = cv2.imread(img_path)
            seg = cv2.imread(seg_path)
            if img is not None and seg is not None:
                sb_images.append(img)
                sb_segs.append(seg)

    # B) Load Glaucous-winged
    gw_images, gw_segs = [], []
    if os.path.exists(GLAUCOUS_WINGED_IMG_DIR):
        gw_filenames = sorted(os.listdir(GLAUCOUS_WINGED_IMG_DIR))[:S]
        for img_name in gw_filenames:
            img_path = os.path.join(GLAUCOUS_WINGED_IMG_DIR, img_name)
            seg_path = os.path.join(GLAUCOUS_WINGED_SEG_DIR, img_name)
            img = cv2.imread(img_path)
            seg = cv2.imread(seg_path)
            if img is not None and seg is not None:
                gw_images.append(img)
                gw_segs.append(seg)

    # C) Analyze each species (intensity only)
    print("Analyzing Slaty-backed Gull images...")
    sb_stats, sb_df = analyze_single_species(sb_images, sb_segs, "Slaty-backed Gull", REGION_COLORS)

    print("Analyzing Glaucous-winged Gull images...")
    gw_stats, gw_df = analyze_single_species(gw_images, gw_segs, "Glaucous-winged Gull", REGION_COLORS)

    # D) Combine Data & Plot Extended Comparison
    print("Combining data for all species...")
    if sb_df is not None and gw_df is not None:
        all_df = pd.concat([sb_df, gw_df])
        all_df.to_csv("Intensity_Results/data/all_species_region_stats.csv", index=False)

        # Now create an extended multi-subplot figure with all metrics
        print("Creating extended region comparison plot (means, medians, variance, CV, histograms)...")
        plot_comparative_statistics_all(sb_df, gw_df, output_dir="../Outputs/Old_Intensity/comparisons")

    print("\nAnalysis complete! See 'Intensity_Results' folder for results.")
    print("- 'Intensity_Results/data': CSV files with stats")
    print("- 'Intensity_Results/visualizations': Single-image figures (histograms, bar plots, tables)")
    print("- 'Intensity_Results/comparisons': Cross-species comparison plots")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()