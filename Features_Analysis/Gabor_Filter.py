from Features_Analysis.config import *

import os
import numpy as np
import cv2
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

# Create output directory for saving results
OUTPUT_DIR = "Results_Gabor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_gabor_filters():
    """
    Create a bank of Gabor filters with different orientations and frequencies.
    """
    filters = []
    num_theta = 4  # Increased from 2 for better coverage
    num_freq = 2  # Number of frequencies
    ksize = 31  # Kernel size
    sigma = 4.0  # Gaussian standard deviation

    for theta in range(num_theta):
        theta_val = theta * np.pi / num_theta
        for freq in range(num_freq):
            frequency = 0.1 + freq * 0.1
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta_val,
                frequency,
                0.5,
                0,
                ktype=cv2.CV_32F
            )
            # Normalize kernel
            kernel /= kernel.sum()
            filter_name = f"gabor_{theta}_{freq}"
            filters.append((filter_name, kernel))

    return filters


def extract_gabor_features(image, mask, filters):
    """
    Extract Gabor filter–based features for a masked region in the image.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # If mask is empty, return None
    if np.sum(mask) == 0:
        return None

    features = {}

    # Apply each Gabor filter and collect stats
    for filter_name, kernel in filters:
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        # Only consider pixels within the mask
        masked_response = filtered[mask > 0]

        features[f"{filter_name}_mean"] = np.mean(masked_response)
        features[f"{filter_name}_std"] = np.std(masked_response)
        features[f"{filter_name}_energy"] = np.sum(masked_response ** 2)

    return features


def plot_gabor_responses(image, mask, filters, title, save_path=None):
    """
    Visualize how each Gabor filter responds to the masked region in the image.
    """
    if np.sum(mask) == 0:
        return

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    n_filters = len(filters)
    n_cols = 4
    # +2 for Original and Masked images
    n_rows = (n_filters + 2 + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))
    plt.suptitle(title, fontsize=16)

    # 1. Original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # 2. Masked image
    plt.subplot(n_rows, n_cols, 2)
    masked_img = gray.copy()
    masked_img[mask == 0] = 0
    plt.imshow(masked_img, cmap='gray')
    plt.title('Masked Region')
    plt.axis('off')

    # 3. Filter responses
    for idx, (filter_name, kernel) in enumerate(filters):
        plt.subplot(n_rows, n_cols, idx + 3)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        filtered[mask == 0] = 0  # Zero out areas outside the mask
        plt.imshow(filtered, cmap='gray')
        plt.title(filter_name)
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_features_summary(sb_features_all, gw_features_all):
    """
    Create a summary dataframe of all features for both species.
    """
    # Initialize dictionaries to store results
    summary = {}
    p_values = {}
    significant_features = []

    # Get all feature keys
    all_keys = set()
    for features_dict in sb_features_all + gw_features_all:
        if features_dict is not None:
            all_keys.update(features_dict.keys())

    # Calculate statistics for each feature
    for key in all_keys:
        sb_values = [f[key] for f in sb_features_all if f is not None and key in f]
        gw_values = [f[key] for f in gw_features_all if f is not None and key in f]

        if len(sb_values) > 0 and len(gw_values) > 0:
            # Calculate means
            sb_mean = np.mean(sb_values)
            gw_mean = np.mean(gw_values)

            # Perform t-test
            t_stat, p_value = ttest_ind(sb_values, gw_values)

            # Store results
            summary[key] = {
                'Slaty-backed Mean': sb_mean,
                'Glaucous-winged Mean': gw_mean,
                'Difference (%)': ((sb_mean - gw_mean) / gw_mean * 100) if gw_mean != 0 else float('inf'),
                'T-statistic': t_stat,
                'P-value': p_value,
                'Significant': p_value < 0.05
            }

            p_values[key] = p_value

            if p_value < 0.05:
                significant_features.append(key)

    return summary, significant_features


def plot_feature_comparison(sb_features, gw_features, feature_name, region_name, save_path=None):
    """
    Create a comparative box plot for a specific Gabor feature across two species.
    """
    sb_values = [f[feature_name] for f in sb_features if f is not None and feature_name in f]
    gw_values = [f[feature_name] for f in gw_features if f is not None and feature_name in f]

    if not sb_values or not gw_values:
        return None, None  # No data to plot

    # Calculate statistics
    t_stat, p_value = ttest_ind(sb_values, gw_values)

    plt.figure(figsize=(8, 5))
    plt.boxplot([sb_values, gw_values], labels=['Slaty-backed', 'Glaucous-winged'])
    plt.title(f"{region_name} - {feature_name}\nT-stat={t_stat:.3f}, p={p_value:.3f}")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return t_stat, p_value


def plot_heatmap(feature_summary, region_name, save_path=None):
    """
    Create a heatmap of p-values to visualize significant features.
    """
    # Extract p-values for all features
    data = {}
    for key, values in feature_summary.items():
        parts = key.split('_')
        if len(parts) >= 4:  # Ensure it's a gabor feature
            filter_type = f"gabor_{parts[1]}_{parts[2]}"
            stat_type = parts[3]

            if filter_type not in data:
                data[filter_type] = {}

            data[filter_type][stat_type] = values['P-value']

    # Convert to dataframe for heatmap
    if not data:
        return

    df = pd.DataFrame(data).T

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='coolwarm_r', vmin=0, vmax=0.10,
                linewidths=.5, cbar_kws={'label': 'p-value'})
    plt.title(f'P-values for {region_name} Features\n(p < 0.05 indicates significant difference)')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def generate_insight_report(summary, significant_features, region_name):
    """
    Generate a text report of insights from the analysis.
    """
    lines = [
        f"====== ANALYSIS REPORT: {region_name} ======",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nNumber of features analyzed: {len(summary)}",
        f"Number of statistically significant features (p < 0.05): {len(significant_features)}",
        "\n----- SIGNIFICANT DIFFERENCES BETWEEN SPECIES -----"
    ]

    if significant_features:
        # Sort by absolute percent difference
        sorted_features = sorted(
            [(k, summary[k]) for k in significant_features],
            key=lambda x: abs(x[1]['Difference (%)']),
            reverse=True
        )

        for feature, stats in sorted_features:
            diff = stats['Difference (%)']
            direction = "higher" if diff > 0 else "lower"
            lines.append(
                f"• {feature}: Slaty-backed gulls show {abs(diff):.1f}% {direction} values "
                f"than Glaucous-winged (p={stats['P-value']:.4f})"
            )

        # Add interpretation
        lines.append("\n----- INTERPRETATION -----")
        texture_features = [f for f in significant_features if "std" in f or "energy" in f]
        orientation_features = {}

        for f in significant_features:
            parts = f.split('_')
            if len(parts) >= 4:
                orientation = parts[1]
                if orientation not in orientation_features:
                    orientation_features[orientation] = []
                orientation_features[orientation].append(f)

        if texture_features:
            lines.append("• Texture differences: The species show different texture patterns in wingtips.")

            std_features = [f for f in significant_features if "std" in f]
            if std_features:
                lines.append(
                    "  - Variation in texture: The standard deviation measures show different levels of texture homogeneity.")

            energy_features = [f for f in significant_features if "energy" in f]
            if energy_features:
                lines.append("  - Texture strength: The energy measures show different overall texture intensities.")

        # Analyze which orientations show differences
        if orientation_features:
            lines.append("• Orientation analysis:")
            for orientation, features in orientation_features.items():
                angle = int(orientation) * 45  # Convert orientation index to degrees
                lines.append(
                    f"  - {angle}° orientation: {len(features)} significant features found at this orientation.")

            max_orientation = max(orientation_features.items(), key=lambda x: len(x[1]))
            max_angle = int(max_orientation[0]) * 45
            lines.append(f"  - The most discriminative orientation is {max_angle}°, suggesting feather patterns")
            lines.append("    or structures aligned in this direction differ most between species.")
    else:
        lines.append("No statistically significant differences were found between species.")

    # Add conclusion
    lines.append("\n----- CONCLUSION -----")
    if significant_features:
        lines.append(
            "The analysis demonstrates quantifiable texture differences in the wingtips between "
            "Slaty-backed and Glaucous-winged gulls. These differences could potentially be used "
            "for automated species identification."
        )
    else:
        lines.append(
            "The analysis did not detect statistically significant texture differences in the wingtips "
            "between Slaty-backed and Glaucous-winged gulls using the current Gabor filter parameters. "
            "Consider adjusting filter parameters or analyzing different regions."
        )

    return "\n".join(lines)


def main():
    print("Starting Improved Gabor-based Texture Analysis...")

    # Number of images to process (set to None to process all images)
    S = None  # Process all available images

    # Create Gabor filters
    filters = create_gabor_filters()

    # Dictionary to store all features by region and species
    all_features = {}

    # --- Load Slaty-backed gull images ---
    sb_images = []
    sb_segs = []
    sb_filenames = sorted(os.listdir(SLATY_BACKED_IMG_DIR))[:S]
    print(f"Found {len(sb_filenames)} Slaty-backed images")

    for img_name in sb_filenames:
        img_path = os.path.join(SLATY_BACKED_IMG_DIR, img_name)
        seg_path = os.path.join(SLATY_BACKED_SEG_DIR, img_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is not None and seg is not None:
            sb_images.append(img)
            sb_segs.append(seg)
        else:
            print(f"Could not load {img_name} or its segmentation.")

    # --- Load Glaucous-winged gull images ---
    gw_images = []
    gw_segs = []
    gw_filenames = sorted(os.listdir(GLAUCOUS_WINGED_IMG_DIR))[:S]
    print(f"Found {len(gw_filenames)} Glaucous-winged images")

    for img_name in gw_filenames:
        img_path = os.path.join(GLAUCOUS_WINGED_IMG_DIR, img_name)
        seg_path = os.path.join(GLAUCOUS_WINGED_SEG_DIR, img_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is not None and seg is not None:
            gw_images.append(img)
            gw_segs.append(seg)
        else:
            print(f"Could not load {img_name} or its segmentation.")

    # --- Analyze each defined region ---
    for region_name, color in REGION_COLORS.items():
        print(f"\nAnalyzing '{region_name}' region...")

        # Create region directory
        region_dir = os.path.join(OUTPUT_DIR, region_name)
        os.makedirs(region_dir, exist_ok=True)

        sb_region_features = []
        gw_region_features = []

        # --- Process Slaty-backed images ---
        for idx, (img, seg) in enumerate(zip(sb_images, sb_segs)):
            tolerance = 10
            lower = np.array([max(c - tolerance, 0) for c in color])
            upper = np.array([min(c + tolerance, 255) for c in color])
            mask = cv2.inRange(seg, lower, upper)

            features = extract_gabor_features(img, mask, filters)
            if features:
                sb_region_features.append(features)

                # Save example Gabor responses for first few images
                if idx < 3:  # Just save first 3 for demonstration
                    save_path = os.path.join(region_dir, f"sb_gabor_response_{idx + 1}.png")
                    plot_gabor_responses(img, mask, filters,
                                         f"Slaty-backed Gull - {region_name} Gabor Responses - Image {idx + 1}",
                                         save_path)

        # --- Process Glaucous-winged images ---
        for idx, (img, seg) in enumerate(zip(gw_images, gw_segs)):
            tolerance = 10
            lower = np.array([max(c - tolerance, 0) for c in color])
            upper = np.array([min(c + tolerance, 255) for c in color])
            mask = cv2.inRange(seg, lower, upper)

            features = extract_gabor_features(img, mask, filters)
            if features:
                gw_region_features.append(features)

                # Save example Gabor responses for first few images
                if idx < 3:  # Just save first 3 for demonstration
                    save_path = os.path.join(region_dir, f"gw_gabor_response_{idx + 1}.png")
                    plot_gabor_responses(img, mask, filters,
                                         f"Glaucous-winged Gull - {region_name} Gabor Responses - Image {idx + 1}",
                                         save_path)

        # Store features for later analysis
        all_features[region_name] = {
            'sb': sb_region_features,
            'gw': gw_region_features
        }

        # --- Generate comprehensive analysis for this region ---
        if sb_region_features and gw_region_features:
            print(f"\nGenerating statistical summary for {region_name} region...")

            # Create feature summary
            feature_summary, significant_features = create_features_summary(
                sb_region_features, gw_region_features)

            # Save summary to CSV
            summary_df = pd.DataFrame.from_dict(feature_summary, orient='index')
            summary_df.to_csv(os.path.join(region_dir, f"{region_name}_statistics.csv"))

            # Plot heatmap of p-values
            heatmap_path = os.path.join(region_dir, f"{region_name}_pvalue_heatmap.png")
            plot_heatmap(feature_summary, region_name, heatmap_path)

            # Plot box plots for significant features
            for feature in significant_features:
                box_plot_path = os.path.join(region_dir, f"{region_name}_{feature}_boxplot.png")
                plot_feature_comparison(
                    sb_region_features, gw_region_features,
                    feature, region_name, box_plot_path)

            # Generate insight report
            report = generate_insight_report(
                feature_summary, significant_features, region_name)

            # Save report
            report_path = os.path.join(region_dir, f"{region_name}_analysis_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)

            # Print report to console
            print("\n" + "=" * 80)
            print(report)
            print("=" * 80)

    print(f"\nAnalysis complete! Results saved to: {OUTPUT_DIR}")
    print("To view the full analysis reports, check the text files in each region directory.")


if __name__ == "__main__":
    main()