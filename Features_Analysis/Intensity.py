
from Features_Analysis.utils import *
from Features_Analysis.config import *

def plot_species_statistics(stats, species_name):
    """Create detailed statistical plots for a single species."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Statistical Analysis for {species_name}', fontsize=16, y=1.02)

    # Plot 1: Mean and SD across images
    ax = axes[0, 0]
    x = range(len(stats['means']))
    ax.errorbar(x, stats['means'], yerr=stats['stds'], fmt='o-', capsize=5,
                label='Mean ± SD')
    ax.set_xlabel('Image Number')
    ax.set_ylabel('Intensity')
    ax.set_title('Mean Intensity with Standard Deviation')
    ax.grid(True, alpha=0.3)
    # Add mean of means line
    mean_of_means = np.mean(stats['means'])
    ax.axhline(y=mean_of_means, color='r', linestyle='--',
               label=f'Mean of Means: {mean_of_means:.2f}')
    ax.legend()

    # Plot 2: Distribution of means with standard error
    ax = axes[0, 1]
    mean_of_means = np.mean(stats['means'])
    std_error = sem(stats['means'])
    ax.hist(stats['means'], bins=10, alpha=0.7)
    ax.axvline(mean_of_means, color='r', linestyle='--',
               label=f'Mean: {mean_of_means:.2f}\nSE: ±{std_error:.2f}')
    ax.set_xlabel('Mean Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Mean Intensities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Box plot of all statistics
    ax = axes[1, 0]
    stats_to_plot = ['means', 'stds', 'medians']
    data_to_plot = [stats[key] for key in stats_to_plot]
    ax.boxplot(data_to_plot, labels=['Means', 'StdDevs', 'Medians'])
    ax.set_ylabel('Value')
    ax.set_title('Distribution of Statistical Measures')
    ax.grid(True, alpha=0.3)

    # Plot 4: Trend of skewness and kurtosis
    ax = axes[1, 1]
    x = range(len(stats['skewness']))
    ax.plot(x, stats['skewness'], 'o-', label='Skewness')
    ax.plot(x, stats['kurtosis'], 's-', label='Kurtosis')
    ax.set_xlabel('Image Number')
    ax.set_ylabel('Value')
    ax.set_title('Skewness and Kurtosis Trends')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Additional plot: Violin plot for all measures
    plt.figure(figsize=(10, 6))
    stats_to_plot = ['means', 'stds', 'medians', 'skewness', 'kurtosis']
    data_to_plot = [stats[key] for key in stats_to_plot]
    plt.violinplot(data_to_plot)
    plt.xticks(range(1, len(stats_to_plot) + 1),
               ['Means', 'StdDevs', 'Medians', 'Skewness', 'Kurtosis'])
    plt.title(f'Distribution of All Statistical Measures for {species_name}')
    plt.grid(True, alpha=0.3)
    plt.show()


import os
import cv2
import numpy as np
from scipy.stats import skew, kurtosis, ks_2samp, sem
import matplotlib.pyplot as plt
from scipy import stats


def analyze_single_image(image, seg_map, species_name, region_colors, image_idx):
    """Detailed analysis of a single bird image with multiple visualizations."""
    # Create a larger figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 3, figure=fig)

    # 1. Original image with mask overlay (larger)
    ax_img = fig.add_subplot(gs[0, :2])
    ax_img.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"{species_name} - Original Image {image_idx + 1}")
    ax_img.axis('off')

    # Storage for region statistics
    all_intensities = []
    region_stats = []

    # Process each region
    for region_name, bgr_color in region_colors.items():
        # Create mask
        tolerance = 10
        lower = np.array([max(c - tolerance, 0) for c in bgr_color], dtype=np.uint8)
        upper = np.array([min(c + tolerance, 255) for c in bgr_color], dtype=np.uint8)
        mask = cv2.inRange(seg_map, lower, upper)

        # Overlay contour on original image
        if np.sum(mask) > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ax_img.contour(mask, levels=[0.5], colors='red', linewidths=1)

        # Extract pixels and compute statistics
        selected_pixels = image[mask > 0]
        if len(selected_pixels) > 0:
            intensities = selected_pixels.mean(axis=1) if len(selected_pixels.shape) > 1 else selected_pixels
            all_intensities.append(intensities)

            # Calculate statistics
            stats_dict = {
                'region': region_name,
                'mean': np.mean(intensities),
                'std': np.std(intensities),
                'median': np.median(intensities),
                'skew': skew(intensities),
                'kurt': kurtosis(intensities),
                'min': np.min(intensities),
                'max': np.max(intensities)
            }
            region_stats.append(stats_dict)

    if region_stats:
        # 2. Main histogram (larger)
        ax_hist = fig.add_subplot(gs[0, 2])
        for i, intensities in enumerate(all_intensities):
            ax_hist.hist(intensities, bins=30, alpha=0.7,
                         label=f"{region_stats[i]['region']}")
        ax_hist.set_title("Intensity Distribution")
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        # 3. Box plot of intensities
        ax_box = fig.add_subplot(gs[1, 0])
        ax_box.boxplot([stats['mean'] for stats in region_stats],
                       labels=[stats['region'] for stats in region_stats])
        ax_box.set_title("Distribution of Mean Intensities")
        ax_box.grid(True, alpha=0.3)

        # 4. Bar plot of means with standard deviation
        ax_bar = fig.add_subplot(gs[1, 1])
        regions = [stats['region'] for stats in region_stats]
        means = [stats['mean'] for stats in region_stats]
        stds = [stats['std'] for stats in region_stats]
        ax_bar.bar(regions, means, yerr=stds, capsize=5)
        ax_bar.set_title("Mean Intensity with Standard Deviation")
        ax_bar.grid(True, alpha=0.3)

        # 5. Violin plot of intensity distributions
        ax_violin = fig.add_subplot(gs[1, 2])
        ax_violin.violinplot(all_intensities)
        ax_violin.set_xticks(range(1, len(regions) + 1))
        ax_violin.set_xticklabels(regions)
        ax_violin.set_title("Intensity Distribution (Violin Plot)")
        ax_violin.grid(True, alpha=0.3)

        # 6. Statistics table
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        table_data = []
        headers = ['Region', 'Mean', 'Std', 'Median', 'Skewness', 'Kurtosis', 'Min', 'Max']
        table_data.append(headers)
        for stats in region_stats:
            row = [
                stats['region'],
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['median']:.2f}",
                f"{stats['skew']:.2f}",
                f"{stats['kurt']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}"
            ]
            table_data.append(row)

        table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        plt.suptitle(f"Detailed Statistical Analysis - {species_name} Image {image_idx + 1}",
                     fontsize=16, y=0.95)
        plt.tight_layout()
        plt.show()

        # Print statistics to console
        print(f"\nDetailed Statistics for {species_name} Image {image_idx + 1}:")
        print("-" * 50)
        for stats in region_stats:
            print(f"\n{stats['region']} Region:")
            for key, value in stats.items():
                if key != 'region':
                    print(f"{key.capitalize()}: {value:.2f}")

        # Return statistics for overall analysis
        return {
            'means': np.mean([s['mean'] for s in region_stats]),
            'stds': np.mean([s['std'] for s in region_stats]),
            'skewness': np.mean([s['skew'] for s in region_stats]),
            'kurtosis': np.mean([s['kurt'] for s in region_stats]),
            'min_values': np.mean([s['min'] for s in region_stats]),
            'max_values': np.mean([s['max'] for s in region_stats]),
            'medians': np.mean([s['median'] for s in region_stats])
        }

    return None


def analyze_single_species(images, seg_maps, species_name, region_colors):
    """Analyze all images for a single species."""
    all_stats = {
        'means': [],
        'stds': [],
        'skewness': [],
        'kurtosis': [],
        'min_values': [],
        'max_values': [],
        'medians': []
    }

    print(f"\n{'=' * 50}")
    print(f"Analysis for {species_name}")
    print(f"{'=' * 50}")

    for idx, (image, seg_map) in enumerate(zip(images, seg_maps)):
        stats = analyze_single_image(image, seg_map, species_name, region_colors, idx)
        if stats:
            for key in all_stats:
                all_stats[key].append(stats[key])

    return all_stats

def plot_comparative_statistics(sb_stats, gw_stats):
    """Create comparative visualizations of statistics."""
    # Define statistics to plot
    stat_pairs = [
        ('means', 'Mean Intensity'),
        ('stds', 'Standard Deviation'),
        ('skewness', 'Skewness'),
        ('kurtosis', 'Kurtosis'),
        ('medians', 'Median Intensity')
    ]

    # Create subplot for each statistic
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, (stat_key, stat_name) in enumerate(stat_pairs):
        if idx < len(axes):
            ax = axes[idx]

            # Calculate means and standard errors
            sb_mean = np.mean(sb_stats[stat_key])
            sb_se = sem(sb_stats[stat_key])
            gw_mean = np.mean(gw_stats[stat_key])
            gw_se = sem(gw_stats[stat_key])

            # Create bar plot
            species = ['Slaty-backed', 'Glaucous-winged']
            means = [sb_mean, gw_mean]
            errors = [sb_se, gw_se]

            ax.bar(species, means, yerr=errors, capsize=5)
            ax.set_title(f'{stat_name} Comparison')
            ax.grid(alpha=0.3)

            # Add value labels
            for i, (mean, error) in enumerate(zip(means, errors)):
                ax.text(i, mean, f'{mean:.2f}\n±{error:.2f}',
                        ha='center', va='bottom')

    # Remove extra subplot if any
    if len(stat_pairs) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()


def main():
    # Load images
    print("Loading images and segmentation maps...")

    # Load SB images
    sb_images = []
    sb_segs = []
    for img_name in sorted(os.listdir(SLATY_BACKED_IMG_DIR))[:S]:
        img_path = os.path.join(SLATY_BACKED_IMG_DIR, img_name)
        seg_path = os.path.join(SLATY_BACKED_SEG_DIR, img_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is not None and seg is not None:
            sb_images.append(img)
            sb_segs.append(seg)

    # Load GW images
    gw_images = []
    gw_segs = []
    for img_name in sorted(os.listdir(GLAUCOUS_WINGED_IMG_DIR))[:S]:
        img_path = os.path.join(GLAUCOUS_WINGED_IMG_DIR, img_name)
        seg_path = os.path.join(GLAUCOUS_WINGED_SEG_DIR, img_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is not None and seg is not None:
            gw_images.append(img)
            gw_segs.append(seg)

    # Analyze each species
    sb_stats = analyze_single_species(sb_images, sb_segs, "Slaty-backed Gull", REGION_COLORS)
    gw_stats = analyze_single_species(gw_images, gw_segs, "Glaucous-winged Gull", REGION_COLORS)

    # Print comprehensive statistical summary
    print("\n" + "=" * 50)
    print("COMPREHENSIVE STATISTICAL SUMMARY")
    print("=" * 50)

    stats_to_report = [
        ('means', 'Mean Intensity'),
        ('stds', 'Standard Deviation'),
        ('skewness', 'Skewness'),
        ('kurtosis', 'Kurtosis'),
        ('min_values', 'Minimum Value'),
        ('max_values', 'Maximum Value'),
        ('medians', 'Median')
    ]

    for stat_key, stat_name in stats_to_report:
        print(f"\n{stat_name}:")
        print("-" * 30)

        # Slaty-backed Gull statistics
        sb_mean = np.mean(sb_stats[stat_key])
        sb_se = sem(sb_stats[stat_key])
        print(f"Slaty-backed Gull:")
        print(f"  Mean of {stat_name.lower()}: {sb_mean:.2f}")
        print(f"  Standard Error: ±{sb_se:.2f}")
        print(f"  Range: {min(sb_stats[stat_key]):.2f} - {max(sb_stats[stat_key]):.2f}")

        # Glaucous-winged Gull statistics
        gw_mean = np.mean(gw_stats[stat_key])
        gw_se = sem(gw_stats[stat_key])
        print(f"\nGlaucous-winged Gull:")
        print(f"  Mean of {stat_name.lower()}: {gw_mean:.2f}")
        print(f"  Standard Error: ±{gw_se:.2f}")
        print(f"  Range: {min(gw_stats[stat_key]):.2f} - {max(gw_stats[stat_key]):.2f}")

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(sb_stats[stat_key], gw_stats[stat_key])
        print(f"\nStatistical Comparison:")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value: {p_value:.3f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

    # Create comparative visualizations
    plot_comparative_statistics(sb_stats, gw_stats)


if __name__ == "__main__":
    main()