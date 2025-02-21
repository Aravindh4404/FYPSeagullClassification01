
from Features_Analysis.utils import *
from Features_Analysis.config import *

import os
import cv2
import numpy as np
from scipy.stats import skew, kurtosis, ks_2samp, sem
import matplotlib.pyplot as plt
from scipy import stats


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


def analyze_single_species(images, seg_maps, species_name, region_colors):
    """Analyze a single species and return detailed statistics."""
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
    print(f"Detailed Analysis for {species_name}")
    print(f"{'=' * 50}")

    for idx, (image, seg_map) in enumerate(zip(images, seg_maps)):
        print(f"\nProcessing Image {idx + 1}:")
        print("-" * 30)

        # Create figure for this image
        plt.figure(figsize=(15, 5))

        # Original image with mask overlay
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"{species_name} - Original Image {idx + 1}")
        plt.axis('off')

        # Extract region pixels and create histogram
        image_stats = []
        for region_name, bgr_color in region_colors.items():
            # Create mask for this region
            tolerance = 10
            lower = np.array([max(c - tolerance, 0) for c in bgr_color], dtype=np.uint8)
            upper = np.array([min(c + tolerance, 255) for c in bgr_color], dtype=np.uint8)
            mask = cv2.inRange(seg_map, lower, upper)

            # Overlay contour on original image
            if np.sum(mask) > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                plt.contour(mask, levels=[0.5], colors='red', linewidths=1)

            # Extract pixels and compute statistics
            selected_pixels = image[mask > 0]
            if len(selected_pixels) > 0:
                intensities = selected_pixels.mean(axis=1) if len(selected_pixels.shape) > 1 else selected_pixels

                # Calculate statistics for this region
                stats_dict = {
                    'mean': np.mean(intensities),
                    'std': np.std(intensities),
                    'skew': skew(intensities),
                    'kurt': kurtosis(intensities),
                    'min': np.min(intensities),
                    'max': np.max(intensities),
                    'median': np.median(intensities)
                }

                image_stats.append(stats_dict)

                # Print region statistics
                print(f"\n{region_name} Region Statistics:")
                for stat_name, stat_value in stats_dict.items():
                    print(f"{stat_name.capitalize()}: {stat_value:.2f}")

                # Plot histogram
                plt.subplot(1, 2, 2)
                plt.hist(intensities, bins=30, alpha=0.7, label=region_name)
                plt.title(f"Intensity Distribution\nMean: {stats_dict['mean']:.2f}, Std: {stats_dict['std']:.2f}")
                plt.xlabel("Intensity")
                plt.ylabel("Frequency")
                plt.legend()

        if image_stats:
            # Aggregate statistics for this image
            all_stats['means'].append(np.mean([s['mean'] for s in image_stats]))
            all_stats['stds'].append(np.mean([s['std'] for s in image_stats]))
            all_stats['skewness'].append(np.mean([s['skew'] for s in image_stats]))
            all_stats['kurtosis'].append(np.mean([s['kurt'] for s in image_stats]))
            all_stats['min_values'].append(np.mean([s['min'] for s in image_stats]))
            all_stats['max_values'].append(np.mean([s['max'] for s in image_stats]))
            all_stats['medians'].append(np.mean([s['median'] for s in image_stats]))

        plt.tight_layout()
        plt.show()

    # Plot detailed statistics for this species
    plot_species_statistics(all_stats, species_name)

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