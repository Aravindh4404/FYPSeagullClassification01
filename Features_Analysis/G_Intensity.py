# Import the image normalization utility (assumed to be in the same directory)
from Features_Analysis.image_normalization import *

S = 5  # Number of images to process per species
OUTPUT_DIR = "Intensity_Results"

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

    # Plot 2 - Removed (Distribution of means with standard error)

    # Plot 3: Box plot of all statistics
    ax = axes[0, 1]
    stats_to_plot = ['means', 'stds', 'medians']
    data_to_plot = [stats[key] for key in stats_to_plot]
    ax.boxplot(data_to_plot, labels=['Means', 'StdDevs', 'Medians'])
    ax.set_ylabel('Value')
    ax.set_title('Distribution of Statistical Measures')
    ax.grid(True, alpha=0.3)

    # Plot 4: Trend of skewness and kurtosis
    ax = axes[1, 0]
    x = range(len(stats['skewness']))
    ax.plot(x, stats['skewness'], 'o-', label='Skewness')
    ax.plot(x, stats['kurtosis'], 's-', label='Kurtosis')
    ax.set_xlabel('Image Number')
    ax.set_ylabel('Value')
    ax.set_title('Skewness and Kurtosis Trends')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 5: Histogram of all means
    ax = axes[1, 1]
    ax.hist(stats['all_region_means'], bins=20, alpha=0.7)
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Region Means')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the figure to the output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{species_name.replace(' ', '_')}_statistics.png"))
    plt.close()


def analyze_single_image(image, seg_map, species_name, region_colors, image_idx, normalize_method="clahe"):
    """Detailed analysis of a single bird image with multiple visualizations."""

    # Create a binary mask for the bird
    bird_mask = create_bird_mask(image)

    # Normalize the image
    normalized_image = normalize_image(image, method=normalize_method, mask=bird_mask)

    # Create a larger figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 3, figure=fig)

    # 1. Original image with mask overlay (larger)
    ax_img = fig.add_subplot(gs[0, :2])
    ax_img.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"{species_name} - Original Image {image_idx + 1}")
    ax_img.axis('off')

    # 2. Normalized image
    ax_norm = fig.add_subplot(gs[0, 2])
    ax_norm.imshow(normalized_image, cmap='gray')
    ax_norm.set_title(f"Normalized Image ({normalize_method})")
    ax_norm.axis('off')

    # Storage for region statistics and CSV data
    all_intensities = []
    region_stats = []
    csv_data = {
        'species': species_name,
        'image_idx': image_idx + 1,
        'image_filename': f"image_{image_idx + 1}"
    }

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

        # Extract pixels from normalized image and compute statistics
        selected_pixels = normalized_image[mask > 0]
        if len(selected_pixels) > 0:
            intensities = selected_pixels
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
                'max': np.max(intensities),
                'pixel_count': len(intensities)
            }
            region_stats.append(stats_dict)

            # Add to CSV data
            csv_data[f'{region_name}_mean'] = stats_dict['mean']
            csv_data[f'{region_name}_std'] = stats_dict['std']
            csv_data[f'{region_name}_median'] = stats_dict['median']
            csv_data[f'{region_name}_skew'] = stats_dict['skew']
            csv_data[f'{region_name}_kurt'] = stats_dict['kurt']
            csv_data[f'{region_name}_min'] = stats_dict['min']
            csv_data[f'{region_name}_max'] = stats_dict['max']
            csv_data[f'{region_name}_pixel_count'] = stats_dict['pixel_count']

    if region_stats:
        # 3. Box plot of intensities
        ax_box = fig.add_subplot(gs[1, 0])
        ax_box.boxplot([stats['mean'] for stats in region_stats],
                       tick_labels=[stats['region'] for stats in region_stats])
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

        # 5. Histogram of all intensities
        ax_hist = fig.add_subplot(gs[1, 2])
        for i, intensities in enumerate(all_intensities):
            ax_hist.hist(intensities, bins=30, alpha=0.7,
                         label=f"{region_stats[i]['region']}")
        ax_hist.set_title("Intensity Distribution")
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        # 6. Statistics table
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        table_data = []
        headers = ['Region', 'Mean', 'Std', 'Median', 'Skewness', 'Kurtosis', 'Min', 'Max', 'Pixels']
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
                f"{stats['max']:.2f}",
                f"{stats['pixel_count']}"
            ]
            table_data.append(row)

        table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        plt.suptitle(f"Detailed Statistical Analysis - {species_name} Image {image_idx + 1}",
                     fontsize=16, y=0.95)
        plt.tight_layout()

        # Save the figure
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{species_name.replace(' ', '_')}_image_{image_idx + 1}.png"))
        plt.close()

        # Print statistics to console
        print(f"\nDetailed Statistics for {species_name} Image {image_idx + 1}:")
        print("-" * 50)
        for stats in region_stats:
            print(f"\n{stats['region']} Region:")
            for key, value in stats.items():
                if key != 'region':
                    print(f"{key.capitalize()}: {value:.2f}")

        # Return statistics for overall analysis and the CSV data
        return {
            'means': np.mean([s['mean'] for s in region_stats]),
            'stds': np.mean([s['std'] for s in region_stats]),
            'skewness': np.mean([s['skew'] for s in region_stats]),
            'kurtosis': np.mean([s['kurt'] for s in region_stats]),
            'min_values': np.mean([s['min'] for s in region_stats]),
            'max_values': np.mean([s['max'] for s in region_stats]),
            'medians': np.mean([s['median'] for s in region_stats]),
            'all_region_means': [s['mean'] for s in region_stats]
        }, csv_data

    return None, None


def analyze_single_species(images, seg_maps, species_name, region_colors, normalize_method="clahe"):
    """Analyze all images for a single species and save data to CSV."""
    all_stats = {
        'means': [],
        'stds': [],
        'skewness': [],
        'kurtosis': [],
        'min_values': [],
        'max_values': [],
        'medians': [],
        'all_region_means': []
    }

    csv_rows = []

    print(f"\n{'=' * 50}")
    print(f"Analysis for {species_name}")
    print(f"{'=' * 50}")

    for idx, (image, seg_map) in enumerate(zip(images, seg_maps)):
        stats, csv_data = analyze_single_image(image, seg_map, species_name, region_colors, idx, normalize_method)
        if stats:
            for key in all_stats:
                if key == 'all_region_means':
                    all_stats[key].extend(stats[key])
                else:
                    all_stats[key].append(stats[key])

            # Add CSV data to rows
            csv_rows.append(csv_data)

    # Save CSV data
    if csv_rows:
        # Ensure output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Define CSV path
        csv_path = os.path.join(OUTPUT_DIR, f"{species_name.replace(' ', '_')}_analysis.csv")

        # Write to CSV
        with open(csv_path, 'w', newline='') as csvfile:
            # Get all field names from the first row
            fieldnames = csv_rows[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

        print(f"CSV data saved to: {csv_path}")

    # Create species-level statistical plots
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

    # Save figure
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, "species_comparison.png"))
    plt.close()


def save_comparison_csv(sb_stats, gw_stats):
    """Save a CSV file with comparative statistics between species."""
    # Define statistics to compare
    stats_to_compare = [
        ('means', 'Mean Intensity'),
        ('stds', 'Standard Deviation'),
        ('skewness', 'Skewness'),
        ('kurtosis', 'Kurtosis'),
        ('min_values', 'Minimum Value'),
        ('max_values', 'Maximum Value'),
        ('medians', 'Median')
    ]

    # Prepare data for CSV
    rows = []

    for stat_key, stat_name in stats_to_compare:
        # Calculate statistics for each species
        sb_mean = np.mean(sb_stats[stat_key])
        sb_se = sem(sb_stats[stat_key])
        gw_mean = np.mean(gw_stats[stat_key])
        gw_se = sem(gw_stats[stat_key])

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(sb_stats[stat_key], gw_stats[stat_key])

        # Create a row for the CSV
        row = {
            'Statistic': stat_name,
            'Slaty_backed_Mean': sb_mean,
            'Slaty_backed_SE': sb_se,
            'Glaucous_winged_Mean': gw_mean,
            'Glaucous_winged_SE': gw_se,
            'T_statistic': t_stat,
            'P_value': p_value,
            'Significant_difference': 'Yes' if p_value < 0.05 else 'No'
        }
        rows.append(row)

    # Save to CSV
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    csv_path = os.path.join(OUTPUT_DIR, "species_comparison.csv")

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Comparison CSV saved to: {csv_path}")


def main(normalize_method="clahe"):
    # Load images
    print("Loading images and segmentation maps...")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

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
    sb_stats = analyze_single_species(sb_images, sb_segs, "Slaty-backed Gull", REGION_COLORS, normalize_method)
    gw_stats = analyze_single_species(gw_images, gw_segs, "Glaucous-winged Gull", REGION_COLORS, normalize_method)

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