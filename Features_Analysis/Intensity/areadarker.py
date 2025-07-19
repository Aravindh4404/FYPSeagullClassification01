import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
from Features_Analysis.config import *

# Define consistent color scheme for species
SPECIES_COLORS = {
    'Glaucous_Winged_Gull': '#3274A1',  # Blue
    'Slaty_Backed_Gull': '#E1812C'  # Orange
}

# Set plotting style for professional-looking visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Create output directory
output_dir = "Wingtip_Dark_Pixel_Analysis"
os.makedirs(output_dir, exist_ok=True)


def analyze_wingtip_dark_pixels(image_path, seg_path, species, file_name, mean_wing_intensity):
    """
    Analyzes wingtip pixels that are darker than the mean wing intensity
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None, None

    # Convert entire image to grayscale first
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Apply min-max normalization to the entire grayscale image
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

    # Extract wingtip region from the normalized grayscale image
    wingtip_region, wingtip_mask = extract_region(gray_img, segmentation_img, "wingtip")

    # Get wingtip pixels
    wingtip_pixels = wingtip_region[wingtip_mask > 0]

    if len(wingtip_pixels) == 0:
        print(f"No wingtip region found in {file_name}")
        return None, None

    # Find pixels darker than mean wing intensity
    darker_pixels_mask = wingtip_pixels < mean_wing_intensity
    darker_pixel_count = np.sum(darker_pixels_mask)

    # Calculate statistics
    total_wingtip_pixels = len(wingtip_pixels)
    darker_pixel_percentage = (darker_pixel_count / total_wingtip_pixels) * 100

    # Create overlay mask for the entire image
    full_image_mask = np.zeros_like(gray_img, dtype=bool)
    full_image_mask[wingtip_mask > 0] = wingtip_pixels < mean_wing_intensity

    # Prepare results
    results = {
        'image_name': file_name,
        'species': species,
        'mean_wing_intensity': mean_wing_intensity,
        'mean_wingtip_intensity': np.mean(wingtip_pixels),
        'total_wingtip_pixels': total_wingtip_pixels,
        'darker_pixel_count': darker_pixel_count,
        'darker_pixel_percentage': darker_pixel_percentage
    }

    return results, full_image_mask


def create_sample_overlay_images(wing_data, dark_pixel_results, n_samples=6):
    """
    Create overlay images showing darker pixels for sample images
    """
    # Select sample images (3 from each species, varying darkness percentages)
    sample_images = []

    for species in wing_data['species'].unique():
        species_data = dark_pixel_results[dark_pixel_results['species'] == species]

        # Sort by darker pixel percentage and select diverse samples
        species_sorted = species_data.sort_values('darker_pixel_percentage')

        # Select low, medium, high darkness examples
        n_per_species = n_samples // 2
        if len(species_sorted) >= n_per_species:
            indices = np.linspace(0, len(species_sorted) - 1, n_per_species, dtype=int)
        else:
            indices = range(len(species_sorted))

        for idx in indices:
            sample_images.append(species_sorted.iloc[idx])

    # Create overlay visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sample Wingtip Dark Pixel Overlays\n(Red = Pixels Darker than Mean Wing Intensity)',
                 fontsize=16, fontweight='bold', y=0.95)

    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]

        if i < len(sample_images):
            sample = sample_images[i]

            # Find the image paths
            species_name = sample['species']
            image_name = sample['image_name']

            # Get image paths
            image_paths = get_image_paths(species_name)
            img_path = None
            seg_path = None

            for img_p, seg_p in image_paths:
                if os.path.basename(img_p) == image_name:
                    img_path = img_p
                    seg_path = seg_p
                    break

            if img_path is None:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                ax.set_title(f'{image_name} - Not Found')
                ax.axis('off')
                continue

            # Load and process image
            original_img = cv2.imread(img_path)
            segmentation_img = cv2.imread(seg_path)

            if original_img is None or segmentation_img is None:
                ax.text(0.5, 0.5, 'Error loading image', ha='center', va='center')
                ax.axis('off')
                continue

            # Convert to RGB for matplotlib
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Get the dark pixel mask
            _, dark_mask = analyze_wingtip_dark_pixels(
                img_path, seg_path, species_name, image_name, sample['mean_wing_intensity']
            )

            if dark_mask is None:
                ax.text(0.5, 0.5, 'Error processing mask', ha='center', va='center')
                ax.axis('off')
                continue

            # Create overlay
            overlay = original_rgb.copy()
            overlay[dark_mask] = [255, 0, 0]  # Red for darker pixels

            # Blend with original
            alpha = 0.6
            blended = cv2.addWeighted(original_rgb, 1 - alpha, overlay, alpha, 0)

            # Display
            ax.imshow(blended)
            display_name = species_name.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')
            ax.set_title(f'{display_name}\n'
                         f'{sample["darker_pixel_percentage"]:.1f}% Dark Pixels\n'
                         f'Wing Mean: {sample["mean_wing_intensity"]:.1f}',
                         fontsize=10, color=SPECIES_COLORS[species_name])
        else:
            ax.text(0.5, 0.5, 'No sample available', ha='center', va='center')
            ax.set_title('No Sample')

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_wingtip_dark_pixel_overlays.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def create_dark_pixel_analysis_plots(dark_pixel_results):
    """
    Create analysis plots for dark pixel percentages
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle('Wingtip Dark Pixel Analysis\n(Percentage of Pixels Darker than Mean Wing Intensity)',
                 fontsize=16, fontweight='bold', y=0.95)

    # 1. Dark Pixel Percentage Distribution (Histogram)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Distribution of Dark Pixel Percentages', fontsize=12, fontweight='bold')

    for species in dark_pixel_results['species'].unique():
        species_data = dark_pixel_results[dark_pixel_results['species'] == species]
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        ax1.hist(species_data['darker_pixel_percentage'], bins=15, alpha=0.7,
                 color=color, label=display_name, density=True)

    ax1.set_xlabel('Dark Pixel Percentage (%)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box Plot Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Dark Pixel Percentage by Species', fontsize=12, fontweight='bold')

    species_data_list = []
    species_labels = []
    colors_list = []

    for species in dark_pixel_results['species'].unique():
        species_data = dark_pixel_results[dark_pixel_results['species'] == species]
        species_data_list.append(species_data['darker_pixel_percentage'])
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')
        species_labels.append(display_name)
        colors_list.append(SPECIES_COLORS[species])

    # Create box plot
    bp = ax2.boxplot(species_data_list, labels=species_labels, patch_artist=True)

    # Color the box plots
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Dark Pixel Percentage (%)')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # 3. Mean Comparison with Error Bars
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('Mean Dark Pixel Percentage Comparison', fontsize=12, fontweight='bold')

    species_names = []
    means = []
    stds = []
    colors = []

    for species in dark_pixel_results['species'].unique():
        species_data = dark_pixel_results[dark_pixel_results['species'] == species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        species_names.append(display_name)
        means.append(species_data['darker_pixel_percentage'].mean())
        stds.append(species_data['darker_pixel_percentage'].std())
        colors.append(SPECIES_COLORS[species])

    bars = ax3.bar(species_names, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax3.set_ylabel('Dark Pixel Percentage (%)')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, means, stds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + std_val + 1,
                 f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. Statistics Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # Calculate statistics
    stats_text = "DARK PIXEL STATISTICS:\n\n"
    for species in dark_pixel_results['species'].unique():
        species_data = dark_pixel_results[dark_pixel_results['species'] == species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        mean_dark_pct = species_data['darker_pixel_percentage'].mean()
        std_dark_pct = species_data['darker_pixel_percentage'].std()
        median_dark_pct = species_data['darker_pixel_percentage'].median()
        min_dark_pct = species_data['darker_pixel_percentage'].min()
        max_dark_pct = species_data['darker_pixel_percentage'].max()

        stats_text += f"{display_name}:\n"
        stats_text += f"  Mean:   {mean_dark_pct:.1f}%\n"
        stats_text += f"  Std:    {std_dark_pct:.1f}%\n"
        stats_text += f"  Median: {median_dark_pct:.1f}%\n"
        stats_text += f"  Range:  {min_dark_pct:.1f}% - {max_dark_pct:.1f}%\n"
        stats_text += f"  Count:  {len(species_data)} images\n\n"

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wingtip_dark_pixel_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to perform wingtip dark pixel analysis
    """
    print("Loading wing intensity data...")

    # Load wing intensity data
    try:
        wing_data = pd.read_csv('Intensity_Results/wing_intensity_analysis.csv')
        print(f"Successfully loaded wing data with {len(wing_data)} images")
    except FileNotFoundError:
        print("Error: wing_intensity_analysis.csv not found!")
        print("Please run the wing intensity analysis script first.")
        return

    print("Analyzing dark pixels in wingtip regions...")

    # Process each image to find dark pixels
    all_results = []

    for _, row in wing_data.iterrows():
        species_name = row['species']
        image_name = row['image_name']
        mean_wing_intensity = row['mean_intensity']

        print(f"Processing {image_name}...")

        # Find the image paths
        image_paths = get_image_paths(species_name)
        img_path = None
        seg_path = None

        for img_p, seg_p in image_paths:
            if os.path.basename(img_p) == image_name:
                img_path = img_p
                seg_path = seg_p
                break

        if img_path is None:
            print(f"Warning: Could not find image path for {image_name}")
            continue

        # Analyze dark pixels
        result, mask = analyze_wingtip_dark_pixels(
            img_path, seg_path, species_name, image_name, mean_wing_intensity
        )

        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No results generated. Check if wingtip regions were detected.")
        return

    # Convert to DataFrame
    dark_pixel_df = pd.DataFrame(all_results)

    # Save results
    csv_path = os.path.join(output_dir, "wingtip_dark_pixel_analysis.csv")
    dark_pixel_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Calculate and save species averages
    species_averages = dark_pixel_df.groupby('species').agg({
        'darker_pixel_percentage': ['mean', 'std', 'median', 'min', 'max'],
        'mean_wing_intensity': ['mean', 'std'],
        'mean_wingtip_intensity': ['mean', 'std'],
        'total_wingtip_pixels': ['mean', 'std'],
        'darker_pixel_count': ['mean', 'std']
    }).round(2)

    avg_csv_path = os.path.join(output_dir, "wingtip_dark_pixel_averages.csv")
    species_averages.to_csv(avg_csv_path)
    print(f"Species averages saved to: {avg_csv_path}")

    # Create visualizations
    print("\nCreating analysis plots...")
    create_dark_pixel_analysis_plots(dark_pixel_df)

    print("\nCreating sample overlay images...")
    create_sample_overlay_images(wing_data, dark_pixel_df)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("WINGTIP DARK PIXEL ANALYSIS SUMMARY")
    print("=" * 60)

    for species in dark_pixel_df['species'].unique():
        species_data = dark_pixel_df[dark_pixel_df['species'] == species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        print(f"\n{display_name}:")
        print(f"  Images analyzed:           {len(species_data)}")
        print(
            f"  Dark pixel percentage:     {species_data['darker_pixel_percentage'].mean():.1f}% ± {species_data['darker_pixel_percentage'].std():.1f}%")
        print(
            f"  Range:                     {species_data['darker_pixel_percentage'].min():.1f}% - {species_data['darker_pixel_percentage'].max():.1f}%")
        print(f"  Median:                    {species_data['darker_pixel_percentage'].median():.1f}%")

    print("\n" + "=" * 60)
    print("Analysis complete! All results and visualizations saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()