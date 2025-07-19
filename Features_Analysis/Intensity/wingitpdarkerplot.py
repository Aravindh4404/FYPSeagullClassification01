import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
from Features_Analysis.config import *


def load_darkness_analysis_data():
    """
    Load the darkness analysis results from CSV file.
    """
    csv_path = "Wingtip_Darkness_Analysis/wingtip_darkness_analysis.csv"

    if not os.path.exists(csv_path):
        print(f"Darkness analysis data not found at {csv_path}")
        print("Please run 'wingtip_darkness_analyzer.py' first to generate the data.")
        return None

    df = pd.read_csv(csv_path)
    print(f"Loaded darkness analysis data for {len(df)} images")
    return df


def create_darkness_overlay(original_img, gray_img, wingtip_mask, mean_wing_intensity, species, file_name,
                            darkness_percentage):
    """
    Create overlay visualization showing darker regions in the wingtip.

    Args:
        original_img: Original color image
        gray_img: Grayscale normalized image
        wingtip_mask: Mask for wingtip region
        mean_wing_intensity: Mean wing intensity threshold
        species: Species name
        file_name: Image file name
        darkness_percentage: Calculated darkness percentage
    """
    # Create output directory
    overlay_dir = "Darkness_Overlay_Visualizations"
    os.makedirs(overlay_dir, exist_ok=True)

    # Create a copy of the original image for overlay
    overlay_img = original_img.copy()

    # Create mask for darker pixels within wingtip
    darker_mask = np.zeros_like(gray_img, dtype=np.uint8)

    # Find pixels in wingtip that are darker than mean wing intensity
    wingtip_coords = np.where(wingtip_mask > 0)
    for y, x in zip(wingtip_coords[0], wingtip_coords[1]):
        if gray_img[y, x] < mean_wing_intensity:
            darker_mask[y, x] = 255

    # Create different overlays
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Darkness Analysis: {species} - {file_name}\nDarkness Percentage: {darkness_percentage:.1f}%',
                 fontsize=16, fontweight='bold')

    # 1. Original image
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 2. Grayscale with wingtip outlined
    axes[0, 1].imshow(gray_img, cmap='gray')
    # Create wingtip contour
    contours, _ = cv2.findContours(wingtip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Convert contour points for matplotlib
        contour_points = contour.reshape(-1, 2)
        axes[0, 1].plot(contour_points[:, 0], contour_points[:, 1], 'cyan', linewidth=2)
    axes[0, 1].set_title('Grayscale with Wingtip Boundary')
    axes[0, 1].axis('off')

    # 3. Darker regions highlighted
    axes[0, 2].imshow(gray_img, cmap='gray')
    axes[0, 2].imshow(darker_mask, cmap='Reds', alpha=0.6)
    axes[0, 2].set_title('Darker Regions Highlighted (Red)')
    axes[0, 2].axis('off')

    # 4. Original with red overlay for darker regions
    overlay_colored = original_img.copy()
    # Create red overlay for darker pixels
    red_overlay = np.zeros_like(original_img)
    red_overlay[darker_mask > 0] = [0, 0, 255]  # Red in BGR

    # Blend the images
    overlay_colored = cv2.addWeighted(overlay_colored, 0.7, red_overlay, 0.3, 0)

    axes[1, 0].imshow(cv2.cvtColor(overlay_colored, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Original + Darker Regions (Red Overlay)')
    axes[1, 0].axis('off')

    # 5. Intensity analysis within wingtip
    wingtip_pixels = gray_img[wingtip_mask > 0]
    axes[1, 1].hist(wingtip_pixels, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(mean_wing_intensity, color='red', linestyle='--', linewidth=2,
                       label=f'Wing Mean: {mean_wing_intensity:.1f}')
    axes[1, 1].axvline(np.mean(wingtip_pixels), color='blue', linestyle='--', linewidth=2,
                       label=f'Wingtip Mean: {np.mean(wingtip_pixels):.1f}')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Wingtip Intensity Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Side-by-side comparison with statistics
    axes[1, 2].axis('off')
    stats_text = f"""
    ANALYSIS STATISTICS:

    Species: {species}
    Image: {file_name}

    Wing Mean Intensity: {mean_wing_intensity:.1f}
    Wingtip Mean Intensity: {np.mean(wingtip_pixels):.1f}

    Total Wingtip Pixels: {len(wingtip_pixels):,}
    Darker Pixels: {np.sum(darker_mask > 0):,}
    Darkness Percentage: {darkness_percentage:.1f}%

    Intensity Difference: {mean_wing_intensity - np.mean(wingtip_pixels):.1f}

    Min Wingtip Intensity: {np.min(wingtip_pixels):.0f}
    Max Wingtip Intensity: {np.max(wingtip_pixels):.0f}
    Std Wingtip Intensity: {np.std(wingtip_pixels):.1f}
    """

    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save the overlay visualization
    clean_filename = file_name.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
    overlay_path = os.path.join(overlay_dir, f'darkness_overlay_{species}_{clean_filename}.png')
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory

    print(f"    Overlay saved: {overlay_path}")


def create_overlays_for_selected_images(results_df, max_overlays_per_species=5):
    """
    Create overlay visualizations for selected images from the analysis results.
    """
    print("Creating overlay visualizations...")

    for species in results_df['species'].unique():
        species_data = results_df[results_df['species'] == species]

        # Select images for overlay creation (first few)
        selected_data = species_data.head(max_overlays_per_species)

        print(f"\nCreating overlays for {species} ({len(selected_data)} images)...")

        for _, row in selected_data.iterrows():
            # Find the image paths
            image_paths = get_image_paths(species)

            # Find matching image
            img_path = None
            seg_path = None
            for img_p, seg_p in image_paths:
                if os.path.basename(img_p) == row['image_name']:
                    img_path = img_p
                    seg_path = seg_p
                    break

            if img_path and seg_path:
                # Load images
                original_img = cv2.imread(img_path)
                segmentation_img = cv2.imread(seg_path)

                if original_img is not None and segmentation_img is not None:
                    # Convert to grayscale and normalize
                    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

                    # Extract wingtip region
                    wingtip_region, wingtip_mask = extract_region(gray_img, segmentation_img, "wingtip")

                    if wingtip_mask is not None and np.any(wingtip_mask > 0):
                        create_darkness_overlay(
                            original_img, gray_img, wingtip_mask,
                            row['mean_wing_intensity'], species, row['image_name'],
                            row['darkness_percentage']
                        )
                    else:
                        print(f"    Warning: No wingtip mask found for {row['image_name']}")
                else:
                    print(f"    Warning: Could not load images for {row['image_name']}")
            else:
                print(f"    Warning: Could not find image files for {row['image_name']}")


def create_combined_overlay_comparison(results_df, max_images_per_species=3):
    """
    Create a combined comparison showing overlays for multiple images from each species.
    """
    overlay_dir = "Darkness_Overlay_Visualizations"
    os.makedirs(overlay_dir, exist_ok=True)

    species_list = results_df['species'].unique()

    # Select representative images (highest, medium, lowest darkness percentages)
    for species in species_list:
        species_data = results_df[results_df['species'] == species].copy()
        species_data = species_data.sort_values('darkness_percentage')

        n_images = min(len(species_data), max_images_per_species)

        if n_images >= 3:
            # Select high, medium, low
            indices = [0, len(species_data) // 2, len(species_data) - 1]
            selected = species_data.iloc[indices]
        else:
            # Select available images
            selected = species_data.iloc[:n_images]

        # Create comparison figure
        fig, axes = plt.subplots(len(selected), 4, figsize=(20, 5 * len(selected)))
        if len(selected) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'{species} - Darkness Analysis Comparison', fontsize=16, fontweight='bold')

        for i, (_, row) in enumerate(selected.iterrows()):
            # This would require reloading images - simplified version for display
            axes[i, 0].text(0.5, 0.5, f"Original\n{row['image_name']}\n{row['darkness_percentage']:.1f}% dark",
                            ha='center', va='center', transform=axes[i, 0].transAxes,
                            bbox=dict(boxstyle='round', facecolor='lightblue'))
            axes[i, 0].set_title(f"Darkness: {row['darkness_percentage']:.1f}%")
            axes[i, 0].axis('off')

            axes[i, 1].text(0.5, 0.5, f"Grayscale\nWing Mean: {row['mean_wing_intensity']:.1f}",
                            ha='center', va='center', transform=axes[i, 1].transAxes,
                            bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[i, 1].set_title("Grayscale + Boundary")
            axes[i, 1].axis('off')

            axes[i, 2].text(0.5, 0.5, f"Darker Regions\n{row['darker_pixel_count']} pixels",
                            ha='center', va='center', transform=axes[i, 2].transAxes,
                            bbox=dict(boxstyle='round', facecolor='lightcoral'))
            axes[i, 2].set_title("Highlighted Darker Areas")
            axes[i, 2].axis('off')

            # Statistics
            stats = f"""
            Wing Mean: {row['mean_wing_intensity']:.1f}
            Wingtip Mean: {row['mean_wingtip_intensity']:.1f}
            Difference: {row['wing_wingtip_diff']:.1f}
            Dark Pixels: {row['darker_pixel_count']:,}
            Total Pixels: {row['total_wingtip_pixels']:,}
            Darkness %: {row['darkness_percentage']:.1f}%
            """
            axes[i, 3].text(0.05, 0.95, stats, transform=axes[i, 3].transAxes,
                            fontsize=9, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightyellow'))
            axes[i, 3].set_title("Statistics")
            axes[i, 3].axis('off')

        plt.tight_layout()
        comparison_path = os.path.join(overlay_dir, f'darkness_comparison_{species}.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Species comparison saved: {comparison_path}")


def create_darkness_plots(results_df):
    """
    Create visualization plots for the darkness analysis results.
    """
    # Create output directory
    plot_dir = "Darkness_Analysis_Plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1. Box plot comparing darkness percentages between species
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    sns.boxplot(data=results_df, x='species', y='darkness_percentage')
    plt.title('Distribution of Wingtip Darkness Percentage by Species')
    plt.ylabel('Darkness Percentage (%)')
    plt.xticks(rotation=45)

    # 2. Scatter plot: Wing intensity vs Darkness percentage
    plt.subplot(2, 2, 2)
    for species in results_df['species'].unique():
        species_data = results_df[results_df['species'] == species]
        plt.scatter(species_data['mean_wing_intensity'], species_data['darkness_percentage'],
                    alpha=0.7, label=species, s=50)
    plt.xlabel('Mean Wing Intensity')
    plt.ylabel('Darkness Percentage (%)')
    plt.title('Wing Intensity vs Wingtip Darkness')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Histogram of darkness percentages
    plt.subplot(2, 2, 3)
    for species in results_df['species'].unique():
        species_data = results_df[results_df['species'] == species]
        plt.hist(species_data['darkness_percentage'], alpha=0.7, label=species, bins=20)
    plt.xlabel('Darkness Percentage (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Darkness Percentages')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Wing-Wingtip intensity difference
    plt.subplot(2, 2, 4)
    sns.boxplot(data=results_df, x='species', y='wing_wingtip_diff')
    plt.title('Wing-Wingtip Intensity Difference by Species')
    plt.ylabel('Intensity Difference')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'darkness_analysis_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Additional detailed plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Individual species darkness distributions
    species_list = results_df['species'].unique()
    colors = ['skyblue', 'lightcoral']

    for i, species in enumerate(species_list):
        species_data = results_df[results_df['species'] == species]

        # Darkness percentage distribution
        axes[0, i].hist(species_data['darkness_percentage'], bins=15,
                        alpha=0.7, color=colors[i], edgecolor='black')
        axes[0, i].set_title(f'{species}\nDarkness Percentage Distribution')
        axes[0, i].set_xlabel('Darkness Percentage (%)')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].axvline(species_data['darkness_percentage'].mean(),
                           color='red', linestyle='--',
                           label=f'Mean: {species_data["darkness_percentage"].mean():.1f}%')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

        # Intensity scatter for each species
        axes[1, i].scatter(species_data['mean_wing_intensity'],
                           species_data['mean_wingtip_intensity'],
                           alpha=0.7, color=colors[i], s=50)
        axes[1, i].set_xlabel('Mean Wing Intensity')
        axes[1, i].set_ylabel('Mean Wingtip Intensity')
        axes[1, i].set_title(f'{species}\nWing vs Wingtip Intensity')

        # Add diagonal line (equal intensities)
        min_val = min(species_data['mean_wing_intensity'].min(),
                      species_data['mean_wingtip_intensity'].min())
        max_val = max(species_data['mean_wing_intensity'].max(),
                      species_data['mean_wingtip_intensity'].max())
        axes[1, i].plot([min_val, max_val], [min_val, max_val],
                        'r--', alpha=0.5, label='Equal intensity line')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'species_specific_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to create all visualizations from the darkness analysis data.
    """
    # Load the analysis results
    results_df = load_darkness_analysis_data()

    if results_df is None:
        return

    print(f"\nLoaded data for {len(results_df)} images across {results_df['species'].nunique()} species")

    # Create statistical plots
    print("\nCreating statistical plots...")
    create_darkness_plots(results_df)
    print("Statistical plots saved to Darkness_Analysis_Plots directory")

    # Create overlay visualizations for selected images
    create_overlays_for_selected_images(results_df, max_overlays_per_species=5)

    # Create combined overlay comparisons
    print("\nCreating species comparison overlays...")
    create_combined_overlay_comparison(results_df)
    print("Overlay comparisons saved to Darkness_Overlay_Visualizations directory")

    print("\n" + "=" * 50)
    print("VISUALIZATION COMPLETE!")
    print("=" * 50)
    print("Check the following directories for outputs:")
    print("- Darkness_Analysis_Plots/ (statistical plots)")
    print("- Darkness_Overlay_Visualizations/ (image overlays and comparisons)")


if __name__ == "__main__":
    main()