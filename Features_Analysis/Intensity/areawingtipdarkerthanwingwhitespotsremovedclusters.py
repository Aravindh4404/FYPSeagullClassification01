import pandas as pd
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Import configuration file
import sys
from pathlib import Path

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
from Features_Analysis.config import *


def remove_white_spots_kmeans(wingtip_pixels, visualization=False, image_name=""):
    """
    Remove white spots and streaks from wingtip pixels using K-means clustering.

    Args:
        wingtip_pixels: Array of wingtip pixel intensities
        visualization: Whether to create visualization plots
        image_name: Name for saving visualization plots

    Returns:
        filtered_pixels: Wingtip pixels with white spots removed
        white_pixel_mask: Boolean mask indicating which pixels were removed
        cluster_info: Dictionary with clustering information
    """

    if len(wingtip_pixels) == 0:
        return wingtip_pixels, np.array([]), {}

    # Reshape pixels for K-means (needs 2D array)
    pixels_reshaped = wingtip_pixels.reshape(-1, 1)

    # Apply K-means clustering with k=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pixels_reshaped)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_.flatten()

    # Identify which cluster represents the darker pixels (actual wingtip)
    # The cluster with lower mean intensity is the darker one
    dark_cluster_idx = np.argmin(cluster_centers)
    white_cluster_idx = np.argmax(cluster_centers)

    # Create mask for dark pixels (actual wingtip)
    dark_pixel_mask = cluster_labels == dark_cluster_idx
    white_pixel_mask = cluster_labels == white_cluster_idx

    # Filter out white spots/streaks
    filtered_pixels = wingtip_pixels[dark_pixel_mask]

    # Prepare cluster information
    cluster_info = {
        'dark_cluster_center': cluster_centers[dark_cluster_idx],
        'white_cluster_center': cluster_centers[white_cluster_idx],
        'dark_pixel_count': np.sum(dark_pixel_mask),
        'white_pixel_count': np.sum(white_pixel_mask),
        'white_pixel_percentage': (np.sum(white_pixel_mask) / len(wingtip_pixels)) * 100,
        'original_pixel_count': len(wingtip_pixels)
    }

    # Create visualization if requested
    if visualization and image_name:
        create_clustering_visualization(wingtip_pixels, cluster_labels, cluster_centers,
                                        dark_cluster_idx, white_cluster_idx, image_name, cluster_info)

    return filtered_pixels, white_pixel_mask, cluster_info


def create_clustering_visualization(pixels, labels, centers, dark_idx, white_idx, image_name, cluster_info):
    """Create visualization of the clustering results"""

    # Create output directory for visualizations
    viz_dir = "Clustering_Visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Original intensity distribution
    ax1.hist(pixels, bins=50, alpha=0.7, color='gray', edgecolor='black')
    ax1.set_title(f'Original Wingtip Intensity Distribution\n{image_name}')
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.mean(pixels), color='red', linestyle='--', label=f'Mean: {np.mean(pixels):.1f}')
    ax1.legend()

    # 2. Clustered intensity distribution
    dark_pixels = pixels[labels == dark_idx]
    white_pixels = pixels[labels == white_idx]

    ax2.hist(dark_pixels, bins=30, alpha=0.7, color='blue', label=f'Dark Cluster ({len(dark_pixels)} pixels)',
             edgecolor='black')
    ax2.hist(white_pixels, bins=30, alpha=0.7, color='lightcoral', label=f'White Cluster ({len(white_pixels)} pixels)',
             edgecolor='black')
    ax2.axvline(centers[dark_idx], color='darkblue', linestyle='--', linewidth=2,
                label=f'Dark Center: {centers[dark_idx]:.1f}')
    ax2.axvline(centers[white_idx], color='red', linestyle='--', linewidth=2,
                label=f'White Center: {centers[white_idx]:.1f}')
    ax2.set_title('K-means Clustering Results')
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # 3. Comparison of means
    original_mean = np.mean(pixels)
    filtered_mean = np.mean(dark_pixels)

    means = [original_mean, filtered_mean]
    labels_bar = ['Original Mean', 'Filtered Mean\n(White spots removed)']
    colors = ['gray', 'blue']

    bars = ax3.bar(labels_bar, means, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Mean Intensity Comparison')
    ax3.set_ylabel('Mean Intensity')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

    # 4. Clustering statistics
    ax4.axis('off')
    stats_text = f"""
    Clustering Statistics for {image_name}:

    Total Pixels: {cluster_info['original_pixel_count']:,}
    Dark Pixels: {cluster_info['dark_pixel_count']:,} ({100 - cluster_info['white_pixel_percentage']:.1f}%)
    White Pixels Removed: {cluster_info['white_pixel_count']:,} ({cluster_info['white_pixel_percentage']:.1f}%)

    Dark Cluster Center: {cluster_info['dark_cluster_center']:.1f}
    White Cluster Center: {cluster_info['white_cluster_center']:.1f}

    Original Mean: {original_mean:.1f}
    Filtered Mean: {filtered_mean:.1f}
    Difference: {abs(original_mean - filtered_mean):.1f}
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'clustering_analysis_{image_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory


def analyze_wingtip_intensity_distribution_with_filtering(image_path, seg_path, species, file_name, create_viz=False):
    """
    Enhanced version of the original function with white spot removal using K-means clustering.
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Convert entire image to grayscale first
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Apply min-max normalization to the entire grayscale image
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

    # Extract wing and wingtip regions from the normalized grayscale image
    wing_region, wing_mask = extract_region(gray_img, segmentation_img, "wing")
    wingtip_region, wingtip_mask = extract_region(gray_img, segmentation_img, "wingtip")

    # Get wing pixels
    wing_pixels = wing_region[wing_mask > 0]

    if len(wing_pixels) == 0:
        print(f"No wing region found in {file_name}")
        return None

    # Calculate mean wing intensity
    mean_wing_intensity = np.mean(wing_pixels)

    # Get wingtip pixels from the normalized grayscale image
    wingtip_pixels = wingtip_region[wingtip_mask > 0]

    if len(wingtip_pixels) == 0:
        print(f"No wingtip region found in {file_name}")
        return None

    # APPLY K-MEANS CLUSTERING TO REMOVE WHITE SPOTS
    filtered_wingtip_pixels, white_pixel_mask, cluster_info = remove_white_spots_kmeans(
        wingtip_pixels, visualization=create_viz, image_name=file_name.split('.')[0]
    )

    if len(filtered_wingtip_pixels) == 0:
        print(f"No dark wingtip pixels found after clustering in {file_name}")
        return None

    # Define intensity ranges (bins)
    intensity_ranges = [
        (0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
        (50, 60), (60, 70), (70, 80), (80, 90), (90, 100),
        (100, 110), (110, 120), (120, 130), (130, 140), (140, 150),
        (150, 160), (160, 170), (170, 180), (180, 190), (190, 200),
        (200, 210), (210, 220), (220, 230), (230, 240), (240, 255)
    ]

    # Count pixels in each intensity range using FILTERED pixels
    range_counts = {}
    for start, end in intensity_ranges:
        # Count raw number of pixels in this range
        pixel_count = np.sum((filtered_wingtip_pixels >= start) & (filtered_wingtip_pixels < end))
        range_counts[f"intensity_{start}_{end}"] = pixel_count
        # Calculate percentage of wingtip pixels in this range
        range_counts[f"pct_{start}_{end}"] = (pixel_count / len(filtered_wingtip_pixels)) * 100

    # Calculate wing-wingtip differences using FILTERED pixels
    # For each wingtip pixel, calculate how much darker it is than the mean wing
    intensity_diffs = mean_wing_intensity - filtered_wingtip_pixels

    # Only keep positive differences (darker pixels)
    positive_diffs = intensity_diffs[intensity_diffs > 0]

    # Define difference thresholds
    diff_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Count pixels with differences above thresholds
    diff_counts = {}
    for threshold in diff_thresholds:
        pixel_count = np.sum(intensity_diffs > threshold)
        diff_counts[f"diff_gt_{threshold}"] = pixel_count
        diff_counts[f"pct_diff_gt_{threshold}"] = (pixel_count / len(filtered_wingtip_pixels)) * 100

    # Calculate statistics about very dark pixels using FILTERED pixels
    very_dark_counts = {}
    dark_thresholds = [30, 40, 50, 60]
    for threshold in dark_thresholds:
        pixel_count = np.sum(filtered_wingtip_pixels < threshold)
        very_dark_counts[f"dark_lt_{threshold}"] = pixel_count
        very_dark_counts[f"pct_dark_lt_{threshold}"] = (pixel_count / len(filtered_wingtip_pixels)) * 100

    # Prepare results with both original and filtered data
    results = {
        "image_name": file_name,
        "species": species,
        "mean_wing_intensity": mean_wing_intensity,

        # Original wingtip data
        "original_wingtip_intensity": np.mean(wingtip_pixels),
        "original_wingtip_pixel_count": len(wingtip_pixels),

        # Filtered wingtip data (main results)
        "mean_wingtip_intensity": np.mean(filtered_wingtip_pixels),
        "wingtip_pixel_count": len(filtered_wingtip_pixels),

        # Clustering information
        "white_pixels_removed": cluster_info['white_pixel_count'],
        "white_pixel_percentage": cluster_info['white_pixel_percentage'],
        "dark_cluster_center": cluster_info['dark_cluster_center'],
        "white_cluster_center": cluster_info['white_cluster_center'],

        # Wing pixel data
        "wing_pixel_count": len(wing_pixels),
        "darker_pixel_count": len(positive_diffs),
        "pct_darker_pixels": (len(positive_diffs) / len(filtered_wingtip_pixels)) * 100,

        # Intensity distribution (based on filtered pixels)
        **range_counts,
        **diff_counts,
        **very_dark_counts
    }

    return results


def main():
    """
    Process all images for both species and save intensity distribution results with white spot removal.
    """
    results = []

    # Create visualizations for first few images of each species
    viz_count_per_species = 3

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} images...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f" Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")

            # Create visualization for first few images
            create_viz = i < viz_count_per_species

            stats = analyze_wingtip_intensity_distribution_with_filtering(
                img_path, seg_path, species_name, file_name, create_viz=create_viz
            )

            if stats:
                results.append(stats)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)

        results_dir = "Wingtip_Intensity_Distribution_Filtered220"
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "wingtip_intensity_distribution_filtered.csv")
        df.to_csv(csv_path, index=False)

        print(f"\nFiltered results saved to: {csv_path}")

        # Calculate averages by species
        # Select columns that are percentages for easier comparison
        pct_columns = [col for col in df.columns if col.startswith("pct_")]
        dark_columns = [col for col in df.columns if col.startswith("dark_")]
        diff_columns = [col for col in df.columns if col.startswith("diff_")]

        # Include clustering information in averages
        clustering_columns = ['white_pixels_removed', 'white_pixel_percentage',
                              'dark_cluster_center', 'white_cluster_center']

        # Calculate species averages
        species_avg = df.groupby('species')[
            ['mean_wing_intensity', 'original_wingtip_intensity', 'mean_wingtip_intensity'] +
            clustering_columns + pct_columns + dark_columns + diff_columns
            ].mean().reset_index()

        avg_csv_path = os.path.join(results_dir, "wingtip_intensity_averages_filtered.csv")
        species_avg.to_csv(avg_csv_path, index=False)

        print(f"\nSpecies averages saved to: {avg_csv_path}")

        # Print summary comparison
        print("\n=== CLUSTERING SUMMARY ===")
        summary = df.groupby('species')[['original_wingtip_intensity', 'mean_wingtip_intensity',
                                         'white_pixel_percentage']].agg(['mean', 'std'])
        print(summary)


if __name__ == "__main__":
    main()