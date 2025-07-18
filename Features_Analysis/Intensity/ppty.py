import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import sys
from pathlib import Path

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
from Features_Analysis.config import *


def create_pixel_overlay_validation(image_path, seg_path, wingtip_pixels, wingtip_mask,
                                    cluster_labels, dark_cluster_idx, white_cluster_idx,
                                    image_name="", save_results=True):
    """
    Create overlay visualization showing which pixels are classified as dark vs white spots

    Args:
        image_path: Path to original image
        seg_path: Path to segmentation image
        wingtip_pixels: Array of wingtip pixel intensities
        wingtip_mask: Boolean mask for wingtip region
        cluster_labels: K-means cluster labels for each pixel
        dark_cluster_idx: Index of the dark cluster
        white_cluster_idx: Index of the white cluster
        image_name: Name for saving files
        save_results: Whether to save overlay images

    Returns:
        Dictionary with overlay images and statistics
    """

    # Load original images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Convert to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Create overlay image (copy of original)
    overlay_img = original_rgb.copy()

    # Get wingtip coordinates
    wingtip_coords = np.where(wingtip_mask > 0)

    # Create color-coded overlay
    dark_pixel_coords = []
    white_pixel_coords = []

    for i, (y, x) in enumerate(zip(wingtip_coords[0], wingtip_coords[1])):
        if cluster_labels[i] == dark_cluster_idx:
            # Color dark pixels blue
            cv2.circle(overlay_img, (x, y), 1, (0, 0, 255), -1)  # Blue for dark pixels
            dark_pixel_coords.append((y, x))
        else:
            # Color white spots red
            cv2.circle(overlay_img, (x, y), 1, (255, 0, 0), -1)  # Red for white spots
            white_pixel_coords.append((y, x))

    # Create separate images for dark and white pixels
    dark_only_img = original_rgb.copy()
    white_only_img = original_rgb.copy()

    # Highlight only dark pixels
    for y, x in dark_pixel_coords:
        cv2.circle(dark_only_img, (x, y), 1, (0, 255, 0), -1)  # Green for dark pixels

    # Highlight only white pixels
    for y, x in white_pixel_coords:
        cv2.circle(white_only_img, (x, y), 1, (255, 0, 0), -1)  # Red for white pixels

    # Create intensity-based visualization
    intensity_overlay = original_rgb.copy()

    # Normalize wingtip pixels to 0-255 range for color mapping
    min_intensity = np.min(wingtip_pixels)
    max_intensity = np.max(wingtip_pixels)

    for i, (y, x) in enumerate(zip(wingtip_coords[0], wingtip_coords[1])):
        # Map intensity to color (blue = dark, red = bright)
        intensity = wingtip_pixels[i]
        normalized_intensity = (intensity - min_intensity) / (max_intensity - min_intensity)

        # Create color based on intensity
        blue_component = int(255 * (1 - normalized_intensity))
        red_component = int(255 * normalized_intensity)

        cv2.circle(intensity_overlay, (x, y), 1, (red_component, 0, blue_component), -1)

    # Calculate statistics
    dark_pixels = wingtip_pixels[cluster_labels == dark_cluster_idx]
    white_pixels = wingtip_pixels[cluster_labels == white_cluster_idx]

    stats = {
        'total_pixels': len(wingtip_pixels),
        'dark_pixels': len(dark_pixels),
        'white_pixels': len(white_pixels),
        'dark_percentage': (len(dark_pixels) / len(wingtip_pixels)) * 100,
        'white_percentage': (len(white_pixels) / len(wingtip_pixels)) * 100,
        'dark_mean_intensity': np.mean(dark_pixels),
        'white_mean_intensity': np.mean(white_pixels),
        'original_mean': np.mean(wingtip_pixels),
        'filtered_mean': np.mean(dark_pixels)
    }

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(f'Pixel Clustering Validation: {image_name}', fontsize=16)

    # 1. Original image with wingtip region outlined
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image with Wingtip Region')

    # Outline wingtip region
    contours, _ = cv2.findContours(wingtip_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        axes[0, 0].plot(contour[:, 0, 0], contour[:, 0, 1], 'yellow', linewidth=2)

    axes[0, 0].axis('off')

    # 2. Overlay with both clusters
    axes[0, 1].imshow(overlay_img)
    axes[0, 1].set_title('Clustering Overlay\n(Blue=Dark, Red=White Spots)')
    axes[0, 1].axis('off')

    # 3. Dark pixels only
    axes[0, 2].imshow(dark_only_img)
    axes[0, 2].set_title(f'Dark Pixels Only\n({len(dark_pixels)} pixels, {stats["dark_percentage"]:.1f}%)')
    axes[0, 2].axis('off')

    # 4. White pixels only
    axes[1, 0].imshow(white_only_img)
    axes[1, 0].set_title(f'White Spots Only\n({len(white_pixels)} pixels, {stats["white_percentage"]:.1f}%)')
    axes[1, 0].axis('off')

    # 5. Intensity-based coloring
    axes[1, 1].imshow(intensity_overlay)
    axes[1, 1].set_title('Intensity-Based Coloring\n(Blue=Dark, Red=Bright)')
    axes[1, 1].axis('off')

    # 6. Statistics panel
    axes[1, 2].axis('off')

    stats_text = f"""
    Clustering Statistics:

    Total Wingtip Pixels: {stats['total_pixels']:,}

    Dark Cluster (Actual Wingtip):
    • Count: {stats['dark_pixels']:,} ({stats['dark_percentage']:.1f}%)
    • Mean Intensity: {stats['dark_mean_intensity']:.1f}

    White Cluster (Spots/Streaks):
    • Count: {stats['white_pixels']:,} ({stats['white_percentage']:.1f}%)
    • Mean Intensity: {stats['white_mean_intensity']:.1f}

    Impact on Mean Calculation:
    • Original Mean: {stats['original_mean']:.1f}
    • Filtered Mean: {stats['filtered_mean']:.1f}
    • Difference: {abs(stats['original_mean'] - stats['filtered_mean']):.1f}

    Intensity Range:
    • Min: {min_intensity:.1f}
    • Max: {max_intensity:.1f}
    • Range: {max_intensity - min_intensity:.1f}
    """

    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save results if requested
    if save_results:
        output_dir = "Pixel_Clustering_Validation"
        os.makedirs(output_dir, exist_ok=True)

        # Save the comprehensive plot
        plt.savefig(os.path.join(output_dir, f'clustering_validation_{image_name}.png'),
                    dpi=300, bbox_inches='tight')

        # Save individual overlay images
        cv2.imwrite(os.path.join(output_dir, f'overlay_both_{image_name}.png'),
                    cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f'dark_only_{image_name}.png'),
                    cv2.cvtColor(dark_only_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f'white_only_{image_name}.png'),
                    cv2.cvtColor(white_only_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f'intensity_overlay_{image_name}.png'),
                    cv2.cvtColor(intensity_overlay, cv2.COLOR_RGB2BGR))

        print(f"Validation images saved to: {output_dir}")

    plt.show()

    return {
        'overlay_both': overlay_img,
        'dark_only': dark_only_img,
        'white_only': white_only_img,
        'intensity_overlay': intensity_overlay,
        'statistics': stats,
        'dark_pixel_coords': dark_pixel_coords,
        'white_pixel_coords': white_pixel_coords
    }


def interactive_pixel_inspector(image_path, seg_path, wingtip_pixels, wingtip_mask,
                                cluster_labels, dark_cluster_idx, white_cluster_idx):
    """
    Create an interactive tool to inspect individual pixels
    """

    print("=== INTERACTIVE PIXEL INSPECTOR ===")
    print("This tool helps you manually verify the clustering results")
    print()

    # Get coordinates of wingtip pixels
    wingtip_coords = np.where(wingtip_mask > 0)

    # Separate dark and white pixel coordinates
    dark_coords = []
    white_coords = []

    for i, (y, x) in enumerate(zip(wingtip_coords[0], wingtip_coords[1])):
        if cluster_labels[i] == dark_cluster_idx:
            dark_coords.append((y, x, wingtip_pixels[i]))
        else:
            white_coords.append((y, x, wingtip_pixels[i]))

    # Sort by intensity
    dark_coords.sort(key=lambda x: x[2])  # Sort by intensity
    white_coords.sort(key=lambda x: x[2])

    print(f"Dark cluster pixels: {len(dark_coords)}")
    print(f"White cluster pixels: {len(white_coords)}")
    print()

    # Show sample pixels from each cluster
    print("DARKEST pixels classified as 'dark' (should be actual wingtip):")
    for i, (y, x, intensity) in enumerate(dark_coords[:10]):
        print(f"  Pixel {i + 1}: Position ({x}, {y}), Intensity: {intensity:.1f}")

    print("\nBRIGHTEST pixels classified as 'white spots' (should be artifacts):")
    for i, (y, x, intensity) in enumerate(white_coords[-10:]):
        print(f"  Pixel {i + 1}: Position ({x}, {y}), Intensity: {intensity:.1f}")

    print("\nBRIGHTEST pixels classified as 'dark' (check if these should be white spots):")
    for i, (y, x, intensity) in enumerate(dark_coords[-10:]):
        print(f"  Pixel {i + 1}: Position ({x}, {y}), Intensity: {intensity:.1f}")

    print("\nDARKEST pixels classified as 'white spots' (check if these should be dark):")
    for i, (y, x, intensity) in enumerate(white_coords[:10]):
        print(f"  Pixel {i + 1}: Position ({x}, {y}), Intensity: {intensity:.1f}")

    # Summary statistics
    dark_intensities = [coord[2] for coord in dark_coords]
    white_intensities = [coord[2] for coord in white_coords]

    print(f"\nCLUSTER SUMMARY:")
    print(
        f"Dark cluster - Mean: {np.mean(dark_intensities):.1f}, Range: {np.min(dark_intensities):.1f}-{np.max(dark_intensities):.1f}")
    print(
        f"White cluster - Mean: {np.mean(white_intensities):.1f}, Range: {np.min(white_intensities):.1f}-{np.max(white_intensities):.1f}")

    # Check for overlap
    dark_max = np.max(dark_intensities)
    white_min = np.min(white_intensities)

    if dark_max > white_min:
        print(f"\n⚠️  WARNING: Cluster overlap detected!")
        print(f"   Brightest dark pixel: {dark_max:.1f}")
        print(f"   Darkest white pixel: {white_min:.1f}")
        print(f"   This suggests the clusters might not be well-separated")
    else:
        print(f"\n✅ Good separation between clusters")
        print(f"   Gap between clusters: {white_min - dark_max:.1f}")


def create_clustering_visualization(pixels, labels, centers, dark_idx, white_idx, image_name, cluster_info):
    """Create basic visualization of the clustering results"""

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


def remove_white_spots_kmeans_enhanced(wingtip_pixels, wingtip_mask=None, image_path=None, seg_path=None,
                                       visualization=False, overlay_validation=False,
                                       interactive_inspection=False, image_name=""):
    """
    Enhanced version of remove_white_spots_kmeans with overlay validation and interactive inspection

    Args:
        wingtip_pixels: Array of wingtip pixel intensities
        wingtip_mask: Boolean mask for wingtip region (required for overlay validation)
        image_path: Path to original image (required for overlay validation)
        seg_path: Path to segmentation image (required for overlay validation)
        visualization: Whether to create basic visualization plots
        overlay_validation: Whether to create overlay validation plots
        interactive_inspection: Whether to run interactive pixel inspector
        image_name: Name for saving visualization plots

    Returns:
        filtered_pixels: Wingtip pixels with white spots removed
        white_pixel_mask: Boolean mask indicating which pixels were removed
        cluster_info: Dictionary with clustering information
        overlay_results: Dictionary with overlay validation results (if requested)
    """

    if len(wingtip_pixels) == 0:
        return wingtip_pixels, np.array([]), {}, None

    # Reshape pixels for K-means (needs 2D array)
    pixels_reshaped = wingtip_pixels.reshape(-1, 1)

    # Apply K-means clustering with k=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pixels_reshaped)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_.flatten()

    # Identify which cluster represents the darker pixels (actual wingtip)
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

    # Create basic visualization if requested
    if visualization and image_name:
        create_clustering_visualization(wingtip_pixels, cluster_labels, cluster_centers,
                                        dark_cluster_idx, white_cluster_idx, image_name, cluster_info)

    # Initialize overlay_results
    overlay_results = None

    # Create overlay validation if requested
    if overlay_validation and wingtip_mask is not None and image_path and seg_path:
        overlay_results = create_pixel_overlay_validation(
            image_path, seg_path, wingtip_pixels, wingtip_mask,
            cluster_labels, dark_cluster_idx, white_cluster_idx,
            image_name=image_name, save_results=True
        )

    # Run interactive inspection if requested
    if interactive_inspection and wingtip_mask is not None and image_path and seg_path:
        interactive_pixel_inspector(
            image_path, seg_path, wingtip_pixels, wingtip_mask,
            cluster_labels, dark_cluster_idx, white_cluster_idx
        )

    return filtered_pixels, white_pixel_mask, cluster_info, overlay_results


def analyze_wingtip_intensity_distribution_with_enhanced_filtering(image_path, seg_path, species, file_name,
                                                                   create_viz=False, create_overlay=False,
                                                                   interactive_inspect=False):
    """
    Enhanced version with comprehensive clustering validation options
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

    # APPLY ENHANCED K-MEANS CLUSTERING TO REMOVE WHITE SPOTS
    filtered_wingtip_pixels, white_pixel_mask, cluster_info, overlay_results = remove_white_spots_kmeans_enhanced(
        wingtip_pixels,
        wingtip_mask=wingtip_mask,
        image_path=image_path,
        seg_path=seg_path,
        visualization=create_viz,
        overlay_validation=create_overlay,
        interactive_inspection=interactive_inspect,
        image_name=file_name.split('.')[0]
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
    intensity_diffs = mean_wing_intensity - filtered_wingtip_pixels
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

    # Prepare results with enhanced clustering information
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

        # Enhanced clustering information
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
        **very_dark_counts,

        # Store overlay validation results if available
        "overlay_validation_performed": overlay_results is not None,
        "overlay_validation_results": overlay_results
    }

    return results


def validate_clustering_with_comprehensive_analysis(image_path, seg_path, species, file_name):
    """
    Complete validation workflow with comprehensive analysis
    """

    print(f"Validating clustering for: {file_name}")
    print("=" * 60)

    # Run comprehensive analysis with all validation options
    results = analyze_wingtip_intensity_distribution_with_enhanced_filtering(
        image_path, seg_path, species, file_name,
        create_viz=True,
        create_overlay=True,
        interactive_inspect=True
    )

    if results is None:
        print("Analysis failed - check image paths and segmentation format")
        return None

    # Print comprehensive summary
    print(f"\nCOMPREHENSIVE CLUSTERING SUMMARY:")
    print(f"Original mean intensity: {results['original_wingtip_intensity']:.1f}")
    print(f"Filtered mean intensity: {results['mean_wingtip_intensity']:.1f}")
    print(f"White pixels removed: {results['white_pixels_removed']} ({results['white_pixel_percentage']:.1f}%)")
    print(f"Dark cluster center: {results['dark_cluster_center']:.1f}")
    print(f"White cluster center: {results['white_cluster_center']:.1f}")

    # Enhanced recommendations
    if results['white_pixel_percentage'] > 20:
        print("\n⚠️  High percentage of white pixels removed - verify these are actually artifacts")
        print("   → Check overlay validation images to confirm classification accuracy")
    elif results['white_pixel_percentage'] < 5:
        print("\n⚠️  Very few white pixels detected - clustering might not be necessary")
        print("   → Consider skipping clustering for this image")
    else:
        print("\n✅ Reasonable amount of white pixels detected for filtering")

    # Check cluster separation
    cluster_gap = results['white_cluster_center'] - results['dark_cluster_center']
    if cluster_gap < 20:
        print(f"\n⚠️  Small cluster separation ({cluster_gap:.1f}) - manual validation recommended")
    else:
        print(f"\n✅ Good cluster separation ({cluster_gap:.1f})")

    return results


def main_enhanced():
    """
    Enhanced main function with comprehensive validation options
    """
    results = []

    # Configuration for different validation levels
    validation_configs = {
        'basic': {'create_viz': False, 'create_overlay': False, 'interactive_inspect': False},
        'standard': {'create_viz': True, 'create_overlay': False, 'interactive_inspect': False},
        'comprehensive': {'create_viz': True, 'create_overlay': True, 'interactive_inspect': False},
        'full_validation': {'create_viz': True, 'create_overlay': True, 'interactive_inspect': True}
    }

    # Choose validation level
    validation_level = 'standard'  # Change this to 'comprehensive' or 'full_validation' as needed
    config = validation_configs[validation_level]

    # Create comprehensive validation for first few images
    comprehensive_validation_count = 1

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} images with {validation_level} validation...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f" Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")

            # Use comprehensive validation for first few images
            if i < comprehensive_validation_count:
                current_config = validation_configs['comprehensive']
                print(f"  → Using comprehensive validation for detailed analysis")
            else:
                current_config = config

            stats = analyze_wingtip_intensity_distribution_with_enhanced_filtering(
                img_path, seg_path, species_name, file_name,
                create_viz=current_config['create_viz'],
                create_overlay=current_config['create_overlay'],
                interactive_inspect=current_config['interactive_inspect']
            )

            if stats:
                # Remove overlay validation results before saving to CSV (too large)
                stats_for_csv = {k: v for k, v in stats.items() if k != 'overlay_validation_results'}
                results.append(stats_for_csv)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)

        results_dir = "Wingtip_Intensity_Distribution_Enhanced"
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "wingtip_intensity_distribution_enhanced.csv")
        df.to_csv(csv_path, index=False)

        print(f"\nEnhanced results saved to: {csv_path}")

        # Calculate averages by species
        pct_columns = [col for col in df.columns if col.startswith("pct_")]
        dark_columns = [col for col in df.columns if col.startswith("dark_")]
        diff_columns = [col for col in df.columns if col.startswith("diff_")]
        clustering_columns = ['white_pixels_removed', 'white_pixel_percentage',
                              'dark_cluster_center', 'white_cluster_center']

        # Calculate species averages
        species_avg = df.groupby('species')[
            ['mean_wing_intensity', 'original_wingtip_intensity', 'mean_wingtip_intensity'] +
            clustering_columns + pct_columns + dark_columns + diff_columns
            ].mean().reset_index()

        avg_csv_path = os.path.join(results_dir, "wingtip_intensity_averages_enhanced.csv")
        species_avg.to_csv(avg_csv_path, index=False)

        print(f"\nSpecies averages saved to: {avg_csv_path}")

        # Print enhanced summary comparison
        print("\n=== ENHANCED CLUSTERING SUMMARY ===")
        summary = df.groupby('species')[['original_wingtip_intensity', 'mean_wingtip_intensity',
                                         'white_pixel_percentage', 'dark_cluster_center',
                                         'white_cluster_center']].agg(['mean', 'std'])
        print(summary)

        # Print validation summary
        overlay_performed = df['overlay_validation_performed'].sum()
        print(f"\nVALIDATION SUMMARY:")
        print(f"Total images processed: {len(df)}")
        print(f"Images with overlay validation: {overlay_performed}")
        print(f"Average white pixels removed: {df['white_pixel_percentage'].mean():.1f}%")


if __name__ == "__main__":
    main_enhanced()
