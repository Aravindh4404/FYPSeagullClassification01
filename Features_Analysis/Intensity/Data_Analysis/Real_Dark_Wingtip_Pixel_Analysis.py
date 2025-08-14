import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
import sys
from pathlib import Path

# Add the root directory to Python path (adjust based on your project structure)
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
from Features_Analysis.config import *

# Create output directory
output_dir = "Wingtip_Dark_Pixel_Visualizations"
os.makedirs(output_dir, exist_ok=True)


def remove_white_spots_kmeans(wingtip_pixels):
    """Remove white spots from wingtip pixels using K-means clustering"""
    if len(wingtip_pixels) == 0:
        return wingtip_pixels, np.array([]), {}

    pixels_reshaped = wingtip_pixels.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pixels_reshaped)

    cluster_centers = kmeans.cluster_centers_.flatten()
    dark_cluster_idx = np.argmin(cluster_centers)

    dark_pixel_mask = cluster_labels == dark_cluster_idx
    filtered_pixels = wingtip_pixels[dark_pixel_mask]

    cluster_info = {
        'dark_cluster_center': cluster_centers[dark_cluster_idx],
        'white_cluster_center': cluster_centers[np.argmax(cluster_centers)],
        'dark_pixel_count': np.sum(dark_pixel_mask),
        'white_pixel_count': np.sum(~dark_pixel_mask)
    }

    return filtered_pixels, dark_pixel_mask, cluster_info


def calculate_dark_pixels_four_methods(image_path, seg_path, species, file_name):
    """
    Calculate dark pixels using four different methods and return overlay masks
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Convert to grayscale and normalize
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

    # Extract regions
    wing_region, wing_mask = extract_region(gray_img, segmentation_img, "wing")
    wingtip_region, wingtip_mask = extract_region(gray_img, segmentation_img, "wingtip")

    # Get wing and wingtip pixels
    wing_pixels = wing_region[wing_mask > 0]
    wingtip_pixels = wingtip_region[wingtip_mask > 0]

    if len(wing_pixels) == 0 or len(wingtip_pixels) == 0:
        print(f"No wing or wingtip region found in {file_name}")
        return None

    # Remove white spots from wingtip pixels
    filtered_wingtip_pixels, white_removal_mask, cluster_info = remove_white_spots_kmeans(wingtip_pixels)

    if len(filtered_wingtip_pixels) == 0:
        print(f"No filtered wingtip pixels found in {file_name}")
        return None

    # Calculate intensities
    mean_wing_intensity = np.mean(wing_pixels)
    mean_wingtip_intensity = np.mean(filtered_wingtip_pixels)
    min_wing_intensity = np.min(wing_pixels)  # NEW: Darkest pixel in wing

    # Create full-size masks for overlays
    img_height, img_width = gray_img.shape

    # Get coordinates of wingtip pixels
    wingtip_coords = np.where(wingtip_mask > 0)

    # Filter coordinates to only include non-white pixels
    filtered_coords_mask = white_removal_mask
    filtered_wingtip_coords = (wingtip_coords[0][filtered_coords_mask],
                               wingtip_coords[1][filtered_coords_mask])

    # Method 1: Pixels darker than mean wing intensity
    method1_mask = np.zeros((img_height, img_width), dtype=bool)
    dark_pixels_method1 = filtered_wingtip_pixels < mean_wing_intensity
    if np.any(dark_pixels_method1):
        method1_y = filtered_wingtip_coords[0][dark_pixels_method1]
        method1_x = filtered_wingtip_coords[1][dark_pixels_method1]
        method1_mask[method1_y, method1_x] = True

    # Method 2: Pixels darker than mean wingtip intensity
    method2_mask = np.zeros((img_height, img_width), dtype=bool)
    dark_pixels_method2 = filtered_wingtip_pixels < mean_wingtip_intensity
    if np.any(dark_pixels_method2):
        method2_y = filtered_wingtip_coords[0][dark_pixels_method2]
        method2_x = filtered_wingtip_coords[1][dark_pixels_method2]
        method2_mask[method2_y, method2_x] = True

    # Method 3: K-means clustering on filtered pixels
    method3_mask = np.zeros((img_height, img_width), dtype=bool)
    if len(filtered_wingtip_pixels) > 1:
        pixels_reshaped = filtered_wingtip_pixels.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pixels_reshaped)

        cluster_centers = kmeans.cluster_centers_.flatten()
        dark_cluster_idx = np.argmin(cluster_centers)
        dark_pixels_method3 = cluster_labels == dark_cluster_idx

        if np.any(dark_pixels_method3):
            method3_y = filtered_wingtip_coords[0][dark_pixels_method3]
            method3_x = filtered_wingtip_coords[1][dark_pixels_method3]
            method3_mask[method3_y, method3_x] = True

    # Method 4: Pixels darker than darkest wing pixel (NEW METHOD)
    method4_mask = np.zeros((img_height, img_width), dtype=bool)
    dark_pixels_method4 = filtered_wingtip_pixels < min_wing_intensity
    if np.any(dark_pixels_method4):
        method4_y = filtered_wingtip_coords[0][dark_pixels_method4]
        method4_x = filtered_wingtip_coords[1][dark_pixels_method4]
        method4_mask[method4_y, method4_x] = True

    # Calculate statistics
    total_filtered_pixels = len(filtered_wingtip_pixels)
    method1_count = np.sum(dark_pixels_method1) if 'dark_pixels_method1' in locals() else 0
    method2_count = np.sum(dark_pixels_method2) if 'dark_pixels_method2' in locals() else 0
    method3_count = np.sum(dark_pixels_method3) if 'dark_pixels_method3' in locals() else 0
    method4_count = np.sum(dark_pixels_method4) if 'dark_pixels_method4' in locals() else 0

    results = {
        'image_name': file_name,
        'species': species,
        'original_image': original_img,
        'gray_image': gray_img,
        'wingtip_mask': wingtip_mask,
        'method1_mask': method1_mask,
        'method2_mask': method2_mask,
        'method3_mask': method3_mask,
        'method4_mask': method4_mask,  # NEW
        'mean_wing_intensity': mean_wing_intensity,
        'mean_wingtip_intensity': mean_wingtip_intensity,
        'min_wing_intensity': min_wing_intensity,  # NEW
        'total_filtered_pixels': total_filtered_pixels,
        'method1_count': method1_count,
        'method2_count': method2_count,
        'method3_count': method3_count,
        'method4_count': method4_count,  # NEW
        'method1_percentage': (method1_count / total_filtered_pixels) * 100 if total_filtered_pixels > 0 else 0,
        'method2_percentage': (method2_count / total_filtered_pixels) * 100 if total_filtered_pixels > 0 else 0,
        'method3_percentage': (method3_count / total_filtered_pixels) * 100 if total_filtered_pixels > 0 else 0,
        'method4_percentage': (method4_count / total_filtered_pixels) * 100 if total_filtered_pixels > 0 else 0,  # NEW
        'white_pixels_removed': cluster_info['white_pixel_count'],
        'white_removal_percentage': (cluster_info['white_pixel_count'] / len(wingtip_pixels)) * 100 if len(
            wingtip_pixels) > 0 else 0
    }

    return results


def create_overlay_visualization(results):
    """Create overlay visualization for all four methods"""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Dark Pixel Detection Methods - {results['species'].replace('_', ' ')} - {results['image_name']}",
                 fontsize=16, fontweight='bold')

    # Original image (top left)
    ax = axes[0, 0]
    original_rgb = cv2.cvtColor(results['original_image'], cv2.COLOR_BGR2RGB)
    ax.imshow(original_rgb)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Method 1: Darker than mean wing intensity (top middle)
    ax = axes[0, 1]
    ax.imshow(original_rgb)
    overlay = np.zeros_like(original_rgb)
    overlay[results['method1_mask']] = [255, 0, 0]  # Red
    ax.imshow(overlay, alpha=0.6)
    ax.set_title(f'Method 1: < Mean Wing Intensity\n'
                 f'Dark pixels: {results["method1_count"]} ({results["method1_percentage"]:.1f}%)\n'
                 f'Threshold: {results["mean_wing_intensity"]:.1f}',
                 fontsize=10, fontweight='bold')
    ax.axis('off')

    # Method 2: Darker than mean wingtip intensity (top right)
    ax = axes[0, 2]
    ax.imshow(original_rgb)
    overlay = np.zeros_like(original_rgb)
    overlay[results['method2_mask']] = [0, 255, 0]  # Green
    ax.imshow(overlay, alpha=0.6)
    ax.set_title(f'Method 2: < Mean Wingtip Intensity\n'
                 f'Dark pixels: {results["method2_count"]} ({results["method2_percentage"]:.1f}%)\n'
                 f'Threshold: {results["mean_wingtip_intensity"]:.1f}',
                 fontsize=10, fontweight='bold')
    ax.axis('off')

    # Method 3: K-means clustering (bottom left)
    ax = axes[1, 0]
    ax.imshow(original_rgb)
    overlay = np.zeros_like(original_rgb)
    overlay[results['method3_mask']] = [0, 0, 255]  # Blue
    ax.imshow(overlay, alpha=0.6)
    ax.set_title(f'Method 3: K-means Clustering (k=2)\n'
                 f'Dark pixels: {results["method3_count"]} ({results["method3_percentage"]:.1f}%)\n'
                 f'Auto-threshold via clustering',
                 fontsize=10, fontweight='bold')
    ax.axis('off')

    # Method 4: Darker than darkest wing pixel (bottom middle) - NEW
    ax = axes[1, 1]
    ax.imshow(original_rgb)
    overlay = np.zeros_like(original_rgb)
    overlay[results['method4_mask']] = [255, 0, 255]  # Magenta
    ax.imshow(overlay, alpha=0.6)
    ax.set_title(f'Method 4: < Darkest Wing Pixel\n'
                 f'Dark pixels: {results["method4_count"]} ({results["method4_percentage"]:.1f}%)\n'
                 f'Threshold: {results["min_wing_intensity"]:.1f}',
                 fontsize=10, fontweight='bold')
    ax.axis('off')

    # Summary comparison (bottom right)
    ax = axes[1, 2]
    methods = ['Method 1\n(< Wing Mean)', 'Method 2\n(< Wingtip Mean)',
               'Method 3\n(K-means)', 'Method 4\n(< Wing Min)']
    percentages = [results['method1_percentage'], results['method2_percentage'],
                   results['method3_percentage'], results['method4_percentage']]
    colors = ['red', 'green', 'blue', 'magenta']

    bars = ax.bar(methods, percentages, color=colors, alpha=0.7)
    ax.set_ylabel('Dark Pixels (%)', fontweight='bold')
    ax.set_title('Method Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(percentages) * 1.1 if percentages else 1)

    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(percentages) * 0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save the visualization
    filename = f"{results['species']}_{results['image_name'].split('.')[0]}_four_methods_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    return filepath


def create_combined_overlay_visualization(results):
    """Create a single overlay showing all four methods together"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(f"Combined Dark Pixel Methods - {results['species'].replace('_', ' ')} - {results['image_name']}",
                 fontsize=16, fontweight='bold')

    original_rgb = cv2.cvtColor(results['original_image'], cv2.COLOR_BGR2RGB)

    # Left: Original image
    ax1.imshow(original_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Right: Combined overlays
    ax2.imshow(original_rgb)

    # Create combined overlay with priority system
    overlay = np.zeros_like(original_rgb)

    # Start with Method 4 (most conservative) - Magenta
    overlay[results['method4_mask']] = [255, 0, 255]

    # Add Method 1 - Red (will mix with magenta where they overlap)
    method1_only = results['method1_mask'] & ~results['method4_mask']
    method1_and_4 = results['method1_mask'] & results['method4_mask']
    overlay[method1_only] = [255, 0, 0]
    overlay[method1_and_4] = [255, 100, 128]  # Red-magenta mix

    # Add Method 3 - Blue
    method3_only = results['method3_mask'] & ~results['method1_mask'] & ~results['method4_mask']
    overlay[method3_only] = [0, 0, 255]

    # Add Method 2 - Green (will create various combinations)
    method2_only = results['method2_mask'] & ~results['method1_mask'] & ~results['method3_mask'] & ~results[
        'method4_mask']
    overlay[method2_only] = [0, 255, 0]

    # Handle complex overlaps with white for multiple method agreement
    multiple_methods = (results['method1_mask'].astype(int) +
                        results['method2_mask'].astype(int) +
                        results['method3_mask'].astype(int) +
                        results['method4_mask'].astype(int)) >= 3
    overlay[multiple_methods] = [255, 255, 255]  # White for 3+ methods

    ax2.imshow(overlay, alpha=0.7)

    # Create detailed legend
    method1_unique = np.sum(
        results["method1_mask"] & ~results["method2_mask"] & ~results["method3_mask"] & ~results["method4_mask"])
    method2_unique = np.sum(
        results["method2_mask"] & ~results["method1_mask"] & ~results["method3_mask"] & ~results["method4_mask"])
    method3_unique = np.sum(
        results["method3_mask"] & ~results["method1_mask"] & ~results["method2_mask"] & ~results["method4_mask"])
    method4_unique = np.sum(
        results["method4_mask"] & ~results["method1_mask"] & ~results["method2_mask"] & ~results["method3_mask"])

    multiple_agreement = np.sum(multiple_methods)

    legend_text = (f'Method Details:\n'
                   f'🔴 Method 1 only: {method1_unique} pixels\n'
                   f'    (< Wing Mean: {results["mean_wing_intensity"]:.1f})\n'
                   f'🟢 Method 2 only: {method2_unique} pixels\n'
                   f'    (< Wingtip Mean: {results["mean_wingtip_intensity"]:.1f})\n'
                   f'🔵 Method 3 only: {method3_unique} pixels\n'
                   f'    (K-means clustering)\n'
                   f'🟣 Method 4 only: {method4_unique} pixels\n'
                   f'    (< Wing Min: {results["min_wing_intensity"]:.1f})\n'
                   f'⚪ 3+ Methods agree: {multiple_agreement} pixels\n\n'
                   f'Total Results:\n'
                   f'Method 1: {results["method1_count"]} ({results["method1_percentage"]:.1f}%)\n'
                   f'Method 2: {results["method2_count"]} ({results["method2_percentage"]:.1f}%)\n'
                   f'Method 3: {results["method3_count"]} ({results["method3_percentage"]:.1f}%)\n'
                   f'Method 4: {results["method4_count"]} ({results["method4_percentage"]:.1f}%)\n\n'
                   f'Total wingtip pixels: {results["total_filtered_pixels"]}\n'
                   f'White pixels removed: {results["white_pixels_removed"]} ({results["white_removal_percentage"]:.1f}%)')

    ax2.set_title('Combined Methods Overlay', fontsize=14, fontweight='bold')
    ax2.text(1.02, 1.0, legend_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    ax2.axis('off')

    plt.tight_layout()

    # Save the combined visualization
    filename = f"{results['species']}_{results['image_name'].split('.')[0]}_combined_four_methods.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    return filepath


def main():
    """Main function to process sample images and create visualizations"""

    print("Creating dark pixel detection method visualizations with 4 methods...")
    print(f"Output directory: {output_dir}")

    all_results = []
    sample_count = 5  # Number of sample images per species

    for species_name, paths in SPECIES.items():
        print(f"\nProcessing {species_name} images...")

        image_paths = get_image_paths(species_name)

        # Process first 5 images for each species
        for i, (img_path, seg_path) in enumerate(image_paths[:sample_count]):
            file_name = os.path.basename(img_path)
            print(f"  Processing image {i + 1}/{sample_count}: {file_name}")

            results = calculate_dark_pixels_four_methods(
                img_path, seg_path, species_name, file_name
            )

            if results:
                all_results.append(results)

                # Create individual method comparison
                print(f"    Creating four-method comparison visualization...")
                create_overlay_visualization(results)

                # Create combined overlay
                print(f"    Creating combined overlay visualization...")
                create_combined_overlay_visualization(results)
            else:
                print(f"    Skipped {file_name} due to processing error")

    # Create summary statistics
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS FOR DARK PIXEL DETECTION METHODS (4 METHODS)")
        print("=" * 80)

        for species in set(r['species'] for r in all_results):
            species_results = [r for r in all_results if r['species'] == species]
            print(f"\n{species.replace('_', ' ')}:")

            method1_avg = np.mean([r['method1_percentage'] for r in species_results])
            method2_avg = np.mean([r['method2_percentage'] for r in species_results])
            method3_avg = np.mean([r['method3_percentage'] for r in species_results])
            method4_avg = np.mean([r['method4_percentage'] for r in species_results])

            print(f"  Method 1 (< wing mean): {method1_avg:.1f}% average")
            print(f"  Method 2 (< wingtip mean): {method2_avg:.1f}% average")
            print(f"  Method 3 (k-means clustering): {method3_avg:.1f}% average")
            print(f"  Method 4 (< wing minimum): {method4_avg:.1f}% average")

            wing_intensity_avg = np.mean([r['mean_wing_intensity'] for r in species_results])
            wingtip_intensity_avg = np.mean([r['mean_wingtip_intensity'] for r in species_results])
            min_wing_intensity_avg = np.mean([r['min_wing_intensity'] for r in species_results])

            print(f"  Average wing mean intensity: {wing_intensity_avg:.1f}")
            print(f"  Average wingtip mean intensity: {wingtip_intensity_avg:.1f}")
            print(f"  Average wing minimum intensity: {min_wing_intensity_avg:.1f}")
            print(
                f"  Average white pixels removed: {np.mean([r['white_removal_percentage'] for r in species_results]):.1f}%")

    print(f"\n✅ Analysis complete! All visualizations saved to {output_dir}/")
    print("\nReview the overlay images to compare all four methods:")
    print("🔴 Method 1: Conservative - pixels darker than wing mean")
    print("🟢 Method 2: Moderate - pixels darker than wingtip mean")
    print("🔵 Method 3: Data-driven - automatic threshold via k-means clustering")
    print("🟣 Method 4: Ultra-conservative - pixels darker than darkest wing pixel")
    print("\nMethod 4 should identify only the most extremely dark pixels in the wingtip region.")


if __name__ == "__main__":
    main()