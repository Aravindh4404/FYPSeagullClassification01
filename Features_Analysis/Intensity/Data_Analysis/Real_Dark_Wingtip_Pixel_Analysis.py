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

# Create main output directory
main_output_dir = "Wingtip_Dark_Pixel_Visualizations"
os.makedirs(main_output_dir, exist_ok=True)


def create_species_output_dir(species_name, main_dir):
    """Create and return species-specific output directory"""
    species_dir = os.path.join(main_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)
    return species_dir


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
    Headless: compute four dark-pixel methods and return numeric results only.
    """
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

    # Extract regions
    wing_region, wing_mask = extract_region(gray_img, segmentation_img, "wing")
    wingtip_region, wingtip_mask = extract_region(gray_img, segmentation_img, "wingtip")

    wing_pixels = wing_region[wing_mask > 0]
    wingtip_pixels = wingtip_region[wingtip_mask > 0]

    if len(wing_pixels) == 0 or len(wingtip_pixels) == 0:
        print(f"No wing or wingtip region found in {file_name}")
        return None

    # Remove white spots from wingtip pixels (k=2: dark vs white)
    filtered_wingtip_pixels, white_removal_mask, cluster_info = remove_white_spots_kmeans(wingtip_pixels)
    if len(filtered_wingtip_pixels) == 0:
        print(f"No filtered wingtip pixels found in {file_name}")
        return None

    # Intensities
    mean_wing_intensity = float(np.mean(wing_pixels))
    mean_wingtip_intensity = float(np.mean(filtered_wingtip_pixels))
    min_wing_intensity = int(np.min(wing_pixels))

    total_filtered_pixels = int(len(filtered_wingtip_pixels))

    # Method 1: < mean wing
    method1_count = int(np.sum(filtered_wingtip_pixels < mean_wing_intensity))
    # Method 2: < mean wingtip
    method2_count = int(np.sum(filtered_wingtip_pixels < mean_wingtip_intensity))
    # Method 3: k-means on filtered wingtip pixels (dark cluster size)
    if len(filtered_wingtip_pixels) > 1:
        labels3 = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(
            filtered_wingtip_pixels.reshape(-1, 1)
        )
        centers3 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(
            filtered_wingtip_pixels.reshape(-1, 1)
        ).cluster_centers_.flatten()
        dark3 = np.argmin(centers3)
        method3_count = int(np.sum(labels3 == dark3))
        dark3_center = float(centers3[dark3])
        white3_center = float(centers3[np.argmax(centers3)])
    else:
        method3_count = 0
        dark3_center = float('nan')
        white3_center = float('nan')

    # Method 4: < darkest wing pixel
    method4_count = int(np.sum(filtered_wingtip_pixels < min_wing_intensity))

    # Percentages
    def pct(x): return (x / total_filtered_pixels) * 100 if total_filtered_pixels > 0 else 0.0
    method1_percentage = float(pct(method1_count))
    method2_percentage = float(pct(method2_count))
    method3_percentage = float(pct(method3_count))
    method4_percentage = float(pct(method4_count))

    return {
        "species": species,
        "image_name": file_name,
        "total_filtered_pixels": total_filtered_pixels,
        "method1_count": method1_count,
        "method2_count": method2_count,
        "method3_count": method3_count,
        "method4_count": method4_count,
        "method1_percentage": method1_percentage,
        "method2_percentage": method2_percentage,
        "method3_percentage": method3_percentage,
        "method4_percentage": method4_percentage,
        "mean_wing_intensity": mean_wing_intensity,
        "mean_wingtip_intensity": mean_wingtip_intensity,
        "min_wing_intensity": min_wing_intensity,
        "white_pixels_removed": int(cluster_info.get("white_pixel_count", 0)),
        "white_removal_percentage": float(
            (cluster_info.get("white_pixel_count", 0) / len(wingtip_pixels)) * 100
            if len(wingtip_pixels) > 0 else 0.0
        ),
        # Optional diagnostics
        "dark_cluster_center_white_removal": float(cluster_info.get("dark_cluster_center", np.nan)),
        "white_cluster_center_white_removal": float(cluster_info.get("white_cluster_center", np.nan)),
        "dark_cluster_center_method3": dark3_center,
        "white_cluster_center_method3": white3_center,
    }

def create_overlay_visualization(results, output_dir):
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

    # Save the visualization in species-specific folder
    filename = f"{results['species']}_{results['image_name'].split('.')[0]}_four_methods_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    return filepath


def create_combined_overlay_visualization(results, output_dir):
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

    # Save the combined visualization in species-specific folder
    filename = f"{results['species']}_{results['image_name'].split('.')[0]}_combined_four_methods.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    return filepath

def main():
    """Process images and save per-image results to a single CSV (no visualizations)."""

    print("Computing dark pixel metrics (4 methods) without visualizations...")
    print(f"Main output directory: {main_output_dir}")

    os.makedirs(main_output_dir, exist_ok=True)

    csv_rows = []
    sample_count = 100  # Number of sample images per species

    for species_name, paths in SPECIES.items():
        print(f"\nProcessing {species_name} images...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:sample_count]):
            file_name = os.path.basename(img_path)
            print(f"  Processing image {i + 1}/{sample_count}: {file_name}")

            results = calculate_dark_pixels_four_methods(
                img_path, seg_path, species_name, file_name
            )

            if results:
                csv_rows.append(results)
            else:
                print(f"    Skipped {file_name} due to processing error")

    # Save consolidated CSV
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        all_csv_path = os.path.join(main_output_dir, "dark_pixel_results_all_images.csv")
        df.to_csv(all_csv_path, index=False)
        print(f"\n✅ Per-image results saved: {all_csv_path}")
    else:
        print("\n⚠️ No results to save (no valid images processed).")

    print("\n✅ Done. No visualizations were generated.")


if __name__ == "__main__":
    main()