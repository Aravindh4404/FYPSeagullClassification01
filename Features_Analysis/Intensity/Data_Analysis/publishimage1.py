import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
from pathlib import Path

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
from Features_Analysis.config import *


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


def process_single_image(image_path, seg_path, output_dir="report_images"):
    """Process a single image and generate the three required visualizations"""

    os.makedirs(output_dir, exist_ok=True)

    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images")
        return None

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

    # Extract regions
    wing_region, wing_mask = extract_region(gray_img, segmentation_img, "wing")
    wingtip_region, wingtip_mask = extract_region(gray_img, segmentation_img, "wingtip")

    wing_pixels = wing_region[wing_mask > 0]
    wingtip_pixels = wingtip_region[wingtip_mask > 0]

    if len(wing_pixels) == 0 or len(wingtip_pixels) == 0:
        print(f"No wing or wingtip region found")
        return None

    # Remove white spots
    filtered_wingtip_pixels, white_removal_mask, cluster_info = remove_white_spots_kmeans(wingtip_pixels)

    if len(filtered_wingtip_pixels) == 0:
        print(f"No filtered wingtip pixels found")
        return None

    # Calculate Method 3 (K-means on filtered pixels)
    labels3 = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(
        filtered_wingtip_pixels.reshape(-1, 1)
    )
    centers3 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(
        filtered_wingtip_pixels.reshape(-1, 1)
    ).cluster_centers_.flatten()
    dark3_idx = np.argmin(centers3)

    # Create masks
    # Method 3 mask on the full 2D wingtip region
    method3_mask_wingtip = np.zeros_like(wingtip_mask, dtype=bool)
    wingtip_coords = np.where(wingtip_mask > 0)
    filtered_coords_indices = np.where(white_removal_mask)[0]
    dark_pixels_in_filtered = labels3 == dark3_idx

    final_dark_indices = filtered_coords_indices[dark_pixels_in_filtered]
    method3_mask_wingtip[wingtip_coords[0][final_dark_indices],
    wingtip_coords[1][final_dark_indices]] = True

    # White removal mask on full 2D wingtip region
    white_removed_mask_2d = np.zeros_like(wingtip_mask, dtype=bool)
    white_removed_mask_2d[wingtip_coords[0][white_removal_mask],
    wingtip_coords[1][white_removal_mask]] = True

    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # IMAGE 1: Original bird image
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.imshow(original_rgb)
    ax1.axis('off')
    plt.tight_layout(pad=0)
    img1_path = os.path.join(output_dir, "1_original_bird.png")
    plt.savefig(img1_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Saved: {img1_path}")

    # IMAGE 2: Primaries with white areas removed (pixels used for calculation)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.imshow(original_rgb)
    overlay2 = np.zeros_like(original_rgb)
    overlay2[white_removed_mask_2d] = [255, 0, 0]  # Green for retained pixels
    ax2.imshow(overlay2, alpha=0.5)
    ax2.axis('off')
    plt.tight_layout(pad=0)
    img2_path = os.path.join(output_dir, "2_white_removed_pixels.png")
    plt.savefig(img2_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Saved: {img2_path}")

    # IMAGE 3: Darker outer primaries highlighted (Method 3 - K-means)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    ax3.imshow(original_rgb)
    overlay3 = np.zeros_like(original_rgb)
    overlay3[method3_mask_wingtip] = [255, 255, 0]  # Blue for dark pixels
    ax3.imshow(overlay3, alpha=0.5)
    ax3.axis('off')
    plt.tight_layout(pad=0)
    img3_path = os.path.join(output_dir, "3_dark_primaries_method3.png")
    plt.savefig(img3_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Saved: {img3_path}")

    print(f"\n📊 Statistics:")
    print(f"   White pixels removed: {cluster_info['white_pixel_count']}")
    print(f"   Dark pixels detected (Method 3): {np.sum(dark_pixels_in_filtered)}")

    return {
        'original': img1_path,
        'white_removed': img2_path,
        'dark_highlighted': img3_path
    }


if __name__ == "__main__":
    # SPECIFY YOUR IMAGE HERE
    # Example: Replace with your actual image path
    species_name = "Glaucous_Winged_Gull"  # Change to your species
    image_filename = "0Z2A3579"  # Change to your image filename

    # Get paths (adjust based on your config)
    image_paths = get_image_paths(species_name)

    # Find the specific image
    target_image = None
    for img_path, seg_path in image_paths:
        if image_filename in img_path:
            target_image = (img_path, seg_path)
            break

    if target_image:
        print(f"Processing: {target_image[0]}")
        results = process_single_image(target_image[0], target_image[1])
        print("\n✅ All images generated successfully!")
    else:
        print(f"Image not found: {image_filename}")