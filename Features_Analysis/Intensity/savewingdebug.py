import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from Features_Analysis.config import extract_region


def plot_image(image, title, cmap=None):
    """Helper function to plot images"""
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def debug_wing_analysis(image_path, seg_path, species):
    """Debug version with step-by-step visualization"""
    print(f"\n{'=' * 50}\nProcessing: {os.path.basename(image_path)}\n{'=' * 50}")

    # 1. Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print("Error loading images!")
        return

    # Convert BGR to RGB for proper display
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    plot_image(original_rgb, 'Original Image')

    # 2. Show segmentation mask
    plot_image(segmentation_img, 'Segmentation Mask', cmap='gray')

    # 3. Extract wing region (replace with your actual extract_region function)
    wing_region, wing_mask = extract_region(original_img, segmentation_img, "wing")
    plot_image(wing_region, 'Extracted Wing Region')
    plot_image(wing_mask, 'Wing Mask', cmap='gray')

    # 4. Convert to grayscale
    gray_wing = cv2.cvtColor(wing_region, cv2.COLOR_BGR2GRAY)
    plot_image(gray_wing, 'Grayscale Wing', cmap='gray')

    # 5. Pixel extraction visualization
    print("\nMask application demonstration:")
    print("Original grayscale matrix (small section):")
    print(gray_wing[:3, :3])  # Show 3x3 top-left corner

    print("\nMask matrix (small section):")
    print(wing_mask[:3, :3])

    # 6. Apply mask and get pixels
    wing_pixels = gray_wing[wing_mask > 0]
    print(f"\nTotal wing pixels: {len(wing_pixels)}")

    # 7. Show pixel intensity distribution
    plt.hist(wing_pixels, bins=50, color='blue', alpha=0.7)
    plt.title('Wing Pixel Intensity Distribution')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.show()

    # 8. Calculate statistics
    stats = {
        'mean': np.mean(wing_pixels),
        'std': np.std(wing_pixels),
        'median': np.median(wing_pixels),
        'min': np.min(wing_pixels),
        'max': np.max(wing_pixels),
        'skewness': skew(wing_pixels),
        'kurtosis': kurtosis(wing_pixels),
        'pixel_count': len(wing_pixels)
    }

    print("\nCalculated Statistics:")
    for k, v in stats.items():
        print(f"{k:15}: {v:.2f}")

    return stats


# Example usage
if __name__ == "__main__":
    # Replace with your actual test image paths
    TEST_IMAGE = "D:/FYPSeagullClassification01/Features_Analysis/Dataset/Original_Images/Slaty_Backed_Gull/0H5A2453.png"
    TEST_MASK = "D:/FYPSeagullClassification01/Features_Analysis/Dataset/Colored_Images/Slaty_Backed_Gull/0H5A2453.png"
    SPECIES_NAME = "slaty_backed_gull"

    debug_wing_analysis(TEST_IMAGE, TEST_MASK, SPECIES_NAME)
