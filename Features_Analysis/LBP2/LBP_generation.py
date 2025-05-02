import pandas as pd
from skimage.feature import local_binary_pattern
from Features_Analysis.config import *  # Import configuration file
import cv2
import numpy as np
import os

# Define LBP parameters
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'  # 'uniform', 'default', etc.

# Output directories for saving results
RESULTS_DIR = "LBP_Abstract_Features"
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_abstract_features(lbp_region):
    """
    Extract abstract features like number of '1s' and transitions in binary patterns.
    """
    if lbp_region is None or len(lbp_region) == 0:
        return None

    # Convert LBP values to binary strings
    binary_patterns = [format(int(val), f'0{N_POINTS}b') for val in lbp_region]

    # Calculate number of '1s' and transitions for each pattern
    num_ones = [pattern.count('1') for pattern in binary_patterns]
    num_transitions = [sum(pattern[i] != pattern[i + 1] for i in range(len(pattern) - 1)) + (pattern[0] != pattern[-1])
                       for pattern in binary_patterns]

    # Compute histogram of abstract features
    ones_hist, _ = np.histogram(num_ones, bins=N_POINTS + 1, range=(0, N_POINTS + 1), density=True)
    transitions_hist, _ = np.histogram(num_transitions, bins=N_POINTS + 1, range=(0, N_POINTS + 1), density=True)

    return ones_hist, transitions_hist


def process_image_lbp(image_path, seg_path, species_name, file_name):
    """
    Process a single image to calculate LBP features and abstract features.
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Convert to grayscale and apply min-max normalization to the entire image
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

    results = []

    # Process each region (wing, wingtip, head)
    for region_name in ['wing', 'wingtip', 'head']:
        # Extract region and mask
        region, mask = extract_region(original_img, segmentation_img, region_name)

        if np.sum(mask) == 0:
            print(f"No {region_name} region found in {file_name}")
            continue

        # Get the normalized grayscale region using the mask
        gray_region = gray_img.copy()
        gray_region = gray_region * (mask > 0)  # Apply mask to get only the region of interest

        # Calculate LBP features
        lbp = local_binary_pattern(gray_region, N_POINTS, RADIUS, METHOD)
        lbp_region = lbp[mask > 0]

        # Compute histogram of LBP values
        hist_lbp, _ = np.histogram(lbp_region, bins=256, range=(0, 255), density=True)

        # Extract abstract features (number of '1s' and transitions)
        ones_hist, transitions_hist = extract_abstract_features(lbp_region)

        # Store results
        results.append({
            "species": species_name,
            "image_name": file_name,
            "region": region_name,
            "lbp_histogram": hist_lbp,
            "ones_histogram": ones_hist,
            "transitions_histogram": transitions_hist,
            "mean_intensity": np.mean(gray_region[mask > 0]),
            "std_intensity": np.std(gray_region[mask > 0])
        })

    return results


def process_all_images():
    """
    Process all images across both species and save results.
    """
    all_results = []

    for species_name in SPECIES.keys():
        print(f"\nProcessing images for {species_name}...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f" Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")

            results = process_image_lbp(img_path, seg_path, species_name, file_name)

            if results:
                all_results.extend(results)

    # Save results to CSV
    df_results = pd.DataFrame(all_results)

    csv_path = os.path.join(RESULTS_DIR, "lbp_abstract_features.csv")
    df_results.to_csv(csv_path, index=False)

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    print("Starting LBP Abstract Feature Extraction...")
    process_all_images()