import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage.feature import local_binary_pattern

# Import your configuration, which should define:
#   - REGION_COLORS (dict: region_name -> BGR color)
#   - extract_region_mask(segmentation, region_name) function
#   - SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR, GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR
from Features_Analysis.config import *

###############################################################################
# SETTINGS
###############################################################################
RADIUS = 1
N_POINTS = 8 * RADIUS
METHOD = 'default'  # or 'default', 'ror', etc.
DEBUG = True  # Set to True to display images inline and mark region centers

# Output directories for saving histogram plots and aggregated results
OUTPUT_DIR = "../Outputs/LBP_Features"
os.makedirs(OUTPUT_DIR, exist_ok=True)
AGGREGATED_RESULTS_PATH = os.path.join(OUTPUT_DIR, "lbp_features.pkl")


def extract_lbp_features(image, mask=None):
    """
    Computes LBP for the image (or masked region) and returns texture features.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute LBP
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)

    # Determine number of bins
    if METHOD == 'uniform':
        n_bins = int(N_POINTS * (N_POINTS - 1) + 3)
    else:
        n_bins = 2 ** N_POINTS

    # If mask is provided, only use those pixels
    if mask is not None and cv2.countNonZero(mask) > 0:
        lbp_masked = lbp[mask > 0]
        hist, bin_edges = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)
        lbp_data = lbp_masked
    else:
        hist, bin_edges = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        lbp_data = lbp.flatten()

    # Calculate texture features
    features = {
        'histogram': hist,
        # Basic statistics
        'mean': np.mean(lbp_data),
        'std': np.std(lbp_data),
        'variance': np.var(lbp_data),
        'median': np.median(lbp_data),
        # Histogram-based features
        'energy': np.sum(hist ** 2),  # Energy/Uniformity
        'entropy': -np.sum(hist * np.log2(hist + 1e-10)),  # Randomness
        'uniformity': np.sum(hist ** 2),  # Same as energy, more intuitive name
        # Pattern-based features
        'contrast': np.sum(np.arange(len(hist)) ** 2 * hist),  # Contrast measure
        'homogeneity': np.sum(hist / (1 + np.arange(len(hist)))),  # Local similarity
        'dissimilarity': np.sum(np.arange(len(hist)) * hist),  # Pattern difference
        # Additional features
        'smoothness': 1 - (1 / (1 + np.var(lbp_data))),  # Relative smoothness
        'skewness': np.mean(((lbp_data - np.mean(lbp_data)) / (np.std(lbp_data) + 1e-10)) ** 3),
        # Distribution asymmetry
        'kurtosis': np.mean(((lbp_data - np.mean(lbp_data)) / (np.std(lbp_data) + 1e-10)) ** 4),  # Peakedness
    }

    return features


def debug_visualization(image, segmentation, region_masks, region_stats):
    """
    For debugging: show the original image, the segmentation, and mark the region centers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Segmentation with centers
    debug_img = np.zeros_like(segmentation)
    for region_name, mask in region_masks.items():
        if region_name in region_stats:
            color = REGION_COLORS[region_name]
            region_pixels = cv2.bitwise_and(
                np.full_like(segmentation, color),
                np.full_like(segmentation, color),
                mask=mask
            )
            debug_img = cv2.add(debug_img, region_pixels)

            cx, cy = region_stats[region_name]["center"]
            cv2.circle(debug_img, (cx, cy), 5, (255, 255, 255), -1)

    axes[1].imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Segmentation + Region Centers")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    print("Starting Enhanced LBP Feature Extraction...")

    # This dictionary will store features for each species and region
    all_features = {}

    # For each species, get images and segmentations
    for species_name in SPECIES:
        print(f"\n--- Processing {species_name} ---")
        pairs = get_image_paths(species_name)
        if not pairs:
            print("  No images found.")
            continue

        # Initialize storage for this species
        if species_name not in all_features:
            all_features[species_name] = {}

        # Process each (image, segmentation) pair
        for i, (img_path, seg_path) in enumerate(pairs, start=1):
            print(f"  [{i}/{len(pairs)}] {os.path.basename(img_path)}")
            image = cv2.imread(img_path)
            seg = cv2.imread(seg_path)
            if image is None or seg is None:
                print("    Warning: could not load image or segmentation.")
                continue

            # Get region masks and stats (e.g., center positions)
            region_masks, region_stats = get_region_masks(seg)

            # Optional debug visualization (only for the first image)
            if DEBUG and i == 1:
                debug_visualization(image, seg, region_masks, region_stats)

            # For each region, compute LBP features
            for region_name, mask in region_masks.items():
                features = extract_lbp_features(image, mask)
                hist = features['histogram']

                # Display the histogram as a bar chart
                plt.figure(figsize=(8, 4))
                plt.bar(range(len(hist)), hist, color='blue', alpha=0.7)
                plt.title(f"{species_name} | {region_name} LBP Histogram\n{os.path.basename(img_path)}")
                plt.xlabel("LBP Code")
                plt.ylabel("Frequency")
                plt.tight_layout()

                # Save the figure
                out_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{region_name}_hist.png"
                out_path = os.path.join(OUTPUT_DIR, species_name)
                os.makedirs(out_path, exist_ok=True)
                plt.savefig(os.path.join(out_path, out_name))

                # Optionally show it on screen
                if DEBUG:
                    plt.show()
                else:
                    plt.close()

                # Save individual features
                sample_id = os.path.splitext(os.path.basename(img_path))[0]
                if region_name not in all_features[species_name]:
                    all_features[species_name][region_name] = {}

                # Store individual image features (exclude histogram to save space)
                feature_dict = {k: v for k, v in features.items() if k != 'histogram'}
                feature_dict['histogram'] = features['histogram']  # Keep histogram for visualization
                all_features[species_name][region_name][sample_id] = feature_dict

    # Compute aggregate statistics for each species and region
    aggregated_stats = {}
    for species_name, regions in all_features.items():
        aggregated_stats[species_name] = {}
        for region_name, samples in regions.items():
            # Get list of all feature names (excluding histogram)
            sample_keys = next(iter(samples.values())).keys()
            feature_names = [name for name in sample_keys if name != 'histogram']

            # Initialize containers for each feature
            aggregated_features = {name: [] for name in feature_names}
            histograms = []

            # Collect all values
            for sample_id, features in samples.items():
                for name in feature_names:
                    aggregated_features[name].append(features[name])
                histograms.append(features['histogram'])

            # Calculate mean and std of each feature across samples
            region_stats = {}
            for name in feature_names:
                values = np.array(aggregated_features[name])
                region_stats[f'{name}_avg'] = np.mean(values)
                region_stats[f'{name}_std'] = np.std(values)

            # Calculate mean histogram
            region_stats['mean_histogram'] = np.mean(np.array(histograms), axis=0)

            # Store aggregated statistics for this region
            aggregated_stats[species_name][region_name] = region_stats

    # Save all features and aggregated statistics
    final_output = {
        'individual_features': all_features,
        'aggregated_stats': aggregated_stats
    }

    with open(AGGREGATED_RESULTS_PATH, "wb") as f:
        pickle.dump(final_output, f)

    print("\nAll LBP features extracted and saved!")
    print(f"Individual histogram images are saved in '{OUTPUT_DIR}'.")
    print(f"All features and statistics saved to '{AGGREGATED_RESULTS_PATH}'.")


if __name__ == "__main__":
    main()