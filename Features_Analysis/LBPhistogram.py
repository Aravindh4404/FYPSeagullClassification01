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
METHOD = 'default'   # or 'default', 'ror', etc.
DEBUG = True         # Set to True to display images inline and mark region centers

# Output directories for saving histogram plots and aggregated results
OUTPUT_DIR = "Outputs/LBP_Histograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)
AGGREGATED_RESULTS_PATH = os.path.join(OUTPUT_DIR, "mean_histograms.pkl")


def extract_lbp_histogram(image, mask=None):
    """
    Computes LBP for the image (or masked region) and returns the normalized histogram.
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
        hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_bins), density=True)
    else:
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    return hist


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
    print("Starting LBP Histogram Extraction...")

    # This dictionary will store histograms for each species and region
    aggregated_histograms = {}

    # For each species, get images and segmentations
    for species_name in SPECIES:
        print(f"\n--- Processing {species_name} ---")
        pairs = get_image_paths(species_name)
        if not pairs:
            print("  No images found.")
            continue

        # Initialize storage for this species
        if species_name not in aggregated_histograms:
            aggregated_histograms[species_name] = {}

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

            # For each region, compute LBP histogram, display/save it, and accumulate the results
            for region_name, mask in region_masks.items():
                hist = extract_lbp_histogram(image, mask)

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

                # Aggregate histogram for later analysis
                if region_name not in aggregated_histograms[species_name]:
                    aggregated_histograms[species_name][region_name] = []
                aggregated_histograms[species_name][region_name].append(hist)

    # Compute mean histograms for each species and region
    mean_histograms = {}
    for species_name, regions in aggregated_histograms.items():
        mean_histograms[species_name] = {}
        for region_name, hist_list in regions.items():
            # Stack histograms and compute the mean along the image axis
            mean_hist = np.mean(np.array(hist_list), axis=0)
            mean_histograms[species_name][region_name] = mean_hist

    # Save the aggregated mean histograms to a pickle file for later analysis
    with open(AGGREGATED_RESULTS_PATH, "wb") as f:
        pickle.dump(mean_histograms, f)

    print("\nAll LBP histograms extracted and saved!")
    print(f"Individual histogram images are saved in '{OUTPUT_DIR}'.")
    print(f"Aggregated mean histograms saved to '{AGGREGATED_RESULTS_PATH}'.")


if __name__ == "__main__":
    main()
