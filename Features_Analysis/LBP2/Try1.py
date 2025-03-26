import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern
from scipy.stats import entropy, wasserstein_distance
from Features_Analysis.config import *  # Import configuration file

# Create output directory for saving results
results_dir = "LBP_Analysis_Results"
os.makedirs(results_dir, exist_ok=True)

# Define LBP parameters
radius = 3  # Radius for LBP calculation
n_points = 8 * radius  # Number of points to consider
method = 'uniform'  # LBP method ('uniform', 'default', 'ror', 'var')


def calculate_lbp(original_img, segmentation_img, region_name):
    """Calculate Local Binary Pattern for a specific region of an image."""
    # Extract the region using functions from config file
    region, mask = extract_region(original_img, segmentation_img, region_name)

    # Convert to grayscale
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Check if region exists
    if np.sum(mask) == 0:
        print(f"No {region_name} region found")
        return None, None, None

    # Calculate LBP for the entire image
    lbp = local_binary_pattern(gray_region, n_points, radius, method)

    # Extract LBP values only for the masked region
    lbp_region = lbp[mask > 0]

    return lbp, lbp_region, mask


def compute_lbp_histogram(lbp_region):
    """Compute histogram of LBP values."""
    if lbp_region is None or len(lbp_region) == 0:
        return None, None

    # For uniform LBP, the number of bins is n_points + 2
    if method == 'uniform':
        n_bins = n_points + 2
    else:
        n_bins = int(lbp_region.max() + 1)

    hist, bins = np.histogram(lbp_region, density=True, bins=n_bins, range=(0, n_bins))
    return hist, bins


def compare_histograms(hist1, hist2):
    """Compare two histograms using multiple metrics."""
    if hist1 is None or hist2 is None:
        return {
            'kl_divergence': float('inf'),
            'earth_movers_distance': float('inf'),
            'chi_square': float('inf')
        }

    # Ensure histograms have the same length
    max_len = max(len(hist1), len(hist2))
    if len(hist1) < max_len:
        hist1 = np.pad(hist1, (0, max_len - len(hist1)))
    if len(hist2) < max_len:
        hist2 = np.pad(hist2, (0, max_len - len(hist2)))

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    hist1_norm = hist1 + epsilon
    hist2_norm = hist2 + epsilon

    # Normalize
    hist1_norm = hist1_norm / np.sum(hist1_norm)
    hist2_norm = hist2_norm / np.sum(hist2_norm)

    # Calculate KL divergence (both directions for symmetry)
    kl_div1 = entropy(hist1_norm, hist2_norm)
    kl_div2 = entropy(hist2_norm, hist1_norm)
    sym_kl_div = (kl_div1 + kl_div2) / 2

    # Earth Mover's Distance (Wasserstein)
    emd = wasserstein_distance(np.arange(len(hist1_norm)), np.arange(len(hist2_norm)),
                               hist1_norm, hist2_norm)

    # Chi-Square Distance
    chi_square = np.sum((hist1_norm - hist2_norm) ** 2 / (hist1_norm + hist2_norm))

    return {
        'kl_divergence': sym_kl_div,
        'earth_movers_distance': emd,
        'chi_square': chi_square
    }


def visualize_lbp(original_img, lbp, mask, region_name, species_name, file_name):
    """Visualize the original image, LBP image, and histogram."""
    if lbp is None:
        return

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Original image with masked region highlighted
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    ax1.imshow(original_rgb)

    # Highlight the masked region
    masked_img = np.zeros_like(original_rgb)
    masked_img[mask > 0] = original_rgb[mask > 0]
    ax1.imshow(masked_img, alpha=0.7)  # Overlay the masked region

    ax1.set_title(f"Original Image - {region_name}")
    ax1.axis('off')

    # 2. LBP image (only for the masked region)
    lbp_display = np.zeros_like(lbp)
    lbp_display[mask > 0] = lbp[mask > 0]

    # Normalize for better visualization
    if lbp_display.max() > 0:
        lbp_display = lbp_display / lbp_display.max() * 255

    ax2.imshow(lbp_display, cmap='jet')
    ax2.set_title(f"LBP of {region_name}")
    ax2.axis('off')

    # 3. Histogram of LBP values
    lbp_values = lbp[mask > 0]
    hist, bins = compute_lbp_histogram(lbp_values)

    if hist is not None:
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax3.bar(bin_centers, hist, width=0.9)
        ax3.set_title(f"LBP Histogram - {region_name}")
        ax3.set_xlabel("Uniform LBP values")
        ax3.set_ylabel("Frequency")

    plt.suptitle(f"{species_name} - {file_name} - {region_name} LBP Analysis")
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(results_dir, f"{species_name}_{file_name}_{region_name}_lbp.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def process_single_image_lbp(species_name, image_idx=0):
    """Process a single image and show LBP analysis for each region."""
    # Get image paths
    image_paths = get_image_paths(species_name)

    if not image_paths or image_idx >= len(image_paths):
        print(f"No images found for {species_name} at index {image_idx}")
        return {}

    # Take the specified image
    img_path, seg_path = image_paths[image_idx]
    file_name = os.path.basename(img_path)

    print(f"Processing {file_name} from {species_name}")

    # Load images
    original_img = cv2.imread(img_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {img_path} or {seg_path}")
        return {}

    results = {}

    # Process each region
    for region_name in ['wing', 'wingtip', 'head']:
        lbp, lbp_region, mask = calculate_lbp(original_img, segmentation_img, region_name)

        if lbp is not None and lbp_region is not None:
            # Visualize
            visualize_lbp(original_img, lbp, mask, region_name, species_name, file_name)

            # Compute histogram
            hist, _ = compute_lbp_histogram(lbp_region)

            # Store results
            results[region_name] = {
                'lbp': lbp,
                'lbp_region': lbp_region,
                'histogram': hist,
                'mask': mask
            }

    return results


def compare_species_lbp(slaty_idx=0, glaucous_idx=0):
    """Compare LBP features between two species."""
    # Process one image from each species
    slaty_results = process_single_image_lbp('Slaty_Backed_Gull', slaty_idx)
    glaucous_results = process_single_image_lbp('Glaucous_Winged_Gull', glaucous_idx)

    # Compare histograms for each region
    comparison_results = {}

    for region in ['wing', 'wingtip', 'head']:
        if region in slaty_results and region in glaucous_results:
            slaty_hist = slaty_results[region]['histogram']
            glaucous_hist = glaucous_results[region]['histogram']

            metrics = compare_histograms(slaty_hist, glaucous_hist)
            comparison_results[region] = metrics

            print(f"\nComparison metrics for {region}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

            # Visualize histogram comparison
            plt.figure(figsize=(10, 6))

            # Ensure histograms have the same length
            max_bins = max(len(slaty_hist) if slaty_hist is not None else 0,
                           len(glaucous_hist) if glaucous_hist is not None else 0)

            x_values = np.arange(max_bins)

            if slaty_hist is not None:
                if len(slaty_hist) < max_bins:
                    slaty_hist = np.pad(slaty_hist, (0, max_bins - len(slaty_hist)))
                plt.bar(x_values - 0.2, slaty_hist, width=0.4, alpha=0.7, label='Slaty-backed Gull')

            if glaucous_hist is not None:
                if len(glaucous_hist) < max_bins:
                    glaucous_hist = np.pad(glaucous_hist, (0, max_bins - len(glaucous_hist)))
                plt.bar(x_values + 0.2, glaucous_hist, width=0.4, alpha=0.7, label='Glaucous-winged Gull')

            plt.title(f"LBP Histogram Comparison for {region}")
            plt.xlabel("Uniform LBP values")
            plt.ylabel("Frequency")
            plt.legend()

            # Add comparison metrics as text
            plt.figtext(0.5, 0.01,
                        f"KL Divergence: {metrics['kl_divergence']:.4f} | "
                        f"Earth Mover's Distance: {metrics['earth_movers_distance']:.4f} | "
                        f"Chi-Square: {metrics['chi_square']:.4f}",
                        ha='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

            # Save the comparison figure
            save_path = os.path.join(results_dir, f"{region}_lbp_histogram_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

    return comparison_results


def process_all_images_lbp():
    """Process all images and compute average LBP histograms for each species and region."""
    results = {}

    for species_name in ['Slaty_Backed_Gull', 'Glaucous_Winged_Gull']:
        results[species_name] = {'wing': [], 'wingtip': [], 'head': []}

        # Get image paths
        image_paths = get_image_paths(species_name)

        if not image_paths:
            print(f"No images found for {species_name}")
            continue

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):  # S is from config
            file_name = os.path.basename(img_path)
            print(f"Processing {i + 1}/{min(S, len(image_paths))}: {file_name} from {species_name}")

            # Load images
            original_img = cv2.imread(img_path)
            segmentation_img = cv2.imread(seg_path)

            if original_img is None or segmentation_img is None:
                print(f"Error loading images: {img_path} or {seg_path}")
                continue

            # Process each region
            for region_name in ['wing', 'wingtip', 'head']:
                _, lbp_region, _ = calculate_lbp(original_img, segmentation_img, region_name)

                if lbp_region is not None and len(lbp_region) > 0:
                    hist, _ = compute_lbp_histogram(lbp_region)
                    if hist is not None:
                        results[species_name][region_name].append(hist)

    # Compute average histograms
    avg_results = {}
    all_metrics = []

    for species_name in results:
        avg_results[species_name] = {}

        for region_name in results[species_name]:
            histograms = results[species_name][region_name]

            if not histograms:
                continue

            # Find the maximum length of histograms
            max_len = max(len(hist) for hist in histograms if hist is not None)

            # Pad all histograms to the same length
            padded_hists = []
            for hist in histograms:
                if hist is not None:
                    if len(hist) < max_len:
                        padded_hist = np.pad(hist, (0, max_len - len(hist)))
                    else:
                        padded_hist = hist
                    padded_hists.append(padded_hist)

            if padded_hists:
                avg_hist = np.mean(padded_hists, axis=0)
                avg_results[species_name][region_name] = avg_hist

    # Compare average histograms and save results
    for region_name in ['wing', 'wingtip', 'head']:
        if ('Slaty_Backed_Gull' in avg_results and
                'Glaucous_Winged_Gull' in avg_results and
                region_name in avg_results['Slaty_Backed_Gull'] and
                region_name in avg_results['Glaucous_Winged_Gull']):

            slaty_hist = avg_results['Slaty_Backed_Gull'][region_name]
            glaucous_hist = avg_results['Glaucous_Winged_Gull'][region_name]

            metrics = compare_histograms(slaty_hist, glaucous_hist)
            all_metrics.append({
                'region': region_name,
                **metrics
            })

            print(f"\nAverage comparison metrics for {region_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

            # Visualize average histogram comparison
            plt.figure(figsize=(10, 6))

            # Ensure histograms have the same length
            max_bins = max(len(slaty_hist), len(glaucous_hist))
            x_values = np.arange(max_bins)

            if len(slaty_hist) < max_bins:
                slaty_hist = np.pad(slaty_hist, (0, max_bins - len(slaty_hist)))

            if len(glaucous_hist) < max_bins:
                glaucous_hist = np.pad(glaucous_hist, (0, max_bins - len(glaucous_hist)))

            plt.bar(x_values - 0.2, slaty_hist, width=0.4, alpha=0.7, label='Slaty-backed Gull')
            plt.bar(x_values + 0.2, glaucous_hist, width=0.4, alpha=0.7, label='Glaucous-winged Gull')

            plt.title(f"Average LBP Histogram Comparison for {region_name}")
            plt.xlabel("Uniform LBP values")
            plt.ylabel("Frequency")
            plt.legend()

            # Add comparison metrics as text
            plt.figtext(0.5, 0.01,
                        f"KL Divergence: {metrics['kl_divergence']:.4f} | "
                        f"Earth Mover's Distance: {metrics['earth_movers_distance']:.4f} | "
                        f"Chi-Square: {metrics['chi_square']:.4f}",
                        ha='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

            # Save the comparison figure
            save_path = os.path.join(results_dir, f"{region_name}_average_lbp_histogram_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

    # Save metrics to CSV
    if all_metrics:
        import pandas as pd
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(results_dir, "lbp_comparison_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nComparison metrics saved to {metrics_path}")

    return avg_results, all_metrics


if __name__ == "__main__":
    print("LBP Analysis for Gull Wing, Wingtip, and Head Regions")
    print("=====================================================")
    print(f"LBP Parameters: radius={radius}, points={n_points}, method='{method}'")
    print(f"Results will be saved to: {results_dir}")

    # Process single images first to demonstrate the analysis
    print("\nProcessing single images from each species...")
    comparison_results = compare_species_lbp(slaty_idx=0, glaucous_idx=0)

    # Uncomment to process all images
    # print("\nProcessing all images...")
    # avg_results, all_metrics = process_all_images_lbp()
