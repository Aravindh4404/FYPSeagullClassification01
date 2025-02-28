import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage import io, color
import cv2
from scipy import stats
from config import (
    SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR,
    GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR,
    REGION_COLORS, S
)

# Parameters for LBP
radius = 3  # radius defines the size of the neighborhood
n_points = 8 * radius  # number of points in the neighborhood
METHOD = 'uniform'  # Uniform pattern to reduce dimensionality

# Directory to store results
RESULT_DIR = "Outputs/LBP_Results"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


def get_image_paths(img_dir, seg_dir, limit=S):
    """Get paired original and segmentation image paths"""
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])[:limit]
    seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])[:limit]

    img_paths = [os.path.join(img_dir, f) for f in img_files]
    seg_paths = [os.path.join(seg_dir, f) for f in seg_files]

    return list(zip(img_paths, seg_paths))


def extract_lbp_features(image, mask):
    """Extract LBP features from a grayscale image with a binary mask"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply LBP
    lbp = local_binary_pattern(gray, n_points, radius, METHOD)

    # Apply mask
    lbp_masked = lbp.copy()
    lbp_masked[mask == 0] = 0

    # Extract histogram of LBP values from masked region
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp[mask > 0], bins=n_bins, range=(0, n_bins), density=True)

    return lbp, lbp_masked, hist


def analyze_species(species_name, img_paths, seg_paths):
    """Analyze LBP features for a species"""
    all_histograms = {}

    for region_name, color in REGION_COLORS.items():
        all_histograms[region_name] = []

    fig, axes = plt.subplots(S, 4, figsize=(20, 5 * S))
    if S == 1:
        axes = [axes]

    for i, (img_path, seg_path) in enumerate(zip(img_paths, seg_paths)):
        # Load images
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)

        # Display original image
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i][0].imshow(rgb_img)
        axes[i][0].set_title(f'Original - {os.path.basename(img_path)}')
        axes[i][0].axis('off')

        # Display segmentation mask
        rgb_seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        axes[i][1].imshow(rgb_seg)
        axes[i][1].set_title('Segmentation Mask')
        axes[i][1].axis('off')

        # Process each region separately
        for region_name, color in REGION_COLORS.items():
            # Create mask for this region
            mask = cv2.inRange(seg, color, color)

            # Extract LBP features
            lbp, lbp_masked, hist = extract_lbp_features(img, mask)

            # Store histogram for later comparison
            all_histograms[region_name].append(hist)

            # Display masked LBP (for the first region only - to avoid cluttering)
            if region_name == list(REGION_COLORS.keys())[0]:
                axes[i][2].imshow(lbp_masked, cmap='gray')
                axes[i][2].set_title(f'LBP - {region_name}')
                axes[i][2].axis('off')

                # Display histogram (for the first region only)
                axes[i][3].bar(range(len(hist)), hist)
                axes[i][3].set_title(f'LBP Histogram - {region_name}')
                axes[i][3].set_xlabel('LBP Value')
                axes[i][3].set_ylabel('Frequency')

    plt.tight_layout()
    analysis_path = os.path.join(RESULT_DIR, f'lbp_analysis_{species_name}.png')
    plt.savefig(analysis_path)
    plt.close()

    return all_histograms


def compare_species_lbp(slaty_histograms, glaucous_histograms):
    """Compare LBP histograms between species using Chi-Square distance"""
    fig, axes = plt.subplots(len(REGION_COLORS), 1, figsize=(12, 5 * len(REGION_COLORS)))

    if len(REGION_COLORS) == 1:
        axes = [axes]

    chi_square_distances = {}

    for idx, region_name in enumerate(REGION_COLORS.keys()):
        # Average histograms across images for each species
        slaty_avg = np.mean(slaty_histograms[region_name], axis=0)
        glaucous_avg = np.mean(glaucous_histograms[region_name], axis=0)

        # Calculate chi-square distance
        # Replace zeros to avoid division by zero
        eps = 1e-10
        sum_avg = slaty_avg + glaucous_avg + eps
        chi_square = 0.5 * np.sum(((slaty_avg - glaucous_avg) ** 2) / sum_avg)
        chi_square_distances[region_name] = chi_square

        # Plot average histograms
        x = range(len(slaty_avg))
        axes[idx].bar(x, slaty_avg, alpha=0.5, label='Slaty-backed Gull')
        axes[idx].bar(x, glaucous_avg, alpha=0.5, label='Glaucous-winged Gull')
        axes[idx].set_title(f'Region: {region_name} - Chi-Square Distance: {chi_square:.4f}')
        axes[idx].set_xlabel('LBP Bin')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()

    plt.tight_layout()
    comparison_path = os.path.join(RESULT_DIR, 'lbp_comparison.png')
    plt.savefig(comparison_path)
    plt.close()

    return chi_square_distances


def perform_statistical_test(slaty_histograms, glaucous_histograms):
    """Perform statistical tests to determine if LBP distributions are significantly different"""
    results = {}

    for region_name in REGION_COLORS.keys():
        # Flatten the histograms for t-test
        slaty_flat = np.concatenate(slaty_histograms[region_name])
        glaucous_flat = np.concatenate(glaucous_histograms[region_name])

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(slaty_flat, glaucous_flat, equal_var=False)

        results[region_name] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    return results


def visualize_lbp_patterns(slaty_paths, glaucous_paths):
    """Visualize LBP patterns for sample images from both species"""
    # Select first image from each species
    slaty_img = cv2.imread(slaty_paths[0][0])
    slaty_seg = cv2.imread(slaty_paths[0][1])

    glaucous_img = cv2.imread(glaucous_paths[0][0])
    glaucous_seg = cv2.imread(glaucous_paths[0][1])

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    images = [slaty_img, glaucous_img]
    segs = [slaty_seg, glaucous_seg]
    titles = ["Slaty-backed Gull", "Glaucous-winged Gull"]

    for i, (img, seg, title) in enumerate(zip(images, segs, titles)):
        # Show original
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f'{title} - Original')
        axes[i, 0].axis('off')

        # Gray image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale')
        axes[i, 1].axis('off')

        # Full LBP
        lbp = local_binary_pattern(gray, n_points, radius, METHOD)
        axes[i, 2].imshow(lbp, cmap='viridis')
        axes[i, 2].set_title(f'LBP (r={radius}, p={n_points})')
        axes[i, 2].axis('off')

        # Region LBP
        region_name = list(REGION_COLORS.keys())[0]  # Just use first region for visualization
        color = REGION_COLORS[region_name]
        mask = cv2.inRange(seg, color, color)

        lbp_masked = lbp.copy()
        lbp_masked[mask == 0] = 0

        axes[i, 3].imshow(lbp_masked, cmap='viridis')
        axes[i, 3].set_title(f'{region_name} LBP')
        axes[i, 3].axis('off')

    plt.tight_layout()
    visualization_path = os.path.join(RESULT_DIR, 'lbp_visualization.png')
    plt.savefig(visualization_path)
    plt.close()


def main():
    # Get image paths
    slaty_paths = get_image_paths(SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR)
    glaucous_paths = get_image_paths(GLAUCOUS_WINGED_IMG_DIR, GLAUCOUS_WINGED_SEG_DIR)

    print(f"Processing {len(slaty_paths)} Slaty-backed Gull images")
    print(f"Processing {len(glaucous_paths)} Glaucous-winged Gull images")

    # Analyze each species
    print("Analyzing Slaty-backed Gull LBP features...")
    slaty_histograms = analyze_species("Slaty_Backed",
                                       [p[0] for p in slaty_paths],
                                       [p[1] for p in slaty_paths])

    print("Analyzing Glaucous-winged Gull LBP features...")
    glaucous_histograms = analyze_species("Glaucous_Winged",
                                          [p[0] for p in glaucous_paths],
                                          [p[1] for p in glaucous_paths])

    # Compare species
    print("Comparing LBP features between species...")
    chi_distances = compare_species_lbp(slaty_histograms, glaucous_histograms)

    # Statistical tests
    print("Performing statistical tests...")
    stats_results = perform_statistical_test(slaty_histograms, glaucous_histograms)

    # Visualize patterns
    print("Generating LBP pattern visualizations...")
    visualize_lbp_patterns(slaty_paths, glaucous_paths)

    # Print results
    print("\nResults:")
    print("Chi-Square Distances between species:")
    for region, distance in chi_distances.items():
        print(f"  - {region}: {distance:.4f}")

    print("\nStatistical Test Results:")
    for region, result in stats_results.items():
        sig_text = "Significant" if result['significant'] else "Not significant"
        print(f"  - {region}: p-value = {result['p_value']:.4f} ({sig_text})")

    print("\nAnalysis complete. Output images saved in the folder:", RESULT_DIR)


if __name__ == "__main__":
    main()
