import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import cv2
from scipy import stats
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from pathlib import Path
from datetime import datetime

from Features_Analysis.config import *

# Build species dictionary from configuration
SPECIES = {
    "Slaty_Backed_Gull": {
        "img_dir": str(SLATY_BACKED_IMG_DIR),
        "seg_dir": str(SLATY_BACKED_SEG_DIR),
    },
    "Glaucous_Winged_Gull": {
        "img_dir": str(GLAUCOUS_WINGED_IMG_DIR),
        "seg_dir": str(GLAUCOUS_WINGED_SEG_DIR),
    },
}

###############################################################################
# LBP PARAMETERS AND RESULT DIRECTORY
###############################################################################

RADIUS = 2  # Radius defines the size of the neighborhood
N_POINTS = 8 * RADIUS  # Number of points in the neighborhood
METHOD = 'uniform'  # Uniform pattern to reduce dimensionality

# Default output directory; this will be updated in main() with a unique run folder.
RESULT_DIR = "Outputs/LBP_Analysis"
os.makedirs(RESULT_DIR, exist_ok=True)


###############################################################################
# FUNCTION DEFINITIONS
###############################################################################

def get_image_paths(species):
    """Get paired original and segmentation image paths for a species"""
    img_dir = SPECIES[species]["img_dir"]
    seg_dir = SPECIES[species]["seg_dir"]

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    seg_files = sorted([f for f in os.listdir(seg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    # Ensure matching files
    paired_files = []
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        matching_seg = [f for f in seg_files if os.path.splitext(f)[0] == base_name]
        if matching_seg:
            paired_files.append((os.path.join(img_dir, img_file),
                                 os.path.join(seg_dir, matching_seg[0])))
    return paired_files


def extract_lbp_features(image, mask=None):
    """Extract LBP features from a grayscale image, optionally with a mask"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)

    # Create histogram
    n_bins = int(N_POINTS * (N_POINTS - 1) + 3) if METHOD == 'uniform' else 2 ** N_POINTS

    if mask is not None:
        masked_pixels = lbp[mask > 0]
        if len(masked_pixels) > 0:
            hist, _ = np.histogram(masked_pixels, bins=n_bins, range=(0, n_bins), density=True)
        else:
            hist = np.zeros(n_bins)
    else:
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    return lbp, hist


def chi_square_distance(hist1, hist2):
    """Calculate chi-square distance between two histograms"""
    eps = 1e-10  # Prevent division by zero
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))


def analyze_species_texture(species_name, limit=S):
    """Analyze LBP features for all images of a species"""
    print(f"Analyzing {species_name} textures...")
    paths = get_image_paths(species_name)[:limit]

    if not paths:
        print(f"No images found for {species_name}")
        return None

    region_features = {region: [] for region in REGION_COLORS}

    for i, (img_path, seg_path) in enumerate(paths):
        print(f"  Processing image {i + 1}/{len(paths)}: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)

        if img is None or seg is None:
            print(f"  Warning: Could not load {img_path} or {seg_path}. Skipping.")
            continue

        for region_name, color in REGION_COLORS.items():
            mask = cv2.inRange(seg, color, color)
            if cv2.countNonZero(mask) == 0:
                print(f"  Warning: Region {region_name} not found in {os.path.basename(img_path)}")
                continue

            _, hist = extract_lbp_features(img, mask)
            region_features[region_name].append(hist)

    return region_features


def visualize_lbp_comparison(species_data, output_filename):
    """Visualize and compare LBP histograms for each region between species"""
    num_regions = len(REGION_COLORS)
    fig, axes = plt.subplots(num_regions, 1, figsize=(14, 4 * num_regions))

    if num_regions == 1:
        axes = [axes]

    distances = {}

    for i, (region_name, region_color) in enumerate(REGION_COLORS.items()):
        ax = axes[i]
        species_with_region = [species_name for species_name in species_data
                               if region_name in species_data[species_name] and species_data[species_name][region_name]]

        if len(species_with_region) < 2:
            ax.text(0.5, 0.5, f"Insufficient data for region: {region_name}",
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(f"Region: {region_name} - Insufficient Data")
            continue

        avg_hists = {}
        for species_name in species_with_region:
            species_hists = species_data[species_name][region_name]
            if species_hists:
                avg_hists[species_name] = np.mean(species_hists, axis=0)

        x = np.arange(len(next(iter(avg_hists.values()))))
        colors = ['blue', 'orange', 'green', 'red']

        for (species_name, hist), color in zip(avg_hists.items(), colors):
            ax.bar(x, hist, alpha=0.6, label=species_name, color=color)

        if len(avg_hists) >= 2:
            species_list = list(avg_hists.keys())
            for i in range(len(species_list)):
                for j in range(i + 1, len(species_list)):
                    sp1, sp2 = species_list[i], species_list[j]
                    distance = chi_square_distance(avg_hists[sp1], avg_hists[sp2])
                    key = f"{region_name}_{sp1}_vs_{sp2}"
                    distances[key] = distance
                    ax.set_title(f"Region: {region_name} - Chi-Square Distance: {distance:.4f}")

        ax.set_xlabel("LBP Bin")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, output_filename))
    plt.close()
    return distances


def visualize_lbp_patterns(species_paths):
    """Create visual examples of LBP application on sample images from each species"""
    num_species = len(species_paths)
    fig, axes = plt.subplots(num_species, 4, figsize=(16, 4 * num_species))

    if num_species == 1:
        axes = axes.reshape(1, 4)

    for i, (species_name, paths) in enumerate(species_paths.items()):
        if not paths:
            continue
        img_path, seg_path = paths[0]
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is None or seg is None:
            continue

        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'{species_name} - Original')
        axes[i, 0].axis('off')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title('Grayscale')
        axes[i, 1].axis('off')

        lbp, _ = extract_lbp_features(img)
        axes[i, 2].imshow(lbp, cmap='viridis')
        axes[i, 2].set_title(f'LBP (r={RADIUS}, p={N_POINTS})')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
        axes[i, 3].set_title('Segmentation')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'lbp_visualization_examples.png'))
    plt.close()


def visualize_region_lbp(species_paths):
    """Visualize LBP for each region in a sample image (per species)"""
    # For each species, create a separate visualization
    for species_name, paths in species_paths.items():
        if not paths:
            continue

        img_path, seg_path = paths[0]
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is None or seg is None:
            continue

        num_regions = len(REGION_COLORS)
        fig, axes = plt.subplots(num_regions, 3, figsize=(12, 4 * num_regions))

        if num_regions == 1:
            axes = axes.reshape(1, 3)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        full_lbp, _ = extract_lbp_features(img)

        for i, (region_name, color) in enumerate(REGION_COLORS.items()):
            mask = cv2.inRange(seg, color, color)
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            masked_lbp = full_lbp.copy()
            masked_lbp[mask == 0] = 0

            axes[i, 0].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title(f'{region_name} Region')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f'{region_name} Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(masked_lbp, cmap='viridis')
            axes[i, 2].set_title(f'{region_name} LBP')
            axes[i, 2].axis('off')

        plt.tight_layout()
        filename = f"{species_name}_region_lbp_visualization.png"
        plt.savefig(os.path.join(RESULT_DIR, filename))
        plt.close()


def build_feature_dataset(species_data):
    """Build dataset for classification from the extracted features"""
    X, y, region_info = [], [], []
    for species_name, regions in species_data.items():
        for region_name, histograms in regions.items():
            for hist in histograms:
                X.append(hist)
                y.append(species_name)
                region_info.append(region_name)
    return np.array(X), np.array(y), np.array(region_info)


def evaluate_classification(X, y, region_info):
    """Evaluate classification performance overall and per region"""
    results = {}
    print("\nOverall classification performance:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"  Test set accuracy: {accuracy:.4f}")
    results['overall'] = {'cv_accuracy': cv_scores.mean(), 'test_accuracy': accuracy}

    print("\nPer-region classification performance:")
    for region in set(region_info):
        region_mask = (region_info == region)
        X_region, y_region = X[region_mask], y[region_mask]

        if len(set(y_region)) < 2 or len(y_region) < 10:
            print(f"  {region}: Insufficient data")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X_region, y_region, test_size=0.3, random_state=42)
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        if len(y_region) >= 10:
            cv_scores = cross_val_score(clf, X_region, y_region, cv=min(5, len(set(y_region))))
            print(f"  {region}: CV accuracy = {cv_scores.mean():.4f}, Test accuracy = {accuracy:.4f}")
            results[region] = {'cv_accuracy': cv_scores.mean(), 'test_accuracy': accuracy}
        else:
            print(f"  {region}: Test accuracy = {accuracy:.4f} (not enough data for CV)")
            results[region] = {'test_accuracy': accuracy}
    return results


def experiment_with_parameters(img_path, mask_path, region_name, species_name, output_dir=RESULT_DIR):
    """Test different radius and points combinations and visualize LBP neighborhood"""
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    if img is None or mask is None:
        return

    region_mask = cv2.inRange(mask, REGION_COLORS[region_name], REGION_COLORS[region_name])
    radii = [1, 2, 3, 5, 8]
    points_list = [8, 16, 24]

    fig, axes = plt.subplots(len(radii), len(points_list), figsize=(4 * len(points_list), 4 * len(radii)))
    for i, radius in enumerate(radii):
        for j, points in enumerate(points_list):
            lbp = local_binary_pattern(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), points, radius, METHOD)
            masked_lbp = lbp.copy()
            masked_lbp[region_mask == 0] = 0
            axes[i, j].imshow(masked_lbp, cmap='viridis')
            axes[i, j].set_title(f'R={radius}, P={points}')
            axes[i, j].axis('off')
    plt.tight_layout()
    exp_filename = f"{species_name}_{region_name}_parameter_experiment.png"
    plt.savefig(os.path.join(output_dir, exp_filename))
    plt.close()

    # Visualize the LBP neighborhood at a chosen point.
    def visualize_lbp_neighborhood(image_path, x, y, radius, species_name, region_name, output_dir):
        img = cv2.imread(image_path)
        if img is None:
            return
        center_img = img.copy()
        cv2.circle(center_img, (x, y), 2, (0, 0, 255), -1)
        neighborhood_img = img.copy()
        cv2.circle(neighborhood_img, (x, y), radius, (0, 255, 0), 1)
        points_img = img.copy()
        for i in range(N_POINTS):
            angle = 2 * np.pi * i / N_POINTS
            px = int(x + radius * np.cos(angle))
            py = int(y + radius * np.sin(angle))
            cv2.circle(points_img, (px, py), 2, (255, 0, 0), -1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Center Pixel')
        axes[1].imshow(cv2.cvtColor(neighborhood_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Radius = {radius}')
        axes[2].imshow(cv2.cvtColor(points_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'{N_POINTS} Sampling Points')
        plt.tight_layout()
        neigh_filename = f"{species_name}_{region_name}_lbp_neighborhood.png"
        plt.savefig(os.path.join(output_dir, neigh_filename))
        plt.close()

    # Choose the center of the image as the point to visualize.
    h, w = img.shape[:2]
    x, y = w // 2, h // 2
    visualize_lbp_neighborhood(img_path, x, y, radii[0], species_name, region_name, output_dir)


def analyze_texture_properties(species_data):
    """Extract meaningful texture properties from LBP histograms"""
    results = {}
    for species_name, regions in species_data.items():
        results[species_name] = {}
        for region_name, histograms in regions.items():
            if not histograms:
                continue
            avg_hist = np.mean(histograms, axis=0)
            uniformity = np.sum(avg_hist ** 2)
            entropy = -np.sum(avg_hist * np.log2(avg_hist + 1e-10))
            contrast = np.var(np.arange(len(avg_hist)) * avg_hist)
            dominant_patterns = np.argsort(avg_hist)[-3:][::-1]
            results[species_name][region_name] = {
                'uniformity': uniformity,
                'entropy': entropy,
                'contrast': contrast,
                'dominant_patterns': dominant_patterns
            }
    # Print and save texture properties per region.
    regions = list(REGION_COLORS.keys())
    species_list = list(results.keys())
    for region in regions:
        print(f"\nTexture properties for {region}:")
        headers = ["Property"] + species_list
        table_data = []
        properties = ["uniformity", "entropy", "contrast"]
        for prop in properties:
            row = [prop]
            for species in species_list:
                if region in results[species] and prop in results[species][region]:
                    row.append(f"{results[species][region][prop]:.4f}")
                else:
                    row.append("N/A")
            table_data.append(row)
        import pandas as pd
        df = pd.DataFrame(table_data, columns=headers)
        print(df)
        with open(os.path.join(RESULT_DIR, f'{region}_texture_properties.txt'), 'w') as f:
            f.write(f"Texture properties for {region}:\n")
            f.write(df.to_string())
    return results


def main():
    """Main function to run the texture analysis with organized outputs."""
    print("Starting bird species texture analysis using LBP...")
    global RESULT_DIR  # Declare global before using it
    # Create a unique run folder using the current timestamp.
    run_folder = os.path.join(RESULT_DIR, f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_folder, exist_ok=True)
    RESULT_DIR = run_folder  # Update the global output folder

    # Get image paths for each species.
    species_paths = {}
    for species_name in SPECIES:
        species_paths[species_name] = get_image_paths(species_name)
        print(f"Found {len(species_paths[species_name])} images for {species_name}")

    # Create LBP visualization examples (all species in one image).
    print("\nCreating LBP visualization examples...")
    visualize_lbp_patterns(species_paths)

    # Create region-specific visualizations for each species.
    visualize_region_lbp(species_paths)

    # Experiment with different LBP parameters on a sample image for one species.
    sample_species = list(species_paths.keys())[0]
    if species_paths[sample_species]:
        sample_img, sample_seg = species_paths[sample_species][0]
        sample_region = list(REGION_COLORS.keys())[0]
        print(f"\nExperimenting with different LBP parameters for {sample_species} - region: {sample_region}...")
        experiment_with_parameters(sample_img, sample_seg, sample_region, sample_species, RESULT_DIR)
    else:
        print("No sample image found for parameter experiment.")

    # Analyze textures for each species.
    species_data = {}
    for species_name in SPECIES:
        species_data[species_name] = analyze_species_texture(species_name)

    # Compare LBP features between species.
    print("\nComparing LBP features between species...")
    distances = visualize_lbp_comparison(species_data, 'lbp_histogram_comparison.png')
    print("\nChi-Square Distances between species:")
    for key, distance in distances.items():
        print(f"  {key}: {distance:.4f}")

    # Detailed texture analysis for each region.
    print("\nAnalyzing texture properties for individual regions...")
    texture_properties = analyze_texture_properties(species_data)

    # Build dataset and evaluate classification.
    print("\nBuilding dataset for classification...")
    X, y, region_info = build_feature_dataset(species_data)
    if len(set(y)) >= 2:
        print("\nEvaluating classification performance...")
        results = evaluate_classification(X, y, region_info)
        summary_path = os.path.join(RESULT_DIR, 'analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Bird Species Texture Analysis Summary\n")
            f.write("====================================\n\n")
            f.write("LBP Parameters:\n")
            f.write(f"  Radius: {RADIUS}\n")
            f.write(f"  Points: {N_POINTS}\n")
            f.write(f"  Method: {METHOD}\n\n")
            f.write("Chi-Square Distances between Species:\n")
            for key, distance in distances.items():
                f.write(f"  {key}: {distance:.4f}\n")
            f.write("\nClassification Results:\n")
            for region, metrics in results.items():
                f.write(f"  {region}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.4f}\n")
    else:
        print("Insufficient data for classification evaluation")

    print("\nAnalysis complete! Results saved to:", RESULT_DIR)


if __name__ == "__main__":
    main()
