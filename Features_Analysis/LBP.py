import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

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
# LBP PARAMETERS \
###############################################################################

RADIUS = 1  # Radius defines the size of the neighborhood
N_POINTS = 8 * RADIUS  # Number of points in the neighborhood
METHOD = 'uniform'  # Uniform pattern to reduce dimensionality

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


def get_region_masks(segmentation, debug=False):
    """
    Extract separate masks for each region in the segmentation image.
    Returns a dictionary of region masks and their statistics.
    Uses extract_region_mask from config.py for each region.
    """
    region_masks = {}
    region_stats = {}

    for region_name in REGION_COLORS:
        # Use the new config function to get the binary mask for this region
        mask = extract_region_mask(segmentation, region_name)
        region_masks[region_name] = mask

        # Calculate region statistics (center and bounding box)
        pixels = cv2.countNonZero(mask)
        if pixels > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    region_stats[region_name] = {
                        "pixels": pixels,
                        "center": (cx, cy),
                        "bbox": (x, y, w, h)
                    }

    if debug:
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

                # Mark center and bounding box
                cx, cy = region_stats[region_name]["center"]
                cv2.circle(debug_img, (cx, cy), 5, (255, 255, 255), -1)
                x, y, w, h = region_stats[region_name]["bbox"]
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (128, 128, 128), 2)

        return region_masks, region_stats, debug_img

    return region_masks, region_stats


def extract_lbp_features(image, mask=None, region_stats=None, region_name=None, debug=False):
    """
    Extract LBP features from a grayscale image, optionally with a mask.
    If region_stats is provided, LBP patterns are visualized around the region center.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply LBP to the entire image
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)

    # Create histogram
    n_bins = int(N_POINTS * (N_POINTS - 1) + 3) if METHOD == 'uniform' else 2 ** N_POINTS
    # n_bins = (
    print(int(lbp.max() + 1))
    print(int(N_POINTS * (N_POINTS - 1) + 3) if METHOD == 'uniform' else 2 ** N_POINTS)

    # If we have a mask, only consider pixels within the mask
    if mask is not None:
        # Create masked version of LBP
        masked_lbp = lbp.copy()
        masked_lbp[mask == 0] = 0

        # Create histogram from masked region
        if cv2.countNonZero(mask) > 0:
            hist, _ = np.histogram(lbp[mask > 0], bins=n_bins, range=(0, n_bins), density=True)
        else:
            hist = np.zeros(n_bins)
    else:
        masked_lbp = lbp
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    # If in debug mode, create visualizations
    debug_imgs = {}
    if debug and region_stats and region_name in region_stats:
        # Get region center
        cx, cy = region_stats[region_name]["center"]

        # Create debug images
        center_img = image.copy()
        neighborhood_img = image.copy()
        points_img = image.copy()

        # Mark center pixel
        cv2.circle(center_img, (cx, cy), 3, (0, 0, 255), -1)

        # Draw neighborhood circle
        cv2.circle(neighborhood_img, (cx, cy), RADIUS, (0, 255, 0), 1)

        # Draw sampling points
        for i in range(N_POINTS):
            angle = 2 * np.pi * i / N_POINTS
            px = int(cx + RADIUS * np.cos(angle))
            py = int(cy + RADIUS * np.sin(angle))
            cv2.circle(points_img, (px, py), 2, (255, 0, 0), -1)

        # Add center point to points image too for reference
        cv2.circle(points_img, (cx, cy), 3, (0, 0, 255), -1)

        debug_imgs = {
            "center": center_img,
            "neighborhood": neighborhood_img,
            "points": points_img,
            "masked_lbp": masked_lbp
        }

    if debug:
        return lbp, masked_lbp, hist, debug_imgs

    return lbp, masked_lbp, hist


def visualize_region_lbp_details(image_path, seg_path, region_name, output_dir):
    """Generate detailed visualization of LBP application on a specific region"""
    img = cv2.imread(image_path)
    seg = cv2.imread(seg_path)

    if img is None or seg is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Get region masks and statistics
    region_masks, region_stats, debug_segmentation = get_region_masks(seg, debug=True)

    if region_name not in region_masks or region_name not in region_stats:
        print(f"Region {region_name} not found in segmentation mask")
        return None

    # Extract LBP features with debugging enabled
    _, masked_lbp, hist, debug_imgs = extract_lbp_features(
        img, region_masks[region_name], region_stats, region_name, debug=True
    )

    # Create visualization of the region and its LBP patterns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image with region mask overlay
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # Segmentation with detected regions
    axes[0, 1].imshow(cv2.cvtColor(debug_segmentation, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Segmentation with Region Centers")
    axes[0, 1].axis('off')

    # Center pixel visualization
    axes[0, 2].imshow(cv2.cvtColor(debug_imgs["center"], cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"{region_name} Region Center")
    axes[0, 2].axis('off')

    # Neighborhood visualization
    axes[1, 0].imshow(cv2.cvtColor(debug_imgs["neighborhood"], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"LBP Neighborhood (R={RADIUS})")
    axes[1, 0].axis('off')

    # Sampling points visualization
    axes[1, 1].imshow(cv2.cvtColor(debug_imgs["points"], cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"LBP Sampling Points (P={N_POINTS})")
    axes[1, 1].axis('off')

    # LBP visualization
    axes[1, 2].imshow(masked_lbp, cmap='viridis')
    axes[1, 2].set_title(f"{region_name} LBP Pattern")
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save the visualization
    filename = f"detailed_{os.path.basename(image_path).split('.')[0]}_{region_name}_lbp.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

    # Also create a histogram visualization
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(hist)), hist)
    plt.title(f"LBP Histogram for {region_name}")
    plt.xlabel("LBP Pattern")
    plt.ylabel("Normalized Frequency")
    plt.savefig(
        os.path.join(output_dir, f"histogram_{os.path.basename(image_path).split('.')[0]}_{region_name}_lbp.png"))
    plt.close()

    return hist


def analyze_species_texture(species_name, limit=S, debug=False):
    """Analyze LBP features for all images of a species with enhanced debugging"""
    print(f"Analyzing {species_name} textures...")
    paths = get_image_paths(species_name)[:limit]

    if not paths:
        print(f"No images found for {species_name}")
        return None

    region_features = {region: [] for region in REGION_COLORS}
    debug_outputs = {region: [] for region in REGION_COLORS}

    for i, (img_path, seg_path) in enumerate(paths):
        print(f"  Processing image {i + 1}/{len(paths)}: {os.path.basename(img_path)}")
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)

        if img is None or seg is None:
            print(f"  Warning: Could not load {img_path} or {seg_path}. Skipping.")
            continue

        # Get all region masks for this image
        region_masks, region_stats = get_region_masks(seg)

        # Create a debug-friendly visualization of all regions
        if debug and i == 0:  # For the first image of each species, create detailed visualizations
            for region_name in REGION_COLORS:
                if region_name in region_masks and region_name in region_stats:
                    region_debug_dir = os.path.join(DEBUG_DIR, f"{species_name}_{region_name}")
                    os.makedirs(region_debug_dir, exist_ok=True)
                    visualize_region_lbp_details(img_path, seg_path, region_name, region_debug_dir)

        # Process each region
        for region_name in REGION_COLORS:
            if region_name in region_masks and cv2.countNonZero(region_masks[region_name]) > 0:
                # Extract LBP features for this region
                _, _, hist = extract_lbp_features(img, region_masks[region_name])
                region_features[region_name].append(hist)
                debug_outputs[region_name].append((img_path, seg_path))
            else:
                print(f"  Warning: Region {region_name} not found in {os.path.basename(img_path)}")

    # For debugging, create a summary of regions found across all images
    if debug:
        for region_name, paths_list in debug_outputs.items():
            coverage = len(paths_list) / len(paths) if paths else 0
            print(f"  Region {region_name}: Found in {len(paths_list)}/{len(paths)} images ({coverage:.1%})")

    return region_features, debug_outputs


def visualize_lbp_comparison(species_data, output_filename):
    """Visualize and compare LBP histograms for each region between species with error bars and dynamic axis limits."""
    # Filter out regions that have no data
    active_regions = []
    for region_name in REGION_COLORS:
        has_data = False
        for species_name, regions in species_data.items():
            if region_name in regions[0] and regions[0][region_name]:
                has_data = True
                break
        if has_data:
            active_regions.append(region_name)

    if not active_regions:
        print("No regions with data found for comparison.")
        return {}

    # Create summary plot for each active region
    fig, axes = plt.subplots(len(active_regions), 1, figsize=(14, 4 * len(active_regions)))
    if len(active_regions) == 1:
        axes = [axes]

    distances = {}
    for idx, region_name in enumerate(active_regions):
        ax = axes[idx]

        # Determine which species have data for this region
        species_with_region = []
        avg_hists = {}
        std_hists = {}
        for species_name, (regions, _) in species_data.items():
            if region_name in regions and regions[region_name]:
                species_with_region.append(species_name)
                # Compute average and standard deviation of histograms
                species_hists = regions[region_name]
                avg_hists[species_name] = np.mean(species_hists, axis=0)
                std_hists[species_name] = np.std(species_hists, axis=0)

        if len(species_with_region) < 2:
            ax.text(0.5, 0.5, f"Insufficient data for comparison: {region_name}",
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title(f"Region: {region_name} - Insufficient Data")
            continue

        x = np.arange(len(next(iter(avg_hists.values()))))
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        for (species_name, avg_hist), color in zip(avg_hists.items(), colors):
            std_hist = std_hists[species_name]
            ax.bar(x, avg_hist, yerr=std_hist, capsize=5, alpha=0.6, label=species_name, color=color)

        # Dynamically set the x-axis limit based on non-zero bins
        max_bin = 0
        for hist in avg_hists.values():
            nonzero_bins = np.nonzero(hist)[0]
            if len(nonzero_bins) > 0:
                max_bin = max(max_bin, nonzero_bins.max() + 1)
        if max_bin > 0:
            ax.set_xlim([0, max_bin])
        else:
            ax.set_xlim([0, len(x)])

        ax.set_xlabel("LBP Bin")
        ax.set_ylabel("Normalized Frequency")

        # Calculate and display chi-square distances between species
        if len(avg_hists) >= 2:
            species_list = list(avg_hists.keys())
            distance_text = []
            for i in range(len(species_list)):
                for j in range(i + 1, len(species_list)):
                    sp1, sp2 = species_list[i], species_list[j]
                    distance = chi_square_distance(avg_hists[sp1], avg_hists[sp2])
                    key = f"{region_name}_{sp1}_vs_{sp2}"
                    distances[key] = distance
                    distance_text.append(f"{sp1} vs {sp2}: {distance:.4f}")
            ax.set_title(f"Region: {region_name}\nChi-Square Distances: " + ", ".join(distance_text))
        else:
            ax.set_title(f"Region: {region_name}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, output_filename))
    plt.close()

    # Also create a more detailed barplot with error bars for each region
    for region_name in active_regions:
        species_with_region = []
        avg_hists = {}
        std_hists = {}
        for species_name, (regions, _) in species_data.items():
            if region_name in regions and regions[region_name]:
                species_with_region.append(species_name)
                avg_hists[species_name] = np.mean(regions[region_name], axis=0)
                std_hists[species_name] = np.std(regions[region_name], axis=0)

        if len(species_with_region) >= 2:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(next(iter(avg_hists.values()))))
            bar_width = 0.8 / len(species_with_region)
            for i, species_name in enumerate(species_with_region):
                offset = i * bar_width - (len(species_with_region) - 1) * bar_width / 2
                ax.bar(x + offset, avg_hists[species_name], width=bar_width,
                       yerr=std_hists[species_name], capsize=5, label=species_name)
            # Adjust x-axis based on non-zero bins
            max_bin = 0
            for hist in avg_hists.values():
                nonzero_bins = np.nonzero(hist)[0]
                if len(nonzero_bins) > 0:
                    max_bin = max(max_bin, nonzero_bins.max() + 1)
            if max_bin > 0:
                ax.set_xlim([0, max_bin])
            else:
                ax.set_xlim([0, len(x)])
            ax.set_xlabel("LBP Pattern")
            ax.set_ylabel("Normalized Frequency")
            ax.set_title(f"LBP Pattern Distribution for {region_name} Region")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(COMPARISON_DIR, f"detailed_{region_name}_comparison.png"))
            plt.close()

    return distances


def chi_square_distance(hist1, hist2):
    """Calculate chi-square distance between two histograms"""
    eps = 1e-10  # Prevent division by zero
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))


def analyze_texture_properties(species_data):
    """Extract meaningful texture properties from LBP histograms"""
    results = {}
    for species_name, (regions, _) in species_data.items():
        results[species_name] = {}
        for region_name, histograms in regions.items():
            if not histograms:
                continue
            avg_hist = np.mean(histograms, axis=0)

            # Calculate various texture metrics
            uniformity = np.sum(avg_hist ** 2)
            entropy = -np.sum(avg_hist * np.log2(avg_hist + 1e-10))
            contrast = np.var(np.arange(len(avg_hist)) * avg_hist)
            dominant_patterns = np.argsort(avg_hist)[-3:][::-1]

            # Calculate additional metrics
            energy = np.sqrt(uniformity)
            smoothness = 1 - (1 / (1 + np.sum(avg_hist * np.arange(len(avg_hist)))))

            results[species_name][region_name] = {
                'uniformity': uniformity,
                'entropy': entropy,
                'contrast': contrast,
                'energy': energy,
                'smoothness': smoothness,
                'dominant_patterns': dominant_patterns
            }

    # Create detailed texture property visualizations
    for region_name in REGION_COLORS:
        # Check if any species has data for this region
        has_data = False
        for species_name in results:
            if region_name in results[species_name]:
                has_data = True
                break

        if not has_data:
            continue

        # Prepare data for visualization
        metrics = ['uniformity', 'entropy', 'contrast', 'energy', 'smoothness']
        species_list = [s for s in results if region_name in results[s]]

        if not species_list:
            continue

        # Create bar chart comparison for each texture property
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            metric_values = [results[species][region_name][metric] if region_name in results[species] else 0
                             for species in species_list]

            axes[i].bar(species_list, metric_values,
                        color=['blue', 'orange', 'green', 'red', 'purple'][:len(species_list)])
            axes[i].set_title(f"{metric.capitalize()} Comparison for {region_name}")
            axes[i].set_ylabel(metric.capitalize())

            # Add values on top of bars
            for j, value in enumerate(metric_values):
                axes[i].text(j, value * 1.05, f"{value:.4f}", ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(REGION_DIR, f"{region_name}_texture_metrics.png"))
        plt.close()

    # Print and save texture properties per region
    regions = list(REGION_COLORS.keys())
    species_list = list(results.keys())

    for region in regions:
        # Check if we have data for this region
        has_data = False
        for species in species_list:
            if region in results[species]:
                has_data = True
                break

        if not has_data:
            continue

        print(f"\nTexture properties for {region}:")
        properties = ["uniformity", "entropy", "contrast", "energy", "smoothness"]

        # Create DataFrame for easier display and export
        data = []
        for prop in properties:
            row = [prop]
            for species in species_list:
                if region in results[species] and prop in results[species][region]:
                    row.append(f"{results[species][region][prop]:.4f}")
                else:
                    row.append("N/A")
            data.append(row)

        headers = ["Property"] + species_list
        df = pd.DataFrame(data, columns=headers)
        print(df)

        # Save to file
        with open(os.path.join(REGION_DIR, f'{region}_texture_properties.txt'), 'w') as f:
            f.write(f"Texture properties for {region}:\n")
            f.write(df.to_string())

        # Also save as CSV for easier analysis
        df.to_csv(os.path.join(REGION_DIR, f'{region}_texture_properties.csv'), index=False)

    return results


def build_feature_dataset(species_data):
    """Build dataset for classification from the extracted features"""
    X, y, region_info = [], [], []

    for species_name, (regions, _) in species_data.items():
        for region_name, histograms in regions.items():
            for hist in histograms:
                X.append(hist)
                y.append(species_name)
                region_info.append(region_name)

    return np.array(X), np.array(y), np.array(region_info)


def evaluate_classification(X, y, region_info):
    """Evaluate classification performance for each region separately"""
    results = {}

    # Get unique regions
    regions = np.unique(region_info)

    for region in regions:
        # Filter data for this region
        region_mask = (region_info == region)
        if np.sum(region_mask) < 10:  # Skip regions with too few samples
            print(f"  Skipping {region} - insufficient data ({np.sum(region_mask)} samples)")
            continue

        X_region = X[region_mask]
        y_region = y[region_mask]

        # Check if we have enough samples per class
        classes, counts = np.unique(y_region, return_counts=True)
        if len(classes) < 2:
            print(f"  Skipping {region} - only one class available")
            continue
        if np.min(counts) < 3:
            print(f"  Skipping {region} - insufficient samples for some classes")
            continue

        print(f"  Evaluating {region} region: {len(X_region)} samples, {len(classes)} classes")

        # Create classifier
        clf = SVC(kernel='rbf', probability=True)

        # If we have enough samples, use train/test split
        if len(X_region) >= 15 and np.min(counts) >= 5:
            X_train, X_test, y_train, y_test = train_test_split(
                X_region, y_region, test_size=0.3, stratify=y_region, random_state=42
            )

            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)

            # Save results
            results[region] = {'accuracy': accuracy}
            print(f"    Test accuracy: {accuracy:.4f}")
        else:
            # Use cross-validation for small datasets
            try:
                cv_scores = cross_val_score(clf, X_region, y_region, cv=min(5, np.min(counts)))
                results[region] = {'cv_accuracy': np.mean(cv_scores)}
                print(f"    Cross-validation accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
            except ValueError as e:
                print(f"    Error in cross-validation: {e}")
                continue

    return results


def visualize_lbp_patterns(species_paths):
    """Create visual examples of LBP patterns on a sample from each species"""
    # Create a figure to show examples
    num_species = len(species_paths)
    fig, axes = plt.subplots(num_species, 3, figsize=(15, 5 * num_species))

    # If there's only one species, wrap axes in a list for consistent indexing
    if num_species == 1:
        axes = [axes]

    for i, (species_name, paths) in enumerate(species_paths.items()):
        if not paths:
            continue

        # Use the first image of each species
        img_path, seg_path = paths[0]
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)

        if img is None or seg is None:
            print(f"Error loading images for {species_name}")
            continue

        # Show original image
        axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"{species_name} - Original")
        axes[i, 0].axis('off')

        # Show segmentation
        axes[i, 1].imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"{species_name} - Segmentation")
        axes[i, 1].axis('off')

        # Convert image to grayscale for LBP
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply LBP
        lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)

        # Show LBP
        axes[i, 2].imshow(lbp, cmap='viridis')
        axes[i, 2].set_title(f"{species_name} - LBP")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "lbp_pattern_examples.png"))
    plt.close()


def visualize_region_lbp(species_paths):
    """Create visualizations of LBP patterns for each region of interest"""
    for species_name, paths in species_paths.items():
        if not paths:
            continue

        # Use the first image of each species
        img_path, seg_path = paths[0]
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)

        if img is None or seg is None:
            print(f"Error loading images for {species_name}")
            continue

        # Get region masks
        region_masks, region_stats, debug_img = get_region_masks(seg, debug=True)

        # Create a debug directory for this species
        species_debug_dir = os.path.join(DEBUG_DIR, species_name)
        os.makedirs(species_debug_dir, exist_ok=True)

        # Save the segmented regions for debugging
        cv2.imwrite(os.path.join(species_debug_dir, "segmented_regions.png"), debug_img)

        # For each region, visualize LBP centered on region's center pixel
        for region_name in REGION_COLORS:
            if region_name in region_masks and region_name in region_stats:
                # Get region center coordinates
                cx, cy = region_stats[region_name]["center"]

                # Create visualization of LBP application on this region
                visualize_region_lbp_details(img_path, seg_path, region_name, species_debug_dir)

                # Extract a small patch around the center for detailed analysis
                # This focuses on the central pixel of the region (especially wingtip)
                patch_size = 21  # Small odd number to have a clear center
                half_size = patch_size // 2

                # Create patch bounds checking for image boundaries
                x_min = max(0, cx - half_size)
                x_max = min(img.shape[1], cx + half_size + 1)
                y_min = max(0, cy - half_size)
                y_max = min(img.shape[0], cy + half_size + 1)

                # Extract patch
                patch = img[y_min:y_max, x_min:x_max]

                if patch.size == 0:
                    continue

                # Convert to grayscale
                gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                # Apply LBP
                lbp_patch = local_binary_pattern(gray_patch, N_POINTS, RADIUS, METHOD)

                # Visualize the central patch and its LBP
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

                ax1.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                ax1.set_title(f"{region_name} Center Patch")
                ax1.axis('off')

                ax2.imshow(lbp_patch, cmap='viridis')
                ax2.set_title(f"{region_name} Center LBP")
                ax2.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(species_debug_dir, f"{region_name}_center_patch_lbp.png"))
                plt.close()

                # Save the patch histogram for texture analysis
                n_bins = int(N_POINTS * (N_POINTS - 1) + 3) if METHOD == 'uniform' else 2 ** N_POINTS
                hist, _ = np.histogram(lbp_patch, bins=n_bins, range=(0, n_bins), density=True)

                plt.figure(figsize=(10, 5))
                plt.bar(range(len(hist)), hist)
                plt.title(f"{region_name} Center Patch LBP Histogram")
                plt.xlabel("LBP Pattern")
                plt.ylabel("Normalized Frequency")
                plt.savefig(os.path.join(species_debug_dir, f"{region_name}_center_histogram.png"))
                plt.close()


def experiment_with_parameters(img_path, seg_path, region_name, species_name, output_dir):
    """
    Test different radius and points combinations and visualize the effect on LBP.
    Focus on the region center rather than the image center.
    """
    img = cv2.imread(img_path)
    seg = cv2.imread(seg_path)

    if img is None or seg is None:
        print(f"Error loading images: {img_path} or {seg_path}")
        return

    # Get region mask and statistics
    region_masks, region_stats = get_region_masks(seg)

    if region_name not in region_masks or region_name not in region_stats:
        print(f"Region {region_name} not found in segmentation mask")
        return

    region_mask = region_masks[region_name]
    cx, cy = region_stats[region_name]["center"]

    # Extract patch around center for focused analysis
    patch_size = 41  # Larger than before to show more context
    half_size = patch_size // 2

    # Create patch bounds checking for image boundaries
    x_min = max(0, cx - half_size)
    x_max = min(img.shape[1], cx + half_size + 1)
    y_min = max(0, cy - half_size)
    y_max = min(img.shape[0], cy + half_size + 1)

    # Extract patch
    patch = img[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        print(f"Could not extract patch for {region_name}")
        return

    # Convert to grayscale
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Parameters to test
    radii = [1, 2, 3, 5, 8]
    points_list = [8, 16, 24]

    # Create a grid of LBP visualizations with different parameters
    fig, axes = plt.subplots(len(radii), len(points_list), figsize=(4 * len(points_list), 4 * len(radii)))

    for i, radius in enumerate(radii):
        for j, points in enumerate(points_list):
            # Calculate LBP with these parameters
            lbp = local_binary_pattern(gray_patch, points, radius, 'uniform')

            # Display in grid
            axes[i, j].imshow(lbp, cmap='viridis')
            axes[i, j].set_title(f'R={radius}, P={points}')
            axes[i, j].axis('off')

    plt.tight_layout()
    exp_dir = os.path.join(output_dir, "parameter_experiments")
    os.makedirs(exp_dir, exist_ok=True)
    exp_filename = f"{species_name}_{region_name}_parameter_experiment.png"
    plt.savefig(os.path.join(exp_dir, exp_filename))
    plt.close()

    # Visualize LBP neighborhoods centered on the region's center
    for radius in [1, 2, 3]:
        points = 8 * radius

        # Create a zoomed-in version of the patch
        zoomed_patch = cv2.resize(patch, (patch.shape[1] * 3, patch.shape[0] * 3),
                                  interpolation=cv2.INTER_NEAREST)
        center_y, center_x = zoomed_patch.shape[0] // 2, zoomed_patch.shape[1] // 2

        # Create visualization of the center point, neighborhood, and sampling points
        center_img = zoomed_patch.copy()
        neighborhood_img = zoomed_patch.copy()
        points_img = zoomed_patch.copy()

        # Scale radius for visualization
        vis_radius = radius * 3

        # Mark center pixel
        cv2.circle(center_img, (center_x, center_y), 3, (0, 0, 255), -1)

        # Draw neighborhood circle
        cv2.circle(neighborhood_img, (center_x, center_y), vis_radius, (0, 255, 0), 1)

        # Draw sampling points
        for k in range(points):
            angle = 2 * np.pi * k / points
            px = int(center_x + vis_radius * np.cos(angle))
            py = int(center_y + vis_radius * np.sin(angle))
            cv2.circle(points_img, (px, py), 2, (255, 0, 0), -1)

        # Add center point to sampling points image for reference
        cv2.circle(points_img, (center_x, center_y), 3, (0, 0, 255), -1)

        # Create combined visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'{region_name} Center Pixel')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(neighborhood_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Radius = {radius}')
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(points_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'{points} Sampling Points')
        axes[2].axis('off')

        plt.tight_layout()
        neigh_filename = f"{species_name}_{region_name}_r{radius}_p{points}_neighborhood.png"
        plt.savefig(os.path.join(exp_dir, neigh_filename))
        plt.close()



def main():
    """Main function to run the texture analysis with organized outputs."""
    print("Starting bird species texture analysis using LBP...")
    global RESULT_DIR, DEBUG_DIR, REGION_DIR, COMPARISON_DIR  # Declare globals

    # Create a unique run folder using the current timestamp
    run_folder = f"R{RADIUS}_P{N_POINTS}_M{METHOD}"
    RESULT_DIR = os.path.join("Outputs/LBP_Analysis", run_folder)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Create all necessary directories
    DEBUG_DIR = os.path.join(RESULT_DIR, "debug_outputs")
    REGION_DIR = os.path.join(RESULT_DIR, "region_analysis")
    COMPARISON_DIR = os.path.join(RESULT_DIR, "species_comparison")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(REGION_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    # Save configuration to a file for reference
    # with open(os.path.join(RESULT_DIR, 'config.txt'), 'w') as f:
        # (Additional config details here)

    # Get image paths for each species and continue with your analysis...
    species_paths = {}
    for species_name in SPECIES:
        species_paths[species_name] = get_image_paths(species_name)
        print(f"Found {len(species_paths[species_name])} images for {species_name}")

    # Create LBP visualization examples (all species in one image)
    print("\nCreating LBP visualization examples...")
    visualize_lbp_patterns(species_paths)

    # Create region-specific visualizations for each species
    print("\nCreating region-specific LBP visualizations...")
    visualize_region_lbp(species_paths)

    # Experiment with different LBP parameters on a sample image for one species
    sample_species = list(species_paths.keys())[0]
    if species_paths[sample_species]:
        sample_img, sample_seg = species_paths[sample_species][0]
        # Focus on wingtip region for parameter experiments
        sample_region = "wingtip"
        print(f"\nExperimenting with different LBP parameters for {sample_species} - region: {sample_region}...")
        experiment_with_parameters(sample_img, sample_seg, sample_region, sample_species, RESULT_DIR)
    else:
        print("No sample image found for parameter experiment.")

    # Analyze textures for each species with focus on region centers
    species_data = {}
    for species_name in SPECIES:
        print(f"\nProcessing {species_name}...")
        species_data[species_name] = analyze_species_texture(species_name, debug=True)

    # Compare LBP features between species
    print("\nComparing LBP features between species...")
    distances = visualize_lbp_comparison(species_data, 'lbp_histogram_comparison.png')
    print("\nChi-Square Distances between species:")
    for key, distance in distances.items():
        print(f"  {key}: {distance:.4f}")

    # Detailed texture analysis for each region
    print("\nAnalyzing texture properties for individual regions...")
    texture_properties = analyze_texture_properties(species_data)

    # Build dataset and evaluate classification
    print("\nBuilding dataset for classification...")
    X, y, region_info = build_feature_dataset(species_data)

    # Check if we have enough data for classification
    if len(np.unique(y)) >= 2:
        print("\nEvaluating classification performance...")
        results = evaluate_classification(X, y, region_info)

        # Create summary report
        summary_path = os.path.join(RESULT_DIR, 'analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Bird Species Texture Analysis Summary\n")
            f.write("====================================\n\n")
            f.write("LBP Parameters:\n")
            f.write(f"  Radius: {RADIUS}\n")
            f.write(f"  Points: {N_POINTS}\n")
            f.write(f"  Method: {METHOD}\n\n")

            f.write("Species Data Summary:\n")
            for species_name, (regions, _) in species_data.items():
                f.write(f"  {species_name}:\n")
                for region_name, histograms in regions.items():
                    f.write(f"    {region_name}: {len(histograms)} samples\n")

            f.write("\nChi-Square Distances between Species:\n")
            for key, distance in distances.items():
                f.write(f"  {key}: {distance:.4f}\n")

            f.write("\nClassification Results:\n")
            for region, metrics in results.items():
                f.write(f"  {region}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.4f}\n")

            # Add texture property summary
            f.write("\nTexture Property Summary:\n")
            for species, region_props in texture_properties.items():
                f.write(f"  {species}:\n")
                for region, props in region_props.items():
                    f.write(f"    {region}:\n")
                    for prop, value in props.items():
                        if prop != 'dominant_patterns':
                            f.write(f"      {prop}: {value:.4f}\n")
                        else:
                            f.write(f"      {prop}: {value}\n")
    else:
        print("Insufficient data for classification evaluation")

    print("\nAnalysis complete! Results saved to:", RESULT_DIR)


if __name__ == "__main__":
    main()