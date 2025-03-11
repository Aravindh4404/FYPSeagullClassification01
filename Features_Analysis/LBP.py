import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# Import your configuration
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
# LBP PARAMETERS
###############################################################################
RADIUS = 1  # Size of the neighborhood
N_POINTS = 8 * RADIUS  # Number of points in the neighborhood
METHOD = 'default'  # Use uniform patterns to reduce dimensionality


###############################################################################
# FUNCTION DEFINITIONS
###############################################################################

def get_image_paths(species):
    """Get paired original and segmentation image paths for a species."""
    img_dir = SPECIES[species]["img_dir"]
    seg_dir = SPECIES[species]["seg_dir"]
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    seg_files = sorted([f for f in os.listdir(seg_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
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
    Uses extract_region_mask from config.py for each region.
    """
    region_masks = {}
    region_stats = {}
    for region_name in REGION_COLORS:
        mask = extract_region_mask(segmentation, region_name)
        region_masks[region_name] = mask
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
                    region_stats[region_name] = {"pixels": pixels,
                                                 "center": (cx, cy),
                                                 "bbox": (x, y, w, h)}
    if debug:
        debug_img = np.zeros_like(segmentation)
        for region_name, mask in region_masks.items():
            if region_name in region_stats:
                color = REGION_COLORS[region_name]
                region_pixels = cv2.bitwise_and(np.full_like(segmentation, color),
                                                np.full_like(segmentation, color),
                                                mask=mask)
                debug_img = cv2.add(debug_img, region_pixels)
                cx, cy = region_stats[region_name]["center"]
                cv2.circle(debug_img, (cx, cy), 5, (255, 255, 255), -1)
                x, y, w, h = region_stats[region_name]["bbox"]
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (128, 128, 128), 2)
        return region_masks, region_stats, debug_img
    return region_masks, region_stats


def extract_lbp_features(image, mask=None, region_stats=None, region_name=None, debug=False):
    """
    Extract LBP features from a grayscale image, optionally using a mask.
    If debug is True and region_stats provided, return debug images.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)
    n_bins = int(N_POINTS * (N_POINTS - 1) + 3) if METHOD == 'uniform' else 2 ** N_POINTS

    if mask is not None:
        masked_lbp = lbp.copy()
        masked_lbp[mask == 0] = 0
        if cv2.countNonZero(mask) > 0:
            hist, _ = np.histogram(lbp[mask > 0], bins=n_bins, range=(0, n_bins), density=True)
        else:
            hist = np.zeros(n_bins)
    else:
        masked_lbp = lbp
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    debug_imgs = {}
    if debug and region_stats and region_name in region_stats:
        cx, cy = region_stats[region_name]["center"]
        center_img = image.copy()
        neighborhood_img = image.copy()
        points_img = image.copy()
        cv2.circle(center_img, (cx, cy), 3, (0, 0, 255), -1)
        cv2.circle(neighborhood_img, (cx, cy), RADIUS, (0, 255, 0), 1)
        for i in range(N_POINTS):
            angle = 2 * np.pi * i / N_POINTS
            px = int(cx + RADIUS * np.cos(angle))
            py = int(cy + RADIUS * np.sin(angle))
            cv2.circle(points_img, (px, py), 2, (255, 0, 0), -1)
        cv2.circle(points_img, (cx, cy), 3, (0, 0, 255), -1)
        debug_imgs = {"center": center_img,
                      "neighborhood": neighborhood_img,
                      "points": points_img,
                      "masked_lbp": masked_lbp}
    if debug:
        return lbp, masked_lbp, hist, debug_imgs
    return lbp, masked_lbp, hist


def visualize_region_lbp_details(image_path, seg_path, region_name, output_dir):
    """Generate detailed visualization of LBP application on a specific region."""
    img = cv2.imread(image_path)
    seg = cv2.imread(seg_path)
    if img is None or seg is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None
    region_masks, region_stats, debug_segmentation = get_region_masks(seg, debug=True)
    if region_name not in region_masks or region_name not in region_stats:
        print(f"Region {region_name} not found in segmentation mask")
        return None
    _, masked_lbp, hist, debug_imgs = extract_lbp_features(img, region_masks[region_name],
                                                           region_stats, region_name, debug=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(cv2.cvtColor(debug_segmentation, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Segmentation with Region Centers")
    axes[0, 1].axis('off')
    axes[0, 2].imshow(cv2.cvtColor(debug_imgs["center"], cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f"{region_name} Region Center")
    axes[0, 2].axis('off')
    axes[1, 0].imshow(cv2.cvtColor(debug_imgs["neighborhood"], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"LBP Neighborhood (R={RADIUS})")
    axes[1, 0].axis('off')
    axes[1, 1].imshow(cv2.cvtColor(debug_imgs["points"], cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"LBP Sampling Points (P={N_POINTS})")
    axes[1, 1].axis('off')
    axes[1, 2].imshow(masked_lbp, cmap='viridis')
    axes[1, 2].set_title(f"{region_name} LBP Pattern")
    axes[1, 2].axis('off')
    plt.tight_layout()
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
    # Optionally display the image:
    plt.figure()
    plt.imshow(masked_lbp, cmap='viridis')
    plt.title(f"{region_name} LBP Pattern (Displayed)")
    plt.axis('off')
    plt.show()
    return hist


def analyze_species_texture(species_name, limit=S, debug=False):
    """Analyze LBP features for all images of a species with enhanced debugging."""
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
        region_masks, region_stats = get_region_masks(seg)
        if debug and i == 0:
            for region_name in REGION_COLORS:
                if region_name in region_masks and region_name in region_stats:
                    region_debug_dir = os.path.join(DEBUG_DIR, f"{species_name}_{region_name}")
                    os.makedirs(region_debug_dir, exist_ok=True)
                    visualize_region_lbp_details(img_path, seg_path, region_name, region_debug_dir)
        for region_name in REGION_COLORS:
            if region_name in region_masks and cv2.countNonZero(region_masks[region_name]) > 0:
                _, _, hist = extract_lbp_features(img, region_masks[region_name])
                region_features[region_name].append(hist)
                debug_outputs[region_name].append((img_path, seg_path))
            else:
                print(f"  Warning: Region {region_name} not found in {os.path.basename(img_path)}")
    if debug:
        for region_name, paths_list in debug_outputs.items():
            coverage = len(paths_list) / len(paths) if paths else 0
            print(f"  Region {region_name}: Found in {len(paths_list)}/{len(paths)} images ({coverage:.1%})")
    return region_features, debug_outputs


def visualize_lbp_comparison(species_data, output_filename):
    """Visualize and compare LBP histograms for each region between species."""
    active_regions = []
    for region_name in REGION_COLORS:
        has_data = any(
            region_name in regions and regions[region_name] for species_name, regions in species_data.items())
        if has_data:
            active_regions.append(region_name)
    if not active_regions:
        print("No regions with data found for comparison.")
        return {}
    fig, axes = plt.subplots(len(active_regions), 1, figsize=(14, 4 * len(active_regions)))
    if len(active_regions) == 1:
        axes = [axes]
    distances = {}
    for idx, region_name in enumerate(active_regions):
        ax = axes[idx]
        species_with_region = []
        avg_hists = {}
        std_hists = {}
        for species_name, (regions, _) in species_data.items():
            if region_name in regions and regions[region_name]:
                species_with_region.append(species_name)
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
        max_bin = 0
        for hist in avg_hists.values():
            nonzero_bins = np.nonzero(hist)[0]
            if len(nonzero_bins) > 0:
                max_bin = max(max_bin, nonzero_bins.max() + 1)
        ax.set_xlim([0, max_bin] if max_bin > 0 else [0, len(x)])
        ax.set_xlabel("LBP Bin")
        ax.set_ylabel("Normalized Frequency")
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
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, output_filename))
    plt.close()
    return distances


def chi_square_distance(hist1, hist2):
    """Calculate chi-square distance between two histograms."""
    eps = 1e-10
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))


def analyze_texture_properties(species_data):
    """
    Extract texture properties from LBP histograms.
    Added correlation based on the histogram distribution.
    """
    results = {}
    for species_name, (regions, _) in species_data.items():
        results[species_name] = {}
        for region_name, histograms in regions.items():
            if not histograms:
                continue
            avg_hist = np.mean(histograms, axis=0)
            uniformity = np.sum(avg_hist ** 2)
            entropy = -np.sum(avg_hist * np.log2(avg_hist + 1e-10))
            contrast = np.sum((np.arange(len(avg_hist)) ** 2) * avg_hist)
            energy = np.sqrt(uniformity)
            smoothness = 1 - (1 / (1 + np.sum(avg_hist * np.arange(len(avg_hist)))))
            mean_val = np.sum(np.arange(len(avg_hist)) * avg_hist)
            std_dev = np.sqrt(np.sum(((np.arange(len(avg_hist)) - mean_val) ** 2) * avg_hist))
            # Using the provided formula for correlation (note: this may trivially equal 1)
            correlation = np.sum(((np.arange(len(avg_hist)) - mean_val) ** 2) * avg_hist) / (std_dev * std_dev + 1e-10)
            dominant_patterns = np.argsort(avg_hist)[-3:][::-1]
            results[species_name][region_name] = {
                'uniformity': uniformity,
                'entropy': entropy,
                'contrast': contrast,
                'energy': energy,
                'smoothness': smoothness,
                'correlation': correlation,
                'dominant_patterns': dominant_patterns
            }
    # Display and save texture property comparisons per region
    regions = list(REGION_COLORS.keys())
    species_list = list(results.keys())
    for region in regions:
        has_data = any(region in results[species] for species in species_list)
        if not has_data:
            continue
        print(f"\nTexture properties for {region}:")
        properties = ["uniformity", "entropy", "contrast", "energy", "smoothness", "correlation"]
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
        with open(os.path.join(REGION_DIR, f'{region}_texture_properties.txt'), 'w') as f:
            f.write(f"Texture properties for {region}:\n")
            f.write(df.to_string())
        df.to_csv(os.path.join(REGION_DIR, f'{region}_texture_properties.csv'), index=False)
    return results


def build_feature_dataset(species_data):
    """Build dataset for classification from the extracted features."""
    X, y, region_info = [], [], []
    for species_name, (regions, _) in species_data.items():
        for region_name, histograms in regions.items():
            for hist in histograms:
                X.append(hist)
                y.append(species_name)
                region_info.append(region_name)
    return np.array(X), np.array(y), np.array(region_info)


def evaluate_classification(X, y, region_info):
    """Evaluate classification performance for each region separately."""
    results = {}
    regions = np.unique(region_info)
    for region in regions:
        region_mask = (region_info == region)
        if np.sum(region_mask) < 10:
            print(f"  Skipping {region} - insufficient data ({np.sum(region_mask)} samples)")
            continue
        X_region = X[region_mask]
        y_region = y[region_mask]
        classes, counts = np.unique(y_region, return_counts=True)
        if len(classes) < 2 or np.min(counts) < 3:
            print(f"  Skipping {region} - insufficient classes or samples")
            continue
        print(f"  Evaluating {region} region: {len(X_region)} samples, {len(classes)} classes")
        clf = SVC(kernel='rbf', probability=True)
        if len(X_region) >= 15 and np.min(counts) >= 5:
            X_train, X_test, y_train, y_test = train_test_split(
                X_region, y_region, test_size=0.3, stratify=y_region, random_state=42
            )
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            results[region] = {'accuracy': accuracy}
            print(f"    Test accuracy: {accuracy:.4f}")
        else:
            try:
                cv_scores = cross_val_score(clf, X_region, y_region, cv=min(5, np.min(counts)))
                results[region] = {'cv_accuracy': np.mean(cv_scores)}
                print(f"    Cross-validation accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
            except ValueError as e:
                print(f"    Error in cross-validation: {e}")
    return results


def visualize_lbp_patterns(species_paths):
    """Create visual examples of LBP patterns on a sample from each species."""
    num_species = len(species_paths)
    fig, axes = plt.subplots(num_species, 3, figsize=(15, 5 * num_species))
    if num_species == 1:
        axes = [axes]
    for i, (species_name, paths) in enumerate(species_paths.items()):
        if not paths:
            continue
        img_path, seg_path = paths[0]
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is None or seg is None:
            print(f"Error loading images for {species_name}")
            continue
        axes[i][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f"{species_name} - Original")
        axes[i][0].axis('off')
        axes[i][1].imshow(cv2.cvtColor(seg, cv2.COLOR_BGR2RGB))
        axes[i][1].set_title(f"{species_name} - Segmentation")
        axes[i][1].axis('off')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)
        axes[i][2].imshow(lbp, cmap='viridis')
        axes[i][2].set_title(f"{species_name} - LBP")
        axes[i][2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "lbp_pattern_examples.png"))
    plt.close()


def visualize_texture_differences(species_data, output_dir):
    """
    Create visualizations to compare texture properties between species.
    Outputs heatmaps, radar charts, grouped bar charts, LBP histogram plots, and 3D scatter plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {}
    # Include correlation in the metrics list
    metrics_list = ['uniformity', 'entropy', 'contrast', 'energy', 'smoothness', 'correlation']
    for species_name, (regions, _) in species_data.items():
        all_metrics[species_name] = {}
        for region_name, histograms in regions.items():
            if not histograms:
                continue
            avg_hist = np.mean(histograms, axis=0)
            uniformity = np.sum(avg_hist ** 2)
            entropy = -np.sum(avg_hist * np.log2(avg_hist + 1e-10))
            contrast = np.sum((np.arange(len(avg_hist)) ** 2) * avg_hist)
            energy = np.sqrt(uniformity)
            smoothness = 1 - (1 / (1 + np.sum(avg_hist * np.arange(len(avg_hist)))))
            mean_val = np.sum(np.arange(len(avg_hist)) * avg_hist)
            std_dev = np.sqrt(np.sum(((np.arange(len(avg_hist)) - mean_val) ** 2) * avg_hist))
            correlation = np.sum(((np.arange(len(avg_hist)) - mean_val) ** 2) * avg_hist) / (std_dev * std_dev + 1e-10)
            all_metrics[species_name][region_name] = {
                'uniformity': uniformity,
                'entropy': entropy,
                'contrast': contrast,
                'energy': energy,
                'smoothness': smoothness,
                'correlation': correlation
            }
    # 1. Heatmaps for each metric
    heatmap_data = []
    for species_name in all_metrics:
        for region_name in all_metrics[species_name]:
            for metric in metrics_list:
                value = all_metrics[species_name][region_name][metric]
                heatmap_data.append({'Species': species_name, 'Region': region_name, 'Metric': metric, 'Value': value})
    if heatmap_data:
        df_heatmap = pd.DataFrame(heatmap_data)
        for metric in metrics_list:
            metric_df = df_heatmap[df_heatmap['Metric'] == metric]
            pivot_df = metric_df.pivot(index='Region', columns='Species', values='Value')
            plt.figure(figsize=(10, 8))
            blue_cmap = LinearSegmentedColormap.from_list('BlueGradient', ['lightblue', 'darkblue'])
            ax = sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap=blue_cmap, linewidths=0.5)
            plt.title(f'{metric.capitalize()} by Species and Region')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'heatmap_{metric}.png'))
            plt.close()
    # 2. Radar charts for each region
    all_species = list(all_metrics.keys())
    all_regions = set()
    for species in all_metrics:
        all_regions.update(all_metrics[species].keys())
    all_regions = list(all_regions)
    for region in all_regions:
        species_with_data = [s for s in all_species if region in all_metrics[s]]
        if not species_with_data:
            continue
        metrics_for_radar = metrics_list
        min_vals = {m: float('inf') for m in metrics_for_radar}
        max_vals = {m: float('-inf') for m in metrics_for_radar}
        for species in species_with_data:
            for metric in metrics_for_radar:
                val = all_metrics[species][region][metric]
                min_vals[metric] = min(min_vals[metric], val)
                max_vals[metric] = max(max_vals[metric], val)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
        angles += angles[:1]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_for_radar)
        colors = plt.cm.tab10(np.linspace(0, 1, len(species_with_data)))
        for i, species in enumerate(species_with_data):
            values = []
            for metric in metrics_for_radar:
                val = all_metrics[species][region][metric]
                min_val = min_vals[metric]
                max_val = max_vals[metric]
                range_val = max_val - min_val
                scaled_val = (val - min_val) / range_val if range_val != 0 else 0.5
                values.append(scaled_val)
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=species, alpha=0.8)
            ax.fill(angles, values, color=colors[i], alpha=0.1)
        ax.set_title(f'Texture Properties Comparison for {region}')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'radar_{region}_comparison.png'))
        plt.close()
    # 3. Grouped bar charts for direct comparisons
    for metric in metrics_list:
        bar_data = []
        for species in all_species:
            for region in all_regions:
                if region in all_metrics[species]:
                    bar_data.append(
                        {'Species': species, 'Region': region, 'Value': all_metrics[species][region][metric]})
        if bar_data:
            df_bar = pd.DataFrame(bar_data)
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Region', y='Value', hue='Species', data=df_bar)
            plt.title(f'{metric.capitalize()} Comparison by Region and Species')
            plt.xlabel('Region')
            plt.ylabel(metric.capitalize())
            for container in ax.containers:
                ax.bar_label(container, fmt='%.4f')
            plt.legend(title='Species')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'barplot_{metric}_comparison.png'))
            plt.close()
    # 4. LBP histogram distribution plots per region
    for region in all_regions:
        species_with_data = [s for s in all_species if region in species_data[s][0] and species_data[s][0][region]]
        if not species_with_data:
            continue
        plt.figure(figsize=(12, 6))
        for species in species_with_data:
            histograms = species_data[species][0][region]
            if histograms:
                avg_hist = np.mean(histograms, axis=0)
                plt.plot(avg_hist, label=species, alpha=0.7)
        plt.title(f'LBP Histogram Distribution for {region}')
        plt.xlabel('LBP Pattern')
        plt.ylabel('Normalized Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'lbp_distribution_{region}.png'))
        plt.close()
    # 5. 3D scatter plot of selected metrics
    selected_metrics = ['entropy', 'contrast', 'smoothness']
    scatter_data = []
    for species in all_species:
        for region in all_regions:
            if region in all_metrics[species] and all(m in all_metrics[species][region] for m in selected_metrics):
                scatter_data.append({'Species': species,
                                     'Region': region,
                                     'entropy': all_metrics[species][region]['entropy'],
                                     'contrast': all_metrics[species][region]['contrast'],
                                     'smoothness': all_metrics[species][region]['smoothness']})
    if scatter_data:
        df_scatter = pd.DataFrame(scatter_data)
        for region in all_regions:
            region_data = df_scatter[df_scatter['Region'] == region]
            if len(region_data) > 0:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                for i, species in enumerate(all_species):
                    species_data_ = region_data[region_data['Species'] == species]
                    if len(species_data_) > 0:
                        ax.scatter(species_data_['entropy'],
                                   species_data_['contrast'],
                                   species_data_['smoothness'],
                                   label=species, s=100, alpha=0.7)
                ax.set_xlabel('Entropy')
                ax.set_ylabel('Contrast')
                ax.set_zlabel('Smoothness')
                ax.set_title(f'3D Feature Space for {region}')
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'3d_features_{region}.png'))
                plt.close()
    print(f"All texture visualization outputs saved to {output_dir}")
    return all_metrics


###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    """Main function to run the bird species texture analysis pipeline."""
    print("Starting bird species texture analysis using LBP...")
    global RESULT_DIR, DEBUG_DIR, REGION_DIR, COMPARISON_DIR
    run_folder = f"R{RADIUS}_P{N_POINTS}_M{METHOD}"
    RESULT_DIR = os.path.join("Outputs", "LBP_Analysis", run_folder)
    os.makedirs(RESULT_DIR, exist_ok=True)
    DEBUG_DIR = os.path.join(RESULT_DIR, "debug_outputs")
    REGION_DIR = os.path.join(RESULT_DIR, "region_analysis")
    COMPARISON_DIR = os.path.join(RESULT_DIR, "species_comparison")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(REGION_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    # Get image paths for each species
    species_paths = {}
    for species_name in SPECIES:
        species_paths[species_name] = get_image_paths(species_name)
        print(f"Found {len(species_paths[species_name])} images for {species_name}")

    # Visualize basic LBP patterns
    print("\nCreating LBP visualization examples...")
    visualize_lbp_patterns(species_paths)

    # Create region-specific LBP visualizations
    print("\nCreating region-specific LBP visualizations...")
    # This will save detailed images (and display one of them) for debugging purposes.
    for species_name, paths in species_paths.items():
        if not paths:
            continue
        img_path, seg_path = paths[0]
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is None or seg is None:
            continue
        region_masks, region_stats, debug_img = get_region_masks(seg, debug=True)
        species_debug_dir = os.path.join(DEBUG_DIR, species_name)
        os.makedirs(species_debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(species_debug_dir, "segmented_regions.png"), debug_img)
        for region_name in REGION_COLORS:
            if region_name in region_masks and region_name in region_stats:
                visualize_region_lbp_details(img_path, seg_path, region_name, species_debug_dir)

    # Analyze textures for each species
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

    # Analyze detailed texture properties (including correlation)
    print("\nAnalyzing texture properties for individual regions...")
    texture_properties = analyze_texture_properties(species_data)

    # Visualize texture differences using heatmaps, radar charts, etc.
    print("\nCreating comprehensive texture difference visualizations...")
    all_metrics = visualize_texture_differences(species_data, output_dir=RESULT_DIR)

    # Build dataset and evaluate classification performance
    print("\nBuilding dataset for classification...")
    X, y, region_info = build_feature_dataset(species_data)
    if len(np.unique(y)) >= 2:
        print("\nEvaluating classification performance...")
        clf_results = evaluate_classification(X, y, region_info)
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
            for region, metrics in clf_results.items():
                f.write(f"  {region}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value:.4f}\n")
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
        print("\nClassification and analysis summary saved to:", RESULT_DIR)
    else:
        print("Insufficient data for classification evaluation")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
