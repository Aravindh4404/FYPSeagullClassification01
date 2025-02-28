from Features_Analysis.config import *

import os
import numpy as np
import cv2
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from matplotlib.patches import Rectangle
import random
from sklearn.decomposition import PCA
from scipy.stats import levene
import matplotlib.gridspec as gridspec

# Create output directory for saving results
OUTPUT_DIR = "Outputs/Results_Gabor_Enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_gabor_filters():
    """
    Create a bank of Gabor filters with different orientations and frequencies.
    """
    filters = []
    num_theta = 4  # Orientations
    num_freq = 2  # Frequencies
    ksize = 31  # Kernel size
    sigma = 4.0  # Gaussian standard deviation

    # Visualize each filter
    plt.figure(figsize=(12, 6))
    for theta in range(num_theta):
        theta_val = theta * np.pi / num_theta
        for freq in range(num_freq):
            frequency = 0.1 + freq * 0.1
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta_val,
                frequency,
                0.5,
                0,
                ktype=cv2.CV_32F
            )
            # Normalize kernel
            kernel /= kernel.sum()
            filter_name = f"gabor_{theta}_{freq}"
            filters.append((filter_name, kernel))

            # Visualize filter
            idx = theta * num_freq + freq + 1
            plt.subplot(2, 4, idx)
            plt.imshow(kernel, cmap='gray')
            angle_degrees = int(theta * 180 / num_theta)
            plt.title(f"θ={angle_degrees}°, f={frequency:.1f}")
            plt.axis('off')

    plt.suptitle("Gabor Filter Bank (Visualization of Filters Used)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gabor_filters.png"))
    plt.close()

    return filters


def extract_gabor_features(image, mask, filters):
    """
    Extract Gabor filter–based features for a masked region in the image.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # If mask is empty, return None
    if np.sum(mask) == 0:
        return None

    features = {}
    filter_responses = {}

    # Apply each Gabor filter and collect stats
    for filter_name, kernel in filters:
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        # Only consider pixels within the mask
        masked_response = filtered[mask > 0]

        features[f"{filter_name}_mean"] = np.mean(masked_response)
        features[f"{filter_name}_std"] = np.std(masked_response)
        features[f"{filter_name}_energy"] = np.sum(masked_response ** 2)

        # Store response for visualization
        filter_responses[filter_name] = filtered

    return features, filter_responses


def plot_gabor_responses(image, mask, filters, filter_responses, title, save_path=None):
    """
    Enhanced visualization showing the image, mask, and most significant filter responses.
    """
    if np.sum(mask) == 0:
        return

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        colored_img = image.copy()
    else:
        gray = image.copy()
        colored_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Set up the figure with GridSpec for better layout control
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig)

    # 1. Original image with mask overlay
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB))

    # Create a red mask overlay
    mask_overlay = np.zeros_like(colored_img)
    mask_overlay[:, :, 2] = mask  # Red channel
    ax1.imshow(mask_overlay, alpha=0.3)

    ax1.set_title('Original with Region Highlighted')
    ax1.axis('off')

    # 2. Zoomed masked region
    ax2 = fig.add_subplot(gs[0, 2:])

    # Find bounding box of mask for zooming
    y, x = np.where(mask > 0)
    if len(y) > 0 and len(x) > 0:
        # Add padding
        padding = 20
        y_min, y_max = max(0, np.min(y) - padding), min(image.shape[0], np.max(y) + padding)
        x_min, x_max = max(0, np.min(x) - padding), min(image.shape[1], np.max(x) + padding)

        # Crop and display zoomed region
        zoomed = colored_img[y_min:y_max, x_min:x_max].copy()
        zoomed_mask = mask[y_min:y_max, x_min:x_max].copy()

        # Create binary mask for clear visualization
        for c in range(3):
            channel = zoomed[:, :, c]
            channel[zoomed_mask == 0] = channel[zoomed_mask == 0] // 2  # Darken non-mask regions
            zoomed[:, :, c] = channel

        ax2.imshow(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB))
        ax2.set_title('Zoomed Region (Analysis Focus)')
        ax2.axis('off')

    # 3. Filtered responses (for each filter)
    for idx, (filter_name, kernel) in enumerate(filters):
        if idx >= 8:  # Only show first 8 filters
            break

        row = (idx // 4) + 1
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])

        # Get filter response
        filtered = filter_responses[filter_name]

        # Apply normalization for better visualization
        response_vis = filtered.copy()
        response_vis = cv2.normalize(response_vis, None, 0, 255, cv2.NORM_MINMAX)

        # Create zoomed version if possible
        if len(y) > 0 and len(x) > 0:
            response_vis = response_vis[y_min:y_max, x_min:x_max]

        ax.imshow(response_vis, cmap='jet')

        # Parse orientation and frequency from filter name
        parts = filter_name.split('_')
        orientation = int(parts[1]) * 180 // 4  # Convert to degrees
        freq = float(parts[2]) * 0.1 + 0.1

        ax.set_title(f"θ={orientation}°, f={freq:.1f}")
        ax.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_features_summary(sb_features_all, gw_features_all):
    """
    Create a summary dataframe of all features for both species.
    Includes effect size and tests for equal variance.
    """
    # Initialize dictionaries to store results
    summary = {}
    p_values = {}
    significant_features = []

    # Get all feature keys
    all_keys = set()
    for features_dict, _ in sb_features_all + gw_features_all:
        if features_dict is not None:
            all_keys.update(features_dict.keys())

    # Calculate statistics for each feature
    for key in all_keys:
        sb_values = [f[key] for f, _ in sb_features_all if f is not None and key in f]
        gw_values = [f[key] for f, _ in gw_features_all if f is not None and key in f]

        if len(sb_values) > 0 and len(gw_values) > 0:
            # Calculate means and std deviations
            sb_mean = np.mean(sb_values)
            gw_mean = np.mean(gw_values)
            sb_std = np.std(sb_values)
            gw_std = np.std(gw_values)

            # Test for equal variances
            levene_stat, levene_p = levene(sb_values, gw_values)
            equal_var = levene_p > 0.05

            # Perform t-test with appropriate equal_var parameter
            t_stat, p_value = ttest_ind(sb_values, gw_values, equal_var=equal_var)

            # Calculate Cohen's d effect size
            pooled_std = np.sqrt(((len(sb_values) - 1) * sb_std ** 2 +
                                  (len(gw_values) - 1) * gw_std ** 2) /
                                 (len(sb_values) + len(gw_values) - 2))

            if pooled_std == 0:
                cohen_d = 0
            else:
                cohen_d = (sb_mean - gw_mean) / pooled_std

            # Store results
            summary[key] = {
                'Slaty-backed Mean': sb_mean,
                'Glaucous-winged Mean': gw_mean,
                'Slaty-backed Std': sb_std,
                'Glaucous-winged Std': gw_std,
                'Difference (%)': ((sb_mean - gw_mean) / gw_mean * 100) if gw_mean != 0 else float('inf'),
                'Equal Variance': equal_var,
                'Levene p-value': levene_p,
                'T-statistic': t_stat,
                'P-value': p_value,
                'Effect Size (Cohen\'s d)': cohen_d,
                'Effect Magnitude': get_effect_magnitude(cohen_d),
                'Significant': p_value < 0.05
            }

            p_values[key] = p_value

            if p_value < 0.05:
                significant_features.append(key)

    return summary, significant_features


def get_effect_magnitude(cohen_d):
    """Helper function to categorize effect size"""
    d = abs(cohen_d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def plot_feature_comparison(sb_features, gw_features, feature_name, region_name, save_path=None):
    """
    Create a comparative box plot with individual data points and effect size for a specific Gabor feature.
    """
    sb_values = [f[feature_name] for f, _ in sb_features if f is not None and feature_name in f]
    gw_values = [f[feature_name] for f, _ in gw_features if f is not None and feature_name in f]

    if not sb_values or not gw_values:
        return None, None  # No data to plot

    # Test for equal variances
    levene_stat, levene_p = levene(sb_values, gw_values)
    equal_var = levene_p > 0.05

    # Calculate statistics
    t_stat, p_value = ttest_ind(sb_values, gw_values, equal_var=equal_var)

    # Calculate effect size (Cohen's d)
    mean1, mean2 = np.mean(sb_values), np.mean(gw_values)
    std1, std2 = np.std(sb_values), np.std(gw_values)
    n1, n2 = len(sb_values), len(gw_values)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        cohen_d = 0
    else:
        cohen_d = (mean1 - mean2) / pooled_std

    effect_magnitude = get_effect_magnitude(cohen_d)

    # Parse important information from feature name
    parts = feature_name.split('_')
    if len(parts) >= 4:
        orientation = int(parts[1]) * 180 // 4  # Convert to degrees
        freq_idx = int(parts[2])
        freq = 0.1 + freq_idx * 0.1
        stat_type = parts[3]  # mean, std, or energy
    else:
        orientation = "?"
        freq = "?"
        stat_type = "?"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create violin plot with individual data points
    positions = [1, 2]
    violins = ax.violinplot([sb_values, gw_values], positions, showmeans=True,
                            showextrema=True, widths=0.7)

    # Add labels and grid
    ax.set_xticks(positions)
    ax.set_xticklabels(['Slaty-backed', 'Glaucous-winged'])
    ax.set_ylabel(f'Feature Value ({stat_type})')
    ax.grid(True, alpha=0.3)

    # Customize title with feature details
    title = f"{region_name} - Feature: {feature_name}\n"
    title += f"Orientation: {orientation}°, Frequency: {freq:.1f}, Measure: {stat_type}\n"
    title += f"T-stat={t_stat:.3f}, p={p_value:.4f}"
    if p_value < 0.05:
        title += f" (significant), Effect Size: {cohen_d:.2f} ({effect_magnitude})"
    else:
        title += f" (not significant), Effect Size: {cohen_d:.2f} ({effect_magnitude})"

    # Add warning if equal variance assumption is violated
    if not equal_var:
        title += "\nNote: Equal variance assumption violated (Welch's t-test used)"

    ax.set_title(title)

    # Add scatter points for individual data values
    x1 = np.random.normal(1, 0.08, size=len(sb_values))
    x2 = np.random.normal(2, 0.08, size=len(gw_values))
    ax.scatter(x1, sb_values, alpha=0.7, s=40, c='blue', edgecolors='black', zorder=3)
    ax.scatter(x2, gw_values, alpha=0.7, s=40, c='orange', edgecolors='black', zorder=3)

    # Add sample sizes
    ax.text(1, min(sb_values), f'n={len(sb_values)}', ha='center', va='bottom')
    ax.text(2, min(gw_values), f'n={len(gw_values)}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return t_stat, p_value


def visualize_discriminative_features(sb_features, gw_features, significant_features, region_name, save_path=None):
    """
    Create a visualization showing the most discriminative features and their patterns.
    """
    if not significant_features:
        return

    # Get the top 3 most significant features based on p-value
    feature_pvalues = []
    for feature in significant_features:
        sb_values = [f[feature] for f, _ in sb_features if f is not None and feature in f]
        gw_values = [f[feature] for f, _ in gw_features if f is not None and feature in f]

        if sb_values and gw_values:
            _, p_value = ttest_ind(sb_values, gw_values)
            feature_pvalues.append((feature, p_value))

    # Sort by p-value (most significant first)
    feature_pvalues.sort(key=lambda x: x[1])
    top_features = [f[0] for f in feature_pvalues[:3]]

    if not top_features:
        return

    # Select one random example from each species
    sb_idx = random.randint(0, len(sb_features) - 1)
    gw_idx = random.randint(0, len(gw_features) - 1)

    # Get the filter responses for these examples
    sb_feature, sb_responses = sb_features[sb_idx]
    gw_feature, gw_responses = gw_features[gw_idx]

    # Set up the visualization
    fig, axes = plt.subplots(len(top_features), 2, figsize=(12, 4 * len(top_features)))

    if len(top_features) == 1:
        axes = np.array([axes])

    # For each top feature, visualize the difference
    for i, feature in enumerate(top_features):
        parts = feature.split('_')
        if len(parts) >= 4:
            filter_name = f"gabor_{parts[1]}_{parts[2]}"
            measure_type = parts[3]

            # Parse important details for title
            orientation = int(parts[1]) * 180 // 4  # Convert to degrees
            freq_idx = int(parts[2])
            freq = 0.1 + freq_idx * 0.1

            # Get mean values for both species
            sb_vals = [f[feature] for f, _ in sb_features if f is not None and feature in f]
            gw_vals = [f[feature] for f, _ in gw_features if f is not None and feature in f]
            sb_mean = np.mean(sb_vals) if sb_vals else 0
            gw_mean = np.mean(gw_vals) if gw_vals else 0

            # Calculate effect size
            if sb_vals and gw_vals:
                std1, std2 = np.std(sb_vals), np.std(gw_vals)
                n1, n2 = len(sb_vals), len(gw_vals)
                pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
                cohen_d = (sb_mean - gw_mean) / pooled_std if pooled_std > 0 else 0
                effect_magnitude = get_effect_magnitude(cohen_d)
            else:
                cohen_d = 0
                effect_magnitude = "Unknown"

            # Get the filter responses
            if filter_name in sb_responses:
                sb_response = sb_responses[filter_name]
                axes[i, 0].imshow(sb_response, cmap='jet')
                axes[i, 0].set_title(f"Slaty-backed: {measure_type}={sb_mean:.2f}")
                axes[i, 0].axis('off')

            if filter_name in gw_responses:
                gw_response = gw_responses[filter_name]
                axes[i, 1].imshow(gw_response, cmap='jet')
                axes[i, 1].set_title(f"Glaucous-winged: {measure_type}={gw_mean:.2f}")
                axes[i, 1].axis('off')

            # Add row title on the left
            fig.text(0.01, 0.5 + (len(top_features) - 1 - i) * (0.9 / len(top_features)),
                     f"Feature {i + 1}: Orientation={orientation}°, Freq={freq:.1f}, Type={measure_type}\n" +
                     f"Effect size: {cohen_d:.2f} ({effect_magnitude})",
                     va='center', ha='left', fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.suptitle(f"Top Discriminative Features for {region_name}\n" +
                 f"(Showing filter responses with colormap: blue=low, red=high)", fontsize=16)
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_feature_pca(sb_features, gw_features, region_name, save_path=None):
    """
    Create a PCA plot showing how the two species separate in feature space
    """
    # Extract feature vectors for both species
    feature_vectors = []
    labels = []

    # Get list of common features across all samples
    all_features = set()
    for features, _ in sb_features + gw_features:
        if features is not None:
            all_features.update(features.keys())

    # Convert to lists for both species
    sb_vectors = []
    for features, _ in sb_features:
        if features is not None:
            vector = [features.get(f, 0) for f in sorted(all_features)]
            sb_vectors.append(vector)
            feature_vectors.append(vector)
            labels.append("Slaty-backed")

    gw_vectors = []
    for features, _ in gw_features:
        if features is not None:
            vector = [features.get(f, 0) for f in sorted(all_features)]
            gw_vectors.append(vector)
            feature_vectors.append(vector)
            labels.append("Glaucous-winged")

    if not feature_vectors:
        return

    # Convert to numpy array
    X = np.array(feature_vectors)

    # Normalize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Avoid division by zero
    X_norm = (X - X_mean) / X_std

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)

    # Plot results
    plt.figure(figsize=(10, 8))

    # Plot each species with different color
    sb_mask = np.array(labels) == "Slaty-backed"
    gw_mask = np.array(labels) == "Glaucous-winged"

    plt.scatter(X_pca[sb_mask, 0], X_pca[sb_mask, 1], c='blue', marker='o', s=80, label='Slaty-backed', alpha=0.7)
    plt.scatter(X_pca[gw_mask, 0], X_pca[gw_mask, 1], c='orange', marker='^', s=80, label='Glaucous-winged', alpha=0.7)

    # Add confidence ellipses
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*
        """
        if x.size != y.size:
            raise ValueError("x and y must have the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                          facecolor=facecolor, **kwargs)

        # Calculating the standard deviation of x from the square root of the variance
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    # Add confidence ellipses if we have enough data points
    ax = plt.gca()
    if np.sum(sb_mask) > 2:
        confidence_ellipse(X_pca[sb_mask, 0], X_pca[sb_mask, 1], ax, n_std=2.0,
                           edgecolor='blue', linestyle='--', linewidth=1, alpha=0.5,
                           label='Slaty-backed 95% Confidence')

    if np.sum(gw_mask) > 2:
        confidence_ellipse(X_pca[gw_mask, 0], X_pca[gw_mask, 1], ax, n_std=2.0,
                           edgecolor='orange', linestyle='--', linewidth=1, alpha=0.5,
                           label='Glaucous-winged 95% Confidence')

    # Calculate overlap percentage (approximate using distance between means)
    if np.sum(sb_mask) > 0 and np.sum(gw_mask) > 0:
        sb_centroid = np.mean(X_pca[sb_mask], axis=0)
        gw_centroid = np.mean(X_pca[gw_mask], axis=0)

        distance = np.linalg.norm(sb_centroid - gw_centroid)

        # Average spread (radius of gyration)
        sb_spread = np.mean(np.linalg.norm(X_pca[sb_mask] - sb_centroid, axis=1))
        gw_spread = np.mean(np.linalg.norm(X_pca[gw_mask] - gw_centroid, axis=1))

        # Approximate overlap as inversely proportional to distance between means
        # relative to the average spread
        avg_spread = (sb_spread + gw_spread) / 2
        if avg_spread > 0:
            overlap_metric = np.exp(-distance / avg_spread)  # 0 = no overlap, 1 = complete overlap
            plt.title(f"{region_name} - PCA of Gabor Features\n" +
                      f"Variance explained: {pca.explained_variance_ratio_[0] * 100:.1f}% (PC1), " +
                      f"{pca.explained_variance_ratio_[1] * 100:.1f}% (PC2)\n" +
                      f"Separation Index: {1 - overlap_metric:.2f}")
        else:
            plt.title(f"{region_name} - PCA of Gabor Features\n" +
                      f"Variance explained: {pca.explained_variance_ratio_[0] * 100:.1f}% (PC1), " +
                      f"{pca.explained_variance_ratio_[1] * 100:.1f}% (PC2)")
    else:
        plt.title(f"{region_name} - PCA of Gabor Features\n" +
                  f"Variance explained: {pca.explained_variance_ratio_[0] * 100:.1f}% (PC1), " +
                  f"{pca.explained_variance_ratio_[1] * 100:.1f}% (PC2)")

    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.grid(alpha=0.3)
    plt.legend()

    # Add sample counts
    plt.text(0.02, 0.98, f"Slaty-backed samples: {np.sum(sb_mask)}\nGlaucous-winged samples: {np.sum(gw_mask)}",
             transform=plt.gca().transAxes, va='top', ha='left',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # If we have PC loadings, annotate top 5 features
    feature_names = sorted(all_features)
    if feature_names and pca.components_.shape[1] == len(feature_names):
        # Get indices of top features (absolute value of loading)
        top_features_pc1 = np.argsort(np.abs(pca.components_[0]))[-5:]

        # Create a text description of top features
        pc1_features = []
        for idx in top_features_pc1:
            feature = feature_names[idx]
            loading = pca.components_[0][idx]
            # Parse the feature name to get more interpretable description
            parts = feature.split('_')
            if len(parts) >= 4:
                orientation = int(parts[1]) * 180 // 4  # Convert to degrees
                freq_idx = int(parts[2])
                freq = 0.1 + freq_idx * 0.1
                measure = parts[3]
                pc1_features.append(f"{orientation}° freq={freq:.1f} {measure}: {loading:.3f}")

        plt.figtext(0.5, 0.01,
                    "Top 5 features for PC1 (with loadings):\n" + ", ".join(pc1_features),
                    ha='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def generate_insight_report(summary, significant_features, region_name):
    """
    Generate a text report of insights from the analysis with added information about
    potential artifacts and statistical considerations.
    """
    lines = [
        f"====== ANALYSIS REPORT: {region_name} ======",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nNumber of features analyzed: {len(summary)}",
        f"Number of statistically significant features (p < 0.05): {len(significant_features)}",
        "\n----- SIGNIFICANT DIFFERENCES BETWEEN SPECIES -----"
    ]

    if significant_features:
        # Sort by p-value first (most significant), or you could sort by absolute percent difference.
        # Here, we show an example sorting by p-value:
        sorted_features = sorted(
            [(k, v) for k, v in summary.items() if k in significant_features],
            key=lambda x: x[1]['P-value']
        )

        for feature, stats in sorted_features:
            diff = stats['Difference (%)']
            direction = "higher" if diff > 0 else "lower"
            p_val = stats['P-value']
            effect_size = stats["Effect Size (Cohen's d)"]
            effect_magnitude = stats['Effect Magnitude']

            lines.append(
                f"• {feature}: "
                f"Slaty-backed gulls show {abs(diff):.1f}% {direction} values than Glaucous-winged "
                f"(p={p_val:.4f}, d={effect_size:.2f} [{effect_magnitude} effect])"
            )

        # ----- INTERPRETATION SECTION -----
        lines.append("\n----- INTERPRETATION -----")

        # Group features by type (e.g., _mean, _std, _energy)
        texture_features = [f for f in significant_features if "std" in f or "energy" in f or "mean" in f]
        orientation_map = {}

        for f in significant_features:
            parts = f.split('_')
            if len(parts) >= 4:
                # parts[1] = orientation index, parts[2] = frequency index
                # Convert orientation index to degrees (theta * 180 / num_theta)
                orientation = int(parts[1]) * (180 // 4)  # since num_theta=4 in create_gabor_filters()
                if orientation not in orientation_map:
                    orientation_map[orientation] = []
                orientation_map[orientation].append(f)

        if texture_features:
            lines.append("• Texture/Intensity differences detected:")
            if any("std" in feat for feat in texture_features):
                lines.append("  - Standard Deviation: Indicates different texture variation or homogeneity.")
            if any("energy" in feat for feat in texture_features):
                lines.append("  - Energy: Reflects overall strength/intensity of texture patterns.")
            if any("mean" in feat for feat in texture_features):
                lines.append("  - Mean: Highlights brightness/intensity differences in the masked region.")

        if orientation_map:
            lines.append("• Orientation Analysis (Gabor filters):")
            for angle, feats in orientation_map.items():
                lines.append(f"  - {angle}° orientation: {len(feats)} significant features.")
            # Find which orientation has the most significant features
            max_orientation = max(orientation_map.items(), key=lambda x: len(x[1]))
            lines.append(
                f"  - The most discriminative orientation is {max_orientation[0]}°, "
                "suggesting notable feather/wingtip pattern differences at this angle."
            )

    else:
        lines.append("No statistically significant differences were found between species.")

    # ----- ADDITIONAL CONSIDERATIONS -----
    lines.append("\n----- ADDITIONAL CONSIDERATIONS -----")
    lines.append(
        "• Check for potential artifacts such as differences in wing angle, lighting, or image quality.\n"
        "• Verify that the sample sizes are adequate and representative of the overall population.\n"
        "• Consider rotation-invariant or scale-invariant texture features (e.g., LBP or SIFT-like methods)\n"
        "  if wing orientation or size variation is large."
    )

    # ----- CONCLUSION -----
    lines.append("\n----- CONCLUSION -----")
    if significant_features:
        lines.append(
            "The analysis demonstrates notable texture/intensity differences in the selected region between "
            "Slaty-backed and Glaucous-winged gulls. These findings could be leveraged for automated species "
            "classification, but be mindful of potential pose or scale artifacts."
        )
    else:
        lines.append(
            "No significant differences were found with the current Gabor filter parameters. "
            "Consider adjusting parameters (e.g., more orientations/frequencies) or exploring "
            "other feature extraction methods."
        )

    return "\n".join(lines)

def plot_heatmap(feature_summary, region_name, save_path=None):
    """
    Create a heatmap of p-values to visualize significant features.
    """
    # Extract p-values for all features
    data = {}
    for key, values in feature_summary.items():
        parts = key.split('_')
        if len(parts) >= 4:  # Ensure it's a gabor feature
            filter_type = f"gabor_{parts[1]}_{parts[2]}"
            stat_type = parts[3]

            if filter_type not in data:
                data[filter_type] = {}

            data[filter_type][stat_type] = values['P-value']

    # Convert to dataframe for heatmap
    if not data:
        return

    df = pd.DataFrame(data).T

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='coolwarm_r', vmin=0, vmax=0.10,
                linewidths=.5, cbar_kws={'label': 'p-value'})
    plt.title(f'P-values for {region_name} Features\n(p < 0.05 indicates significant difference)')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    print("Starting Enhanced Gabor-based Texture Analysis...")

    # Number of images to process (set to None to process all images)
    S = None  # Process all available images

    # Create Gabor filters and visualize them
    filters = create_gabor_filters()

    # Dictionary to store all features by region and species
    all_features = {}

    # --- Load Slaty-backed gull images ---
    sb_images = []
    sb_segs = []
    sb_filenames = sorted(os.listdir(SLATY_BACKED_IMG_DIR))[:S]
    print(f"Found {len(sb_filenames)} Slaty-backed images")

    for img_name in sb_filenames:
        img_path = os.path.join(SLATY_BACKED_IMG_DIR, img_name)
        seg_path = os.path.join(SLATY_BACKED_SEG_DIR, img_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is not None and seg is not None:
            sb_images.append(img)
            sb_segs.append(seg)
        else:
            print(f"Could not load {img_name} or its segmentation.")

    # --- Load Glaucous-winged gull images ---
    gw_images = []
    gw_segs = []
    gw_filenames = sorted(os.listdir(GLAUCOUS_WINGED_IMG_DIR))[:S]
    print(f"Found {len(gw_filenames)} Glaucous-winged images")

    for img_name in gw_filenames:
        img_path = os.path.join(GLAUCOUS_WINGED_IMG_DIR, img_name)
        seg_path = os.path.join(GLAUCOUS_WINGED_SEG_DIR, img_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is not None and seg is not None:
            gw_images.append(img)
            gw_segs.append(seg)
        else:
            print(f"Could not load {img_name} or its segmentation.")

    # --- Process each defined region ---
    for region_name, color in REGION_COLORS.items():
        print(f"\nAnalyzing region: {region_name}")
        region_dir = os.path.join(OUTPUT_DIR, region_name)
        os.makedirs(region_dir, exist_ok=True)

        sb_region_features = []  # List of tuples: (features, filter_responses)
        gw_region_features = []  # List of tuples: (features, filter_responses)

        # --- Process Slaty-backed images ---
        for idx, (img, seg) in enumerate(zip(sb_images, sb_segs)):
            tolerance = 10
            lower = np.array([max(c - tolerance, 0) for c in color])
            upper = np.array([min(c + tolerance, 255) for c in color])
            mask = cv2.inRange(seg, lower, upper)

            result = extract_gabor_features(img, mask, filters)
            if result:
                features, filter_responses = result
                sb_region_features.append((features, filter_responses))

                # Save enhanced Gabor response visualization for first 3 images
                if idx < 3:
                    save_path = os.path.join(region_dir, f"sb_enhanced_gabor_response_{idx+1}.png")
                    plot_gabor_responses(
                        img, mask, filters, filter_responses,
                        f"Slaty-backed Gull - {region_name} Enhanced Gabor Responses - Image {idx+1}",
                        save_path
                    )

        # --- Process Glaucous-winged images ---
        for idx, (img, seg) in enumerate(zip(gw_images, gw_segs)):
            tolerance = 10
            lower = np.array([max(c - tolerance, 0) for c in color])
            upper = np.array([min(c + tolerance, 255) for c in color])
            mask = cv2.inRange(seg, lower, upper)

            result = extract_gabor_features(img, mask, filters)
            if result:
                features, filter_responses = result
                gw_region_features.append((features, filter_responses))

                # Save enhanced Gabor response visualization for first 3 images
                if idx < 3:
                    save_path = os.path.join(region_dir, f"gw_enhanced_gabor_response_{idx+1}.png")
                    plot_gabor_responses(
                        img, mask, filters, filter_responses,
                        f"Glaucous-winged Gull - {region_name} Enhanced Gabor Responses - Image {idx+1}",
                        save_path
                    )

        all_features[region_name] = {
            'sb': sb_region_features,
            'gw': gw_region_features
        }

        # --- Generate comprehensive analysis for this region ---
        if sb_region_features and gw_region_features:
            print(f"\nGenerating statistical summary for region: {region_name}")

            # Create feature summary (with enhanced statistics)
            feature_summary, significant_features = create_features_summary(sb_region_features, gw_region_features)

            # Save summary to CSV
            summary_df = pd.DataFrame.from_dict(feature_summary, orient='index')
            summary_csv_path = os.path.join(region_dir, f"{region_name}_enhanced_statistics.csv")
            summary_df.to_csv(summary_csv_path)

            # Plot heatmap of p-values
            heatmap_path = os.path.join(region_dir, f"{region_name}_enhanced_pvalue_heatmap.png")
            plot_heatmap(feature_summary, region_name, heatmap_path)

            # Plot box plots for each significant feature
            for feature in significant_features:
                box_plot_path = os.path.join(region_dir, f"{region_name}_{feature}_enhanced_boxplot.png")
                plot_feature_comparison(sb_region_features, gw_region_features, feature, region_name, box_plot_path)

            # Visualize the most discriminative features
            discriminative_plot_path = os.path.join(region_dir, f"{region_name}_discriminative_features.png")
            visualize_discriminative_features(sb_region_features, gw_region_features, significant_features, region_name, discriminative_plot_path)

            # Plot PCA of the features
            pca_plot_path = os.path.join(region_dir, f"{region_name}_enhanced_pca.png")
            plot_feature_pca(sb_region_features, gw_region_features, region_name, pca_plot_path)

            # Generate insight report and save to a text file
            report = generate_insight_report(feature_summary, significant_features, region_name)
            report_path = os.path.join(region_dir, f"{region_name}_enhanced_analysis_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)

            print("\n" + "=" * 80)
            print(report)
            print("=" * 80)
        else:
            print(f"Insufficient feature data for region: {region_name}")

    print(f"\nEnhanced analysis complete! Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
