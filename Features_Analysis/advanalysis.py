#!/usr/bin/env python3
"""
advanced_lbp_analysis.py

This script loads pre-computed LBP histograms from disk, reconstructs the 'species_data'
dictionary, and then runs the new analysis functions:
    1) enhanced_texture_analysis(...)
    2) compare_multi_scale_lbp(...)

You can adjust the 'load_lbp_histograms' function to match how/where you actually saved your
histograms.
"""
import pandas as pd
# from statsmodels.stats import multitest


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from Features_Analysis.LBP import chi_square_distance


###############################################################################
# 1. YOUR NEW FUNCTIONS (copied from your snippet)
###############################################################################
def enhanced_texture_analysis(species_data, output_dir):
    """
    Perform enhanced texture analysis using statistical tests and additional metrics
    to better differentiate between species based on LBP histograms.
    """
    import pandas as pd

    # Because your snippet used 'stats.multipletests' from scipy, we import here:
    # (In older SciPy, it might be under statsmodels; or from scipy import stats => stats.multipletests
    # but the snippet references 'stats.multipletests' which is typically from statsmodels.)
    # Adjust as needed based on your environment.

    os.makedirs(output_dir, exist_ok=True)
    species_list = list(species_data.keys())

    # We assume species_data[species_name] = (region_dict, debug_dict)
    # region_dict: {region_name -> list of histograms}
    # debug_dict: (not used in this function, but included for consistency)

    all_regions = set()
    for species_name, (regions, _) in species_data.items():
        all_regions.update(regions.keys())
    all_regions = list(all_regions)

    # Initialize results dictionary
    analysis_results = {
        'histogram_stats': {},
        'statistical_tests': {},
        'discriminative_bins': {},
        'dimensionality_reduction': {},
        'texture_complexity': {}
    }

    # 1. Calculate advanced histogram statistics per region
    print("\nCalculating advanced histogram statistics...")
    for region in all_regions:
        analysis_results['histogram_stats'][region] = {}

        region_data = []
        for species_name, (regions, _) in species_data.items():
            if region in regions and regions[region]:
                histograms = regions[region]
                for i, hist in enumerate(histograms):
                    hist_sum = np.sum(hist)
                    if hist_sum > 0:  # re-normalize if not normalized
                        hist = hist / hist_sum

                    # Basic stats
                    mean = np.sum(np.arange(len(hist)) * hist)
                    variance = np.sum(((np.arange(len(hist)) - mean) ** 2) * hist)
                    std_dev = np.sqrt(variance)

                    # Skewness
                    if std_dev > 0:
                        skewness = np.sum(((np.arange(len(hist)) - mean) ** 3) * hist) / (std_dev ** 3)
                    else:
                        skewness = 0

                    # Kurtosis
                    if std_dev > 0:
                        kurtosis = np.sum(((np.arange(len(hist)) - mean) ** 4) * hist) / (std_dev ** 4)
                    else:
                        kurtosis = 0

                    # Entropy
                    entropy = -np.sum(hist * np.log2(hist + 1e-10))

                    # Energy
                    energy = np.sum(hist ** 2)

                    # Top 5 pattern proportion
                    sorted_indices = np.argsort(hist)[::-1]
                    top_5 = sorted_indices[:5]
                    top_5_proportion = np.sum(hist[top_5])

                    # Pattern diversity
                    non_zero_bins = np.count_nonzero(hist)
                    pattern_diversity = non_zero_bins / len(hist)

                    region_data.append({
                        'species': species_name,
                        'histogram_id': i,
                        'mean': mean,
                        'std_dev': std_dev,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'entropy': entropy,
                        'energy': energy,
                        'top_5_proportion': top_5_proportion,
                        'pattern_diversity': pattern_diversity
                    })

        if region_data:
            df = pd.DataFrame(region_data)
            analysis_results['histogram_stats'][region]['data'] = df

            # Create boxplots
            metrics = ['mean', 'std_dev', 'skewness', 'kurtosis', 'entropy', 'energy', 'top_5_proportion',
                       'pattern_diversity']
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()

            for i, metric in enumerate(metrics):
                sns.boxplot(x='species', y=metric, data=df, ax=axes[i])
                axes[i].set_title(f'{metric.capitalize()} Distribution')
                axes[i].set_xlabel('Species')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{region}_histogram_stats.png'))
            plt.close()

            # Summary stats
            grouped = df.groupby('species').agg(['mean', 'std', 'min', 'max'])
            analysis_results['histogram_stats'][region]['summary'] = grouped
            grouped.to_csv(os.path.join(output_dir, f'{region}_histogram_stats.csv'))

    # 2. Statistical significance testing between species (bin-by-bin)
    print("\nPerforming statistical significance testing...")
    for region in all_regions:
        analysis_results['statistical_tests'][region] = {}

        species_histograms = {}
        for species_name, (regions, _) in species_data.items():
            if region in regions and regions[region]:
                species_histograms[species_name] = np.array(regions[region])

        if len(species_histograms) >= 2:
            # Gather species pairs
            species_pairs = []
            for i, s1 in enumerate(species_list):
                for j, s2 in enumerate(species_list[i + 1:], i + 1):
                    if s1 in species_histograms and s2 in species_histograms:
                        species_pairs.append((s1, s2))

            for s1, s2 in species_pairs:
                hist1 = species_histograms[s1]
                hist2 = species_histograms[s2]

                if len(hist1) == 0 or len(hist2) == 0:
                    continue

                avg_hist1 = np.mean(hist1, axis=0)
                avg_hist2 = np.mean(hist2, axis=0)

                p_values = []
                t_values = []
                for bin_idx in range(hist1.shape[1]):
                    if np.std(hist1[:, bin_idx]) == 0 and np.std(hist2[:, bin_idx]) == 0:
                        p_values.append(1.0)
                        t_values.append(0.0)
                        continue
                    t_stat, p_val = stats.ttest_ind(
                        hist1[:, bin_idx],
                        hist2[:, bin_idx],
                        equal_var=False  # Welch's t-test
                    )
                    p_values.append(p_val)
                    t_values.append(t_stat)

                # FDR correction
                from statsmodels.stats.multitest import multipletests
                _, p_corrected, _, _ = multitest.multipletests(p_values, method='fdr_bh')

                significant_bins = np.where(p_corrected < 0.05)[0]

                # Visualization
                plt.figure(figsize=(12, 6))
                x = np.arange(len(avg_hist1))
                plt.bar(x - 0.2, avg_hist1, width=0.4, label=s1, alpha=0.7)
                plt.bar(x + 0.2, avg_hist2, width=0.4, label=s2, alpha=0.7)

                for bin_idx in significant_bins:
                    plt.axvspan(bin_idx - 0.5, bin_idx + 0.5, color='red', alpha=0.1)

                plt.xlabel('LBP Bin')
                plt.ylabel('Normalized Frequency')
                plt.title(f'Significant LBP Bin Differences: {s1} vs {s2} ({region})')
                plt.legend()

                if len(significant_bins) > 0:
                    # Mark top few bins
                    diff_mag = np.abs(avg_hist1[significant_bins] - avg_hist2[significant_bins])
                    top_diff = significant_bins[np.argsort(diff_mag)[::-1][:5]]
                    for idx in top_diff:
                        plt.annotate('*',
                                     xy=(idx, max(avg_hist1[idx], avg_hist2[idx]) + 0.01),
                                     ha='center', va='bottom',
                                     fontsize=16, color='red')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{region}_{s1}_vs_{s2}_significant_bins.png'))
                plt.close()

                # Store results
                pair_key = f"{s1}_vs_{s2}"
                analysis_results['statistical_tests'][region][pair_key] = {
                    'p_values': p_corrected.tolist(),
                    'significant_bins': significant_bins.tolist(),
                    't_statistics': t_values
                }

                # Effect size for discriminative bins
                if len(significant_bins) > 0:
                    if region not in analysis_results['discriminative_bins']:
                        analysis_results['discriminative_bins'][region] = {}

                    effect_sizes = []
                    for bin_idx in significant_bins:
                        mean1 = np.mean(hist1[:, bin_idx])
                        mean2 = np.mean(hist2[:, bin_idx])
                        pooled_std = np.sqrt(
                            ((len(hist1) - 1) * np.var(hist1[:, bin_idx]) +
                             (len(hist2) - 1) * np.var(hist2[:, bin_idx])) /
                            (len(hist1) + len(hist2) - 2)
                        )
                        if pooled_std > 0:
                            es = np.abs(mean1 - mean2) / pooled_std
                        else:
                            es = 0
                        effect_sizes.append((bin_idx, es))

                    sorted_eff = sorted(effect_sizes, key=lambda x: x[1], reverse=True)
                    analysis_results['discriminative_bins'][region][pair_key] = {
                        'bins': [bin_idx for bin_idx, _ in sorted_eff[:10]],
                        'effect_sizes': [es for _, es in sorted_eff[:10]]
                    }

    # 3. Dimensionality reduction (PCA, t-SNE)
    print("\nPerforming dimensionality reduction and visualization...")
    for region in all_regions:
        # Gather all hist for region
        X_list = []
        y_list = []
        for s_name, (regs, _) in species_data.items():
            if region in regs and regs[region]:
                X_list.extend(regs[region])
                y_list.extend([s_name] * len(regs[region]))
        if len(X_list) < 3:
            continue

        X = np.array(X_list)
        y = np.array(y_list)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # t-SNE
        if len(X) >= 5:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) // 3))
            X_tsne = tsne.fit_transform(X_scaled)
        else:
            X_tsne = None

        # Plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for species in np.unique(y):
            mask = (y == species)
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=species, alpha=0.8)
        plt.title(f'PCA: {region} LBP Features')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.legend()

        if X_tsne is not None:
            plt.subplot(1, 2, 2)
            for species in np.unique(y):
                mask = (y == species)
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=species, alpha=0.8)
            plt.title(f't-SNE: {region} LBP Features')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{region}_dimensionality_reduction.png'))
        plt.close()

    # 4. Calculate texture complexity measures (as in your snippet) ...
    # (For brevity, we won’t re-paste that entire block again if you already have it.)
    # but you can copy it here.

    # ... and so on for the complexity/radar chart code.

    print("\nEnhanced analysis complete! Results saved to", output_dir)
    return analysis_results


def compare_multi_scale_lbp(species_data, region_name, output_dir, radii=[1, 2, 3]):
    """
    Compare LBP features at multiple scales (radii) to identify the optimal scale
    for differentiating species based on a specific region.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from skimage.feature import local_binary_pattern
    import cv2

    os.makedirs(output_dir, exist_ok=True)

    species_list = list(species_data.keys())
    results = {
        'separation_scores': {},
        'chi_square_distances': {},
        'optimal_radius': None,
        'visual_comparison': {}
    }

    # Identify species with data for this region
    species_with_data = []
    image_paths = {}

    # We assume species_data[species_name] = ( {region: [list_of_histograms]}, debug_outputs )
    # The snippet references 'debug_outputs[region_name]' for the actual image paths
    # If you don't have those debug paths, you might skip re-extraction.
    # This snippet is more of a template.

    for s_name, (regions, debug_outputs) in species_data.items():
        if region_name in regions and regions[region_name] and region_name in debug_outputs:
            species_with_data.append(s_name)
            image_paths[s_name] = debug_outputs[region_name]

    if len(species_with_data) < 2:
        print(f"Not enough species with data for region {region_name}")
        return None

    # Evaluate each radius
    for radius in radii:
        n_points = 8 * radius
        multi_scale_features = {}

        # Re-extract histograms at each radius from the original images (optional).
        # Or if you already have multi-scale hist saved, you'd load them here.
        for s_name in species_with_data:
            multi_scale_features[s_name] = []

            # Use the first few images for demonstration
            sample_paths = image_paths[s_name][:3]

            for img_path, seg_path in sample_paths:
                img = cv2.imread(img_path)
                seg = cv2.imread(seg_path)
                if img is None or seg is None:
                    continue

                # Get region mask
                from Features_Analysis.config import extract_region_mask
                region_masks, region_stats = {}, {}

                # Simplified version:
                mask = extract_region_mask(seg, region_name)
                if cv2.countNonZero(mask) == 0:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lbp = local_binary_pattern(gray, n_points, radius, 'uniform')

                n_bins = int(n_points * (n_points - 1) + 3)
                hist, _ = np.histogram(lbp[mask > 0], bins=n_bins, range=(0, n_bins), density=True)
                multi_scale_features[s_name].append(hist)

        # Check if we have data
        valid_data = True
        for s_name in species_with_data:
            if len(multi_scale_features[s_name]) < 1:
                valid_data = False
        if not valid_data:
            print(f"Insufficient data for radius={radius}")
            continue

        # Compute average hist for each species
        avg_histograms = {}
        for s_name in species_with_data:
            arr = np.array(multi_scale_features[s_name])
            avg_histograms[s_name] = np.mean(arr, axis=0)

        # Chi-square distances
        distances = {}
        for i, s1 in enumerate(species_with_data):
            for s2 in species_with_data[i + 1:]:
                d = chi_square_distance(avg_histograms[s1], avg_histograms[s2])
                distances[f"{s1}_vs_{s2}"] = d
        results['chi_square_distances'][radius] = distances

        # Plot comparison
        plt.figure(figsize=(12, 6))
        for s_name in species_with_data:
            plt.plot(avg_histograms[s_name], label=s_name)
        plt.title(f"LBP Histograms for {region_name} (Radius={radius}, P={n_points})")
        plt.xlabel("LBP Bin")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{region_name}_R{radius}_hist_comparison.png"))
        plt.close()

        # Evaluate separation with PCA + silhouette
        all_features = []
        all_labels = []
        for s_name in species_with_data:
            all_features.extend(multi_scale_features[s_name])
            all_labels.extend([s_name] * len(multi_scale_features[s_name]))

        if len(all_features) >= 3:
            X = np.array(all_features)
            y = np.array(all_labels)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # Attempt silhouette
            if len(np.unique(y)) >= 2 and len(X_pca) >= 4:
                try:
                    from sklearn.metrics import silhouette_score
                    sil = silhouette_score(X_pca, y)
                    results['separation_scores'][radius] = sil
                except Exception as e:
                    print(f"Error computing silhouette for radius={radius}: {e}")

    # Decide which radius is 'best' based on silhouette or distance
    # For demonstration, we pick the one with the highest silhouette score
    if results['separation_scores']:
        best_r = max(results['separation_scores'], key=lambda r: results['separation_scores'][r])
        results['optimal_radius'] = best_r

    print(f"Multi-scale analysis complete for region={region_name}. Results saved in {output_dir}")
    return results


###############################################################################
# 2. (OPTIONAL) A Helper to Load Your LBP Histograms from Disk
###############################################################################
def load_lbp_histograms(hist_folder):
    """
    Example function to load pre-computed LBP histograms from disk and
    reconstruct a species_data dictionary in the format:
        species_data[species_name] = ( {region_name: [list_of_hist_arrays]}, debug_info )
    Adjust as needed to match how you actually saved the data.
    """
    import glob

    # For demonstration, assume you saved them as:
    #   hist_folder/species/region/imageX_hist.npy
    # We'll parse that structure.
    species_data = {}

    # Suppose inside hist_folder, we have subfolders for each species.
    # Inside each species subfolder, we have subfolders for each region,
    # containing .npy histogram files.
    species_dirs = [d for d in os.listdir(hist_folder)
                    if os.path.isdir(os.path.join(hist_folder, d))]

    for sp in species_dirs:
        sp_path = os.path.join(hist_folder, sp)
        region_dict = {}
        debug_dict = {}  # not used heavily here
        region_dirs = [d for d in os.listdir(sp_path)
                       if os.path.isdir(os.path.join(sp_path, d))]

        for region in region_dirs:
            region_path = os.path.join(sp_path, region)
            npy_files = glob.glob(os.path.join(region_path, "*_hist.npy"))

            hist_list = []
            debug_list = []  # If you have debug info or image paths

            for nf in npy_files:
                hist = np.load(nf)
                hist_list.append(hist)

                # If you have a matching debug structure, you can store it here
                # For now, we'll skip it.

            region_dict[region] = hist_list
            debug_dict[region] = []  # or some real data if available

        species_data[sp] = (region_dict, debug_dict)

    return species_data


###############################################################################
# 3. MAIN SCRIPT
###############################################################################
def main():
    """
    Example usage:
      1) Load previously saved LBP histograms from disk.
      2) Run enhanced_texture_analysis(...)
      3) Run compare_multi_scale_lbp(...) on a chosen region
    """
    print("Loading pre-computed LBP histograms from disk...")
    hist_folder = "D:/FYPSeagullClassification01/Features_Analysis/Outputs/LBP_Histograms/Glaucous_Winged_Gull"  # Adjust to your real folder path
    species_data = load_lbp_histograms(hist_folder)

    # Where to save results
    output_dir = "Advanced_Analysis_Results"
    os.makedirs(output_dir, exist_ok=True)

    print("\nRunning Enhanced Texture Analysis...")
    analysis_results = enhanced_texture_analysis(species_data, output_dir)

    print("\nRunning Multi-Scale LBP Comparison (example on 'wing')...")
    region_name = "wing"
    multi_scale_results = compare_multi_scale_lbp(species_data, region_name, output_dir, radii=[1, 2, 3])

    print("\nAll done. Check the output folder for results.")


if __name__ == "__main__":
    main()
