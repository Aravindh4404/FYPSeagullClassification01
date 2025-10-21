from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import ttest_ind, chi2
import sys

output_dir = os.path.join(os.path.dirname(__file__), "clustering_results")
os.makedirs(output_dir, exist_ok=True)
log_file = open(os.path.join(output_dir, "console_output.txt"), "w", encoding='utf-8')
sys.stdout = log_file

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

# Set a consistent random state for reproducibility
RANDOM_STATE = 42


def load_data(file_path):
    """Load and prepare data with proper column naming"""
    df = pd.read_csv(file_path)

    if 'method3_percentage' in df.columns:
        df = df.rename(columns={'method3_percentage': 'percentage_dark_pixels_wingtip'})

    features = df[['mean_wing_intensity', 'mean_wingtip_intensity', 'percentage_dark_pixels_wingtip']]

    if 'species' in df.columns:
        species_mapping = {'Slaty_Backed_Gull': 0, 'Glaucous_Winged_Gull': 1}
        true_labels = df['species'].map(species_mapping)
        species_names = {0: 'Slaty_Backed_Gull', 1: 'Glaucous_Winged_Gull'}
    else:
        true_labels = None
        species_names = None

    return df, features, true_labels, species_names


def preprocess_data(features):
    """Standardize features using z-score normalization"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return X_scaled, scaler


def apply_pca(X_scaled, n_components=2):
    """Apply PCA and return transformed data with statistics"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    return X_pca, pca, explained_variance


def test_pc_differences(X_pca, true_labels, species_names):
    """Statistical tests for species differences on each PC"""
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS: Species Differences on Principal Components")
    print("=" * 70)

    results = []
    for i in range(X_pca.shape[1]):
        sb_scores = X_pca[true_labels == 0, i]
        gw_scores = X_pca[true_labels == 1, i]

        t_stat, p_val = ttest_ind(sb_scores, gw_scores)

        cohen_d = (np.mean(sb_scores) - np.mean(gw_scores)) / \
                  np.sqrt((np.std(sb_scores) ** 2 + np.std(gw_scores) ** 2) / 2)

        print(f"\nPC{i + 1}:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4e}")
        print(
            f"  Cohen's d: {cohen_d:.4f} ({'large' if abs(cohen_d) > 0.8 else 'medium' if abs(cohen_d) > 0.5 else 'small'} effect)")
        print(f"  SB mean: {np.mean(sb_scores):.3f} ± {np.std(sb_scores):.3f}")
        print(f"  GW mean: {np.mean(gw_scores):.3f} ± {np.std(gw_scores):.3f}")

        results.append({
            'PC': i + 1,
            't_statistic': t_stat,
            'p_value': p_val,
            'cohens_d': cohen_d,
            'SB_mean': np.mean(sb_scores),
            'GW_mean': np.mean(gw_scores)
        })

    return pd.DataFrame(results)


def visualize_pca_biplot_corrected(X_pca, true_labels, species_names, pca,
                                   feature_names, explained_variance, save_path=None):
    """
    CORRECTED PCA biplot with proper scaling and interpretation
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot data points
    colors = ['#1f77b4', '#ff7f0e']
    for i, (label_id, species_name) in enumerate(species_names.items()):
        mask = true_labels == label_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=species_name, alpha=0.6, s=60, c=colors[i],
                   edgecolors='black', linewidth=0.5)

    # Calculate correlation loadings (proper method)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # CORRECTED SCALING: Scale arrows to fit data range proportionally
    data_range = np.ptp(X_pca, axis=0)  # Peak-to-peak (max - min) on each axis
    loading_range = np.max(np.abs(loadings), axis=0)

    # Scale so largest arrow is ~70% of data range (prevents overcrowding)
    scale = np.min(data_range / loading_range) * 0.7

    print("\n" + "=" * 70)
    print("PCA BIPLOT - LOADING STATISTICS")
    print("=" * 70)
    print(f"\nData range (PC1, PC2): {data_range}")
    print(f"Loading range (PC1, PC2): {loading_range}")
    print(f"Calculated scale factor: {scale:.3f}")
    print(f"\nNote: Scale factor is calculated to fit arrows within 70% of data range")

    # Plot feature vectors with proper scaling
    arrow_colors = ['red', 'darkgreen', 'purple']

    print("\n" + "-" * 70)
    print("Feature Loadings (Correlation with PCs):")
    print("-" * 70)
    print(f"{'Feature':<40} {'PC1':>10} {'PC2':>10} {'|Loading|':>12}")
    print("-" * 70)

    for i, feature in enumerate(feature_names):
        corr_pc1 = loadings[i, 0]
        corr_pc2 = loadings[i, 1]
        loading_magnitude = np.sqrt(corr_pc1 ** 2 + corr_pc2 ** 2)

        print(f"{feature:<40} {corr_pc1:>10.3f} {corr_pc2:>10.3f} {loading_magnitude:>12.3f}")

        # Draw arrow
        ax.arrow(0, 0,
                 loadings[i, 0] * scale, loadings[i, 1] * scale,
                 head_width=0.15, head_length=0.15,
                 fc=arrow_colors[i], ec=arrow_colors[i],
                 linewidth=2.5, alpha=0.8, length_includes_head=True, zorder=100)

        # Label with correlation values
        label_text = f"{feature}\n(r₁={corr_pc1:.2f}, r₂={corr_pc2:.2f})"
        ax.text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15,
                label_text, fontsize=10, ha='center', va='center',
                fontweight='bold', color=arrow_colors[i],
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=arrow_colors[i], alpha=0.9))

    print("-" * 70)
    print("\nInterpretation:")
    print("  • Loading values are correlations between original features and PCs")
    print("  • Range: -1 (perfect negative correlation) to +1 (perfect positive)")
    print("  • |r| > 0.7: Strong relationship")
    print("  • |r| = 0.5-0.7: Moderate relationship")
    print("  • |r| < 0.5: Weak relationship")

    ax.set_xlabel(f'PC1 ({explained_variance[0] * 100:.1f}% variance)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance[1] * 100:.1f}% variance)',
                  fontsize=12, fontweight='bold')
    ax.set_title('PCA Biplot with Feature Loadings (Correlation-based)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Return loadings as dataframe
    loadings_df = pd.DataFrame(
        loadings,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    loadings_df['Loading_Magnitude'] = np.sqrt(loadings_df['PC1'] ** 2 + loadings_df['PC2'] ** 2)

    return loadings_df


def identify_overlap_zone_birds_corrected(X_pca, X_scaled, df, true_labels,
                                          species_names, method='lda',
                                          threshold=0.2, save_path=None):
    """
    CORRECTED overlap zone identification using LDA posterior probabilities

    Parameters:
    - method: 'lda' (Linear Discriminant Analysis) or 'gmm' (Gaussian Mixture Model)
    - threshold: probability threshold for overlap (0.3-0.7 = uncertain if threshold=0.2)
    """
    print("\n" + "=" * 70)
    print(f"OVERLAP ZONE IDENTIFICATION - Method: {method.upper()}")
    print("=" * 70)

    if method == 'lda':
        # Use LDA for probabilistic classification
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_scaled, true_labels)

        # Get posterior probabilities
        probs = lda.predict_proba(X_scaled)

        # Calculate uncertainty (distance from decision boundary)
        # Prob close to 0.5 = high uncertainty
        uncertainty = np.abs(probs[:, 0] - 0.5)

        # Birds in overlap zone: probability between 0.3 and 0.7
        overlap_mask = uncertainty <= threshold

        print(f"\nLDA Classification Results:")
        print(f"  Decision boundary: posterior probability = 0.5")
        print(f"  Overlap threshold: |p - 0.5| ≤ {threshold}")
        print(f"  This means: 0.3 ≤ p ≤ 0.7 are considered uncertain")

    elif method == 'gmm':
        # Use Gaussian Mixture Model
        gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
        gmm.fit(X_scaled)

        probs = gmm.predict_proba(X_scaled)
        uncertainty = np.abs(probs[:, 0] - 0.5)
        overlap_mask = uncertainty <= threshold

        print(f"\nGMM Classification Results:")
        print(f"  Using mixture model probabilities")
        print(f"  Overlap threshold: |p - 0.5| ≤ {threshold}")

    # Analyze overlap zone
    overlap_df = df.copy()
    overlap_df['PC1'] = X_pca[:, 0]
    overlap_df['PC2'] = X_pca[:, 1]
    overlap_df['Prob_SB'] = probs[:, 0]
    overlap_df['Prob_GW'] = probs[:, 1]
    overlap_df['Uncertainty'] = uncertainty
    overlap_df['In_Overlap_Zone'] = overlap_mask
    overlap_df['True_Species'] = [species_names[label] for label in true_labels]

    # Get overlap birds sorted by uncertainty
    overlap_birds = overlap_df[overlap_mask].sort_values('Uncertainty')

    print(f"\n{'Species':<25} {'In Overlap':>12} {'Total':>8} {'Percentage':>12}")
    print("-" * 70)
    for label_id, species_name in species_names.items():
        mask = (true_labels == label_id) & overlap_mask
        count = mask.sum()
        total = (true_labels == label_id).sum()
        pct = count / total * 100 if total > 0 else 0
        print(f"{species_name:<25} {count:>12} {total:>8} {pct:>11.1f}%")

    print("-" * 70)
    print(
        f"{'Total Overlap Birds':<25} {overlap_mask.sum():>12} {len(df):>8} {overlap_mask.sum() / len(df) * 100:>11.1f}%")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot non-overlap birds (low opacity)
    non_overlap_mask = ~overlap_mask
    for label_id, species_name in species_names.items():
        mask = (true_labels == label_id) & non_overlap_mask
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'{species_name} (typical)', alpha=0.4, s=50,
                   edgecolors='gray', linewidth=0.3)

    # Plot overlap zone birds (high opacity, special markers)
    for label_id, species_name in species_names.items():
        mask = (true_labels == label_id) & overlap_mask
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'{species_name} (overlap)', alpha=0.9, s=120,
                   marker='D', edgecolors='red', linewidth=2)

    if method == 'lda':
        # Draw LDA decision boundary
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                             np.linspace(ylim[0], ylim[1], 100))

        # Transform grid to original space for LDA prediction
        # This is approximate - for exact boundary, would need inverse PCA transform
        ax.text(0.02, 0.98, f'Overlap threshold: p ∈ [0.3, 0.7]',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title(f'Birds in Overlap Zone ({method.upper()} Method)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save CSV
        csv_path = save_path.replace('.png', '.csv')
        overlap_birds.to_csv(csv_path, index=False)
        print(f"\n✓ Overlap zone birds saved to: {csv_path}")
    else:
        plt.show()

    return overlap_df, overlap_birds


def locate_hokkaido_birds(X_pca, df, true_labels, species_names, explained_variance,
                          hokkaido_filenames, filename_column='filename', save_path=None):
    """
    Locate and highlight specific Hokkaido birds on PCA plot
    """
    if isinstance(hokkaido_filenames, str):
        hokkaido_filenames = [hokkaido_filenames]

    if filename_column not in df.columns:
        print(f"\n❌ ERROR: Column '{filename_column}' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    print("\n" + "=" * 70)
    print("LOCATING HOKKAIDO BIRDS IN PCA SPACE")
    print("=" * 70)

    results = []
    found_mask = np.zeros(len(df), dtype=bool)

    for filename in hokkaido_filenames:
        mask = df[filename_column] == filename
        if mask.sum() == 0:
            print(f"\n❌ NOT FOUND: '{filename}'")
        else:
            found_mask = found_mask | mask.values
            idx = df[mask].index[0]
            species_id = true_labels.iloc[idx] if isinstance(true_labels, pd.Series) else true_labels[idx]
            species = species_names[species_id]
            pc1 = X_pca[idx, 0]
            pc2 = X_pca[idx, 1]

            results.append({
                'filename': filename,
                'species': species,
                'PC1': pc1,
                'PC2': pc2,
                'index': idx
            })

            print(f"\n✓ FOUND: '{filename}'")
            print(f"   Species: {species}")
            print(f"   PCA Position: PC1={pc1:.3f}, PC2={pc2:.3f}")

    if not results:
        print("\n⚠️  No Hokkaido birds found!")
        return None

    results_df = pd.DataFrame(results)

    # Visualize
    fig, ax = plt.subplots(figsize=(14, 10))

    colors = {'Slaty_Backed_Gull': '#1f77b4', 'Glaucous_Winged_Gull': '#ff7f0e'}

    for label_id, species_name in species_names.items():
        mask = (true_labels == label_id) & (~found_mask)
        color = colors.get(species_name, 'gray')
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'{species_name} (other)', alpha=0.3, s=50,
                   c=color, edgecolors='gray', linewidth=0.3)

    # Highlight Hokkaido birds
    hokkaido_mask = found_mask
    ax.scatter(X_pca[hokkaido_mask, 0], X_pca[hokkaido_mask, 1],
               marker='*', s=600, c='red',
               edgecolors='black', linewidth=2.5,
               zorder=100, alpha=1.0,
               label='Hokkaido SB')

    # Add labels
    for _, row in results_df.iterrows():
        ax.annotate(row['filename'],
                    xy=(row['PC1'], row['PC2']),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                              edgecolor='black', alpha=0.85),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                    lw=1.5, color='black'))

    ax.set_xlabel(f'PC1 ({explained_variance[0] * 100:.1f}% variance)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance[1] * 100:.1f}% variance)',
                  fontsize=13, fontweight='bold')
    ax.set_title('Hokkaido Slaty-Backed Gulls vs Other Populations',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        csv_path = save_path.replace('.png', '_details.csv')
        detailed_results = df[found_mask].copy()
        detailed_results['PC1'] = X_pca[found_mask, 0]
        detailed_results['PC2'] = X_pca[found_mask, 1]
        detailed_results.to_csv(csv_path, index=False)
        print(f"\n✓ Hokkaido bird details saved to: {csv_path}")
    else:
        plt.show()

    print(f"\n✓ Successfully located {len(results)} Hokkaido bird(s)")

    return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CORRECTED GULL CLUSTERING ANALYSIS")
    print("=" * 70)

    # STEP 1: Load data
    file_path = r"D:\FYPSeagullClassification01\Clustering\dark_pixel_results_all_images.csv"

    df, features, true_labels, species_names = load_data(file_path)
    print(f"\n✓ Loaded {len(df)} birds")
    print(f"✓ Features: {features.columns.tolist()}")

    # STEP 2: Preprocess and PCA
    X_scaled, scaler = preprocess_data(features)
    X_pca, pca, explained_variance = apply_pca(X_scaled, n_components=2)

    feature_names = ['mean_wing_intensity', 'mean_wingtip_intensity',
                     'percentage_dark_pixels_wingtip']

    print(f"\n✓ PCA complete")
    print(f"  PC1 explains {explained_variance[0] * 100:.1f}% variance")
    print(f"  PC2 explains {explained_variance[1] * 100:.1f}% variance")
    print(f"  Total variance explained: {explained_variance.sum() * 100:.1f}%")

    # STEP 3: Statistical testing
    test_results = test_pc_differences(X_pca, true_labels, species_names)
    test_results.to_csv(os.path.join(output_dir, "statistical_tests.csv"), index=False)

    # STEP 4: Corrected PCA biplot
    print("\n" + "=" * 70)
    print("1. CREATING CORRECTED PCA BIPLOT")
    print("=" * 70)

    loadings_df = visualize_pca_biplot_corrected(
        X_pca=X_pca,
        true_labels=true_labels,
        species_names=species_names,
        pca=pca,
        feature_names=feature_names,
        explained_variance=explained_variance,
        save_path=os.path.join(output_dir, "1_pca_biplot_corrected.png")
    )
    loadings_df.to_csv(os.path.join(output_dir, "pca_loadings.csv"))

    # STEP 5: Corrected overlap zone identification
    print("\n" + "=" * 70)
    print("2. IDENTIFYING OVERLAP ZONE (LDA METHOD)")
    print("=" * 70)

    overlap_df, overlap_birds = identify_overlap_zone_birds_corrected(
        X_pca=X_pca,
        X_scaled=X_scaled,
        df=df,
        true_labels=true_labels,
        species_names=species_names,
        method='lda',
        threshold=0.2,
        save_path=os.path.join(output_dir, "2_overlap_zone_lda.png")
    )

    # STEP 6: Hokkaido birds
    print("\n" + "=" * 70)
    print("3. HIGHLIGHTING HOKKAIDO BIRDS")
    print("=" * 70)

    hokkaido_filenames = [
        "mod-0R8A4927.JPG",
        "mod-0R8A5109.JPG",
        "mod-0R8A5217.JPG",
    ]

    hokkaido_results = locate_hokkaido_birds(
        X_pca=X_pca,
        df=df,
        true_labels=true_labels,
        species_names=species_names,
        explained_variance=explained_variance,
        hokkaido_filenames=hokkaido_filenames,
        filename_column='image_name',
        save_path=os.path.join(output_dir, "3_hokkaido_birds_highlighted.png")
    )

    # Final summary
    print("\n" + "=" * 70)
    print("✓ CORRECTED ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {output_dir}")
    print("\n📊 Files created:")
    print("   1. 1_pca_biplot_corrected.png - Properly scaled PCA biplot")
    print("   2. pca_loadings.csv - Feature correlation loadings")
    print("   3. statistical_tests.csv - t-tests for PC differences")
    print("   4. 2_overlap_zone_lda.png - LDA-based overlap identification")
    print("   5. 2_overlap_zone_lda.csv - Overlap birds with probabilities")
    print("   6. 3_hokkaido_birds_highlighted.png - Hokkaido bird locations")
    print("\n" + "=" * 70)

log_file.close()