"""
Publication-Safe PCA Visualization
No controversial biplot overlays - just clean, interpretable figures
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind


def publication_pca_analysis(df, features, true_labels, species_names,
                             feature_names, output_dir):
    """
    Publication-ready PCA analysis with separate, unambiguous visualizations
    """

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Run PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Extract key statistics
    explained_var = pca.explained_variance_ratio_
    loadings = pca.components_.T

    print("=" * 70)
    print("PCA ANALYSIS - PUBLICATION STATISTICS")
    print("=" * 70)
    print(f"\nVariance Explained:")
    print(f"  PC1: {explained_var[0] * 100:.2f}%")
    print(f"  PC2: {explained_var[1] * 100:.2f}%")
    print(f"  Total (2 PCs): {explained_var.sum() * 100:.2f}%")

    # Statistical tests on PCs
    print(f"\n{'Component':<10} {'t-statistic':<15} {'p-value':<12} {'Cohen d':<10}")
    print("-" * 70)
    for i in range(2):
        sb_scores = X_pca[true_labels == 0, i]
        gw_scores = X_pca[true_labels == 1, i]
        t_stat, p_val = ttest_ind(sb_scores, gw_scores)
        cohen_d = (np.mean(sb_scores) - np.mean(gw_scores)) / \
                  np.sqrt((np.std(sb_scores) ** 2 + np.std(gw_scores) ** 2) / 2)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"PC{i + 1:<9} {t_stat:<15.3f} {p_val:<12.4e} {cohen_d:<10.3f} {sig}")

    # Create publication figures
    fig = plt.figure(figsize=(16, 6))

    # ====================================================================
    # FIGURE A: PCA Scatter Plot (Data Only - No Arrows)
    # ====================================================================
    ax1 = plt.subplot(1, 3, 1)

    colors = ['#1f77b4', '#ff7f0e']
    markers = ['o', 's']

    for i, (label_id, species_name) in enumerate(species_names.items()):
        mask = true_labels == label_id

        # Calculate 95% confidence ellipse
        from matplotlib.patches import Ellipse
        mean = X_pca[mask].mean(axis=0)
        cov = np.cov(X_pca[mask].T)

        # Chi-square value for 95% confidence (2 DOF)
        chi2_val = 5.991
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(chi2_val * eigenvalues)

        ellipse = Ellipse(mean, width, height, angle=angle,
                          facecolor=colors[i], alpha=0.2, edgecolor=colors[i],
                          linewidth=2, linestyle='--')
        ax1.add_patch(ellipse)

        # Plot points
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=species_name.replace('_', ' '),
                    alpha=0.6, s=50, c=colors[i],
                    marker=markers[i], edgecolors='black', linewidth=0.5)

    ax1.set_xlabel(f'PC1 ({explained_var[0] * 100:.1f}% variance)',
                   fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({explained_var[1] * 100:.1f}% variance)',
                   fontsize=11, fontweight='bold')
    ax1.set_title('(A) PCA Score Plot', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color='k', linewidth=0.5, alpha=0.3)

    # ====================================================================
    # FIGURE B: Loading Plot (Arrows Only - No Data)
    # ====================================================================
    ax2 = plt.subplot(1, 3, 2)

    arrow_colors = ['#d62728', '#2ca02c', '#9467bd']

    for i, feature in enumerate(feature_names):
        # Plot arrow
        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.06, head_length=0.06,
                  fc=arrow_colors[i], ec=arrow_colors[i],
                  linewidth=2.5, alpha=0.8, length_includes_head=True)

        # Add label
        label_pos = 1.15
        ax2.text(loadings[i, 0] * label_pos, loadings[i, 1] * label_pos,
                 feature.replace('_', ' '), fontsize=9, ha='center', va='center',
                 fontweight='bold', color=arrow_colors[i],
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=arrow_colors[i], linewidth=1.5, alpha=0.9))

    # Set axis limits to [-1, 1] (correlation range)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel('PC1 Loading', fontsize=11, fontweight='bold')
    ax2.set_ylabel('PC2 Loading', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Feature Loadings', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.axvline(0, color='k', linewidth=0.5)

    # Add correlation circle
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='gray',
                        linestyle='--', linewidth=1, alpha=0.5)
    ax2.add_patch(circle)
    ax2.set_aspect('equal')

    # ====================================================================
    # FIGURE C: Loading Table (For Paper)
    # ====================================================================
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')

    # Create loading table
    table_data = []
    table_data.append(['Feature', 'PC1', 'PC2', '|Loading|'])
    table_data.append(['─' * 30, '─' * 8, '─' * 8, '─' * 10])

    for i, feature in enumerate(feature_names):
        mag = np.sqrt(loadings[i, 0] ** 2 + loadings[i, 1] ** 2)
        table_data.append([
            feature.replace('_', ' ')[:28],
            f'{loadings[i, 0]:>7.3f}',
            f'{loadings[i, 1]:>7.3f}',
            f'{mag:>9.3f}'
        ])

    table_data.append(['─' * 30, '─' * 8, '─' * 8, '─' * 10])
    table_data.append(['Explained variance',
                       f'{explained_var[0] * 100:>6.1f}%',
                       f'{explained_var[1] * 100:>6.1f}%',
                       f'{explained_var.sum() * 100:>8.1f}%'])

    # Display table
    table = ax3.table(cellText=table_data, cellLoc='left',
                      bbox=[0, 0.2, 1, 0.8], edges='horizontal')
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', fontsize=10)
        cell.set_facecolor('#e0e0e0')

    # Style data rows
    for i in range(2, len(table_data) - 2):
        for j in range(4):
            cell = table[(i, j)]
            if j == 0:
                cell.set_text_props(fontsize=8)

    # Style last row
    for i in range(4):
        cell = table[(len(table_data) - 1, i)]
        cell.set_text_props(weight='bold', fontsize=9)
        cell.set_facecolor('#f0f0f0')

    ax3.set_title('(C) Loading Coefficients', fontsize=12, fontweight='bold',
                  pad=20)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/publication_figure_pca.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/publication_figure_pca.pdf', bbox_inches='tight')
    plt.close()

    # ====================================================================
    # Save supplementary table (CSV for paper)
    # ====================================================================
    loadings_df = pd.DataFrame(
        loadings,
        columns=['PC1', 'PC2'],
        index=[f.replace('_', ' ') for f in feature_names]
    )
    loadings_df['Loading_Magnitude'] = np.sqrt(loadings_df['PC1'] ** 2 + loadings_df['PC2'] ** 2)
    loadings_df['Variance_PC1'] = explained_var[0] * 100
    loadings_df['Variance_PC2'] = explained_var[1] * 100

    loadings_df.to_csv(f'{output_dir}/table_pca_loadings.csv')

    print("\n" + "=" * 70)
    print("LOADING COEFFICIENTS (for table in paper)")
    print("=" * 70)
    print(loadings_df.to_string())

    # ====================================================================
    # Statistical interpretation
    # ====================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE FOR PAPER")
    print("=" * 70)

    print("\nPC1 interpretation:")
    pc1_loadings = loadings[:, 0]
    dominant_pc1 = np.argmax(np.abs(pc1_loadings))
    print(f"  Primary contributor: {feature_names[dominant_pc1]}")
    print(f"  Loading: {pc1_loadings[dominant_pc1]:.3f}")
    print(f"  Interpretation: {'Positive' if pc1_loadings[dominant_pc1] > 0 else 'Negative'} association")

    print("\nPC2 interpretation:")
    pc2_loadings = loadings[:, 1]
    dominant_pc2 = np.argmax(np.abs(pc2_loadings))
    print(f"  Primary contributor: {feature_names[dominant_pc2]}")
    print(f"  Loading: {pc2_loadings[dominant_pc2]:.3f}")
    print(f"  Interpretation: {'Positive' if pc2_loadings[dominant_pc2] > 0 else 'Negative'} association")

    print("\n" + "=" * 70)
    print("✓ PUBLICATION FIGURES SAVED")
    print("=" * 70)
    print(f"  • publication_figure_pca.png (for paper)")
    print(f"  • publication_figure_pca.pdf (for submission)")
    print(f"  • table_pca_loadings.csv (supplementary table)")

    return X_pca, pca, loadings_df


# ====================================================================
# EXAMPLE USAGE
# ====================================================================
if __name__ == "__main__":
    import os

    output_dir = "publication_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load your data
    file_path = r"D:\FYPSeagullClassification01\Clustering\dark_pixel_results_all_images.csv"
    df = pd.read_csv(file_path)

    if 'method3_percentage' in df.columns:
        df = df.rename(columns={'method3_percentage': 'percentage_dark_pixels_wingtip'})

    features = df[['mean_wing_intensity', 'mean_wingtip_intensity', 'percentage_dark_pixels_wingtip']]

    species_mapping = {'Slaty_Backed_Gull': 0, 'Glaucous_Winged_Gull': 1}
    true_labels = df['species'].map(species_mapping).values
    species_names = {0: 'Slaty_Backed_Gull', 1: 'Glaucous_Winged_Gull'}

    feature_names = ['mean_wing_intensity', 'mean_wingtip_intensity',
                     'percentage_dark_pixels_wingtip']

    # Run publication-safe analysis
    X_pca, pca, loadings_df = publication_pca_analysis(
        df, features, true_labels, species_names, feature_names, output_dir
    )

    print("\n✓ Analysis complete - ready for publication!")