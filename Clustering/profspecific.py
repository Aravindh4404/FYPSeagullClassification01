from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
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


# Function to load and prepare data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Rename method3_percentage to a more descriptive name
    if 'method3_percentage' in df.columns:
        df = df.rename(columns={'method3_percentage': 'percentage_dark_pixels_wingtip'})

    # Updated feature columns with new name
    features = df[['mean_wing_intensity', 'mean_wingtip_intensity', 'percentage_dark_pixels_wingtip']]

    # Create mapping if it doesn't exist already
    if 'species' in df.columns:
        species_mapping = {'Slaty_Backed_Gull': 0, 'Glaucous_Winged_Gull': 1}
        true_labels = df['species'].map(species_mapping)
        # Keep the original species names for later use
        species_names = {0: 'Slaty_Backed_Gull', 1: 'Glaucous_Winged_Gull'}
    else:
        true_labels = None
        species_names = None

    return df, features, true_labels, species_names


# Function to standardize features
def preprocess_data(features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return X_scaled, scaler


# Function to reduce dimensionality with PCA for visualization
def apply_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    return X_pca, pca, explained_variance


# ============================================================================
# NEW FUNCTION 1: PCA BIPLOT WITH FEATURE VECTORS (What professor asked for!)
# ============================================================================
def visualize_pca_biplot(X_pca, true_labels, species_names, pca, feature_names, explained_variance, save_path=None):
    """
    Create a PCA biplot showing both data points and feature loading vectors
    This shows "the three variables shown as vectors, so we can see the directions"
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each species with different colors
    colors = ['#1f77b4', '#ff7f0e']  # blue for SB, orange for GW
    for i, (label_id, species_name) in enumerate(species_names.items()):
        mask = true_labels == label_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=species_name, alpha=0.6, s=60, c=colors[i], edgecolors='black', linewidth=0.5)

    # Get the loadings (components)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Scale factor for arrows (adjust based on your data)
    scale = 3.5

    # Plot feature vectors
    arrow_colors = ['red', 'darkgreen', 'purple']
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0,
                 loadings[i, 0] * scale, loadings[i, 1] * scale,
                 head_width=0.15, head_length=0.15, fc=arrow_colors[i], ec=arrow_colors[i],
                 linewidth=2.5, alpha=0.8, length_includes_head=True)

        # Add feature labels at the end of arrows
        ax.text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15,
                feature, fontsize=11, ha='center', va='center',
                fontweight='bold', color=arrow_colors[i],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=arrow_colors[i], alpha=0.8))

    ax.set_xlabel(f'PC1 ({explained_variance[0] * 100:.1f}% variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance[1] * 100:.1f}% variance)', fontsize=12, fontweight='bold')
    ax.set_title('PCA Biplot with Feature Loadings', fontsize=14, fontweight='bold')
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

    # Print component loadings table
    print("\n=== PCA Component Loadings ===")
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    print(loadings_df.to_string())

    return loadings_df


# ============================================================================
# NEW FUNCTION 2: IDENTIFY OVERLAP ZONE BIRDS (What professor asked for!)
# ============================================================================
def identify_overlap_zone_birds(X_pca, df, true_labels, species_names, threshold_percentile=15, save_path=None):
    """
    Identify birds in the overlap zone between species clusters
    "Pull out the images of the birds I have circled as these ones sit in overlap zones"

    Parameters:
    - threshold_percentile: percentage of birds closest to the boundary (default 15%)
    """
    # Calculate centroids for each species
    centroids = {}
    for label_id, species_name in species_names.items():
        mask = true_labels == label_id
        centroids[label_id] = X_pca[mask].mean(axis=0)

    # Calculate distance from decision boundary (perpendicular distance to line connecting centroids)
    centroid_0 = centroids[0]
    centroid_1 = centroids[1]

    # Vector between centroids
    centroid_diff = centroid_1 - centroid_0

    # For each point, calculate perpendicular distance to boundary (midpoint between centroids)
    midpoint = (centroid_0 + centroid_1) / 2

    # Distance from midpoint along the line connecting centroids
    distances_to_boundary = []
    for point in X_pca:
        # Project point onto line connecting centroids
        vec_to_point = point - centroid_0
        projection_length = np.dot(vec_to_point, centroid_diff) / np.linalg.norm(centroid_diff)

        # Distance from midpoint (0 = on boundary, positive/negative = on either side)
        centroid_distance = np.linalg.norm(centroid_diff) / 2
        distance_from_boundary = abs(projection_length - centroid_distance)
        distances_to_boundary.append(distance_from_boundary)

    distances_to_boundary = np.array(distances_to_boundary)

    # Identify overlap zone birds (closest to boundary)
    threshold = np.percentile(distances_to_boundary, threshold_percentile)
    overlap_mask = distances_to_boundary <= threshold

    # Create analysis dataframe
    overlap_df = df.copy()
    overlap_df['PC1'] = X_pca[:, 0]
    overlap_df['PC2'] = X_pca[:, 1]
    overlap_df['Distance_to_Boundary'] = distances_to_boundary
    overlap_df['In_Overlap_Zone'] = overlap_mask
    overlap_df['True_Species'] = [species_names[label] for label in true_labels]

    # Save overlap zone birds to CSV
    overlap_birds = overlap_df[overlap_mask].sort_values('Distance_to_Boundary')

    if save_path:
        csv_path = save_path.replace('.png', '.csv')
        overlap_birds.to_csv(csv_path, index=False)
        print(f"\n✓ Overlap zone birds saved to: {csv_path}")
        print(f"✓ Number of birds in overlap zone: {len(overlap_birds)} ({len(overlap_birds) / len(df) * 100:.1f}%)")

    # Visualize overlap zone
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot non-overlap birds
    non_overlap_mask = ~overlap_mask
    for label_id, species_name in species_names.items():
        mask = (true_labels == label_id) & non_overlap_mask
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'{species_name} (typical)', alpha=0.5, s=60, edgecolors='black', linewidth=0.5)

    # Plot overlap zone birds with special markers
    for label_id, species_name in species_names.items():
        mask = (true_labels == label_id) & overlap_mask
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'{species_name} (overlap zone)', alpha=0.9, s=120,
                   marker='D', edgecolors='red', linewidth=2)

    # Draw boundary line
    boundary_vector = np.array([-centroid_diff[1], centroid_diff[0]])  # perpendicular
    boundary_vector = boundary_vector / np.linalg.norm(boundary_vector)
    line_length = 10
    ax.plot([midpoint[0] - boundary_vector[0] * line_length, midpoint[0] + boundary_vector[0] * line_length],
            [midpoint[1] - boundary_vector[1] * line_length, midpoint[1] + boundary_vector[1] * line_length],
            'k--', linewidth=2, alpha=0.5, label='Decision Boundary')

    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title('Birds in Overlap Zone (Intermediate Features)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Print summary
    print("\n=== Overlap Zone Analysis ===")
    print(f"Threshold distance: {threshold:.3f}")
    print(f"\nBreakdown by species:")
    for label_id, species_name in species_names.items():
        mask = (true_labels == label_id) & overlap_mask
        count = mask.sum()
        total = (true_labels == label_id).sum()
        print(f"{species_name}: {count} birds ({count / total * 100:.1f}% of total {species_name})")

    return overlap_df, overlap_birds


# ============================================================================
# NEW FUNCTION 3: LOCATE SPECIFIC HOKKAIDO BIRDS (What professor asked for!)
# ============================================================================
def locate_hokkaido_birds(X_pca, df, true_labels, species_names, explained_variance,
                          hokkaido_filenames, filename_column='filename', save_path=None):
    """
    Locate and highlight specific Hokkaido birds by their filename on the PCA plot
    "Can you highlight these birds on the PCA so we can see how they compare to GW"

    Parameters:
    - hokkaido_filenames: List of filenames (or single filename string) for Hokkaido birds
    - filename_column: Name of column containing filenames in your dataframe
    """

    # Convert single filename to list
    if isinstance(hokkaido_filenames, str):
        hokkaido_filenames = [hokkaido_filenames]

    # Check if column exists
    if filename_column not in df.columns:
        print(f"\n❌ ERROR: Column '{filename_column}' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        print("\nFirst few rows of dataframe:")
        print(df.head())
        return None

    print("\n" + "=" * 70)
    print("LOCATING HOKKAIDO BIRDS IN PCA SPACE")
    print("=" * 70)

    # Find birds in dataset
    results = []
    found_mask = np.zeros(len(df), dtype=bool)

    for filename in hokkaido_filenames:
        mask = df[filename_column] == filename
        if mask.sum() == 0:
            print(f"\n❌ NOT FOUND: '{filename}'")
            print(f"   Hint: Check exact spelling and file extension")
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
            print(f"   Dataset row index: {idx}")

    if not results:
        print("\n⚠️  No Hokkaido birds found! Please check your filenames.")
        print(f"\nFirst 10 filenames in your dataset:")
        print(df[filename_column].head(10).tolist())
        return None

    results_df = pd.DataFrame(results)

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot all birds with reduced opacity
    colors = {'Slaty_Backed_Gull': '#1f77b4', 'Glaucous_Winged_Gull': '#ff7f0e'}

    for label_id, species_name in species_names.items():
        mask = (true_labels == label_id) & (~found_mask)
        color = colors.get(species_name, 'gray')
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'{species_name} (other populations)', alpha=0.3, s=50,
                   c=color, edgecolors='gray', linewidth=0.3)

    # Highlight Hokkaido birds with large prominent stars
    hokkaido_mask = found_mask
    hokkaido_species = [species_names[label] for label in true_labels[hokkaido_mask]]

    ax.scatter(X_pca[hokkaido_mask, 0], X_pca[hokkaido_mask, 1],
               marker='*', s=600, c='red',
               edgecolors='black', linewidth=2.5,
               zorder=100, alpha=1.0,
               label='Hokkaido SB')

    # Add labels for each Hokkaido bird
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
        print(f"\n📊 Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()

    # Save detailed results to CSV
    if save_path:
        csv_path = save_path.replace('.png', '_details.csv')

        # Add all features from original dataframe
        detailed_results = df[found_mask].copy()
        detailed_results['PC1'] = X_pca[found_mask, 0]
        detailed_results['PC2'] = X_pca[found_mask, 1]

        detailed_results.to_csv(csv_path, index=False)
        print(f"📄 Hokkaido bird details saved to: {csv_path}")

    print("\n" + "=" * 70)
    print(f"✓ Successfully located {len(results)} Hokkaido bird(s)")
    print("=" * 70)

    return results_df


# ============================================================================
# ALL YOUR ORIGINAL FUNCTIONS BELOW (keeping them for compatibility)
# ============================================================================

def visualize_pca_by_species(X_pca, true_labels, species_names, explained_variance, save_path=None):
    plt.figure(figsize=(10, 6))
    for label_id, species_name in species_names.items():
        mask = true_labels == label_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=species_name, alpha=0.7, s=50)
    plt.xlabel(f'PCA Component 1 ({explained_variance[0] * 100:.1f}% variance)')
    plt.ylabel(f'PCA Component 2 ({explained_variance[1] * 100:.1f}% variance)')
    plt.title('PCA Visualization by Species')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def evaluate_clustering(X_scaled, clusters, true_labels=None):
    silhouette = silhouette_score(X_scaled, clusters)
    results = {"silhouette_score": silhouette}
    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, clusters)
        results["adjusted_rand_index"] = ari
    return results


def visualize_clusters(X_pca, clusters, centroids=None, title="Clustering Results", save_path=None):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='X', c='red', label='Centroids')
        plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ============================================================================
# MAIN EXECUTION - THIS IS WHERE YOU CUSTOMIZE FOR YOUR DATA
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING GULL CLUSTERING ANALYSIS")
    print("=" * 70)

    # ========================================
    # STEP 1: LOAD YOUR DATA
    # ========================================
    file_path = r"D:\FYPSeagullClassification01\Clustering\dark_pixel_results_all_images.csv"  # ← CHANGE THIS to your CSV file path

    df, features, true_labels, species_names = load_data(file_path)

    print(f"\n✓ Loaded {len(df)} birds")
    print(f"✓ Features: {features.columns.tolist()}")
    print(f"\n📋 Dataset columns: {df.columns.tolist()}")

    # ========================================
    # STEP 2: PREPROCESS AND RUN PCA
    # ========================================
    X_scaled, scaler = preprocess_data(features)
    X_pca, pca, explained_variance = apply_pca(X_scaled, n_components=2)

    feature_names = ['mean_wing_intensity', 'mean_wingtip_intensity', 'percentage_dark_pixels_wingtip']

    print(f"\n✓ PCA complete")
    print(f"  PC1 explains {explained_variance[0] * 100:.1f}% variance")
    print(f"  PC2 explains {explained_variance[1] * 100:.1f}% variance")

    # ========================================
    # STEP 3: CREATE PCA BIPLOT WITH FEATURE VECTORS
    # (Answers: "show the three variables as vectors")
    # ========================================
    print("\n" + "=" * 70)
    print("1. CREATING PCA BIPLOT WITH FEATURE VECTORS")
    print("=" * 70)

    loadings_df = visualize_pca_biplot(
        X_pca=X_pca,
        true_labels=true_labels,
        species_names=species_names,
        pca=pca,
        feature_names=feature_names,
        explained_variance=explained_variance,
        save_path=os.path.join(output_dir, "1_pca_biplot_with_feature_vectors.png")
    )

    # ========================================
    # STEP 4: IDENTIFY OVERLAP ZONE BIRDS
    # (Answers: "pull out the images of birds in overlap zones")
    # ========================================
    print("\n" + "=" * 70)
    print("2. IDENTIFYING BIRDS IN OVERLAP ZONES")
    print("=" * 70)

    overlap_df, overlap_birds = identify_overlap_zone_birds(
        X_pca=X_pca,
        df=df,
        true_labels=true_labels,
        species_names=species_names,
        threshold_percentile=15,  # Adjust this: lower = fewer birds, higher = more birds
        save_path=os.path.join(output_dir, "2_overlap_zone_birds.png")
    )

    print("\n📋 OVERLAP ZONE BIRDS - Check the CSV file for filenames!")
    print(f"   File: {os.path.join(output_dir, '2_overlap_zone_birds.csv')}")

    # ========================================
    # STEP 5: HIGHLIGHT HOKKAIDO BIRDS
    # (Answers: "highlight these birds on the PCA")
    # ========================================
    print("\n" + "=" * 70)
    print("3. HIGHLIGHTING HOKKAIDO BIRDS")
    print("=" * 70)

    # ← CHANGE THESE to your actual Hokkaido bird filenames!
    hokkaido_filenames = [
        "mod-0R8A4927.JPG",
        # "mod-0R8A4952",
        # "mod-0R8A5091",
        "mod-0R8A5109.JPG",
        "mod-0R8A5217.JPG",
        # Add more as needed
    ]

    # ← CHANGE THIS to match your filename column name
    filename_column = 'image_name'  # Could be 'image_name', 'file', 'image_id', etc.

    hokkaido_results = locate_hokkaido_birds(
        X_pca=X_pca,
        df=df,
        true_labels=true_labels,
        species_names=species_names,
        explained_variance=explained_variance,
        hokkaido_filenames=hokkaido_filenames,
        filename_column=filename_column,
        save_path=os.path.join(output_dir, "3_hokkaido_birds_highlighted.png")
    )

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📁 All results saved to: {output_dir}")
    print("\n📊 Files created:")
    print("   1. 1_pca_biplot_with_feature_vectors.png - PCA with feature direction arrows")
    print("   2. 2_overlap_zone_birds.png - Visual of overlap zone")
    print("   3. 2_overlap_zone_birds.csv - List of overlap birds with filenames")
    print("   4. 3_hokkaido_birds_highlighted.png - Hokkaido birds highlighted")
    print("   5. 3_hokkaido_birds_highlighted_details.csv - Hokkaido bird details")
    print("\n💡 Next steps:")
    print("   - Use the CSV files to locate the actual image files")
    print("   - Combine the plots into your paper's Figure")
    print("=" * 70)