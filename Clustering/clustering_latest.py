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
log_file = open(os.path.join(output_dir, "console_output.txt"), "w")
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


# NEW: Function to create PCA biplot with feature vectors
def visualize_pca_biplot(X_pca, true_labels, species_names, pca, feature_names, explained_variance, save_path=None):
    """
    Create a PCA biplot showing both data points and feature loading vectors
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


# NEW: Function to identify birds in overlap zones
def identify_overlap_zone_birds(X_pca, df, true_labels, species_names, threshold_percentile=15, save_path=None):
    """
    Identify birds in the overlap zone between species clusters

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
        print(f"\nOverlap zone birds saved to: {csv_path}")
        print(f"Number of birds in overlap zone: {len(overlap_birds)} ({len(overlap_birds) / len(df) * 100:.1f}%)")

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


# NEW: Function to highlight specific population (e.g., Hokkaido SB)
def highlight_population_on_pca(X_pca, df, true_labels, species_names, explained_variance,
                                population_column, population_value, save_path=None):
    """
    Highlight a specific population on the PCA plot

    Parameters:
    - population_column: column name in df that identifies populations (e.g., 'location', 'population')
    - population_value: value to highlight (e.g., 'Hokkaido')
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Check if population column exists
    if population_column not in df.columns:
        print(f"Warning: Column '{population_column}' not found in dataframe.")
        print(f"Available columns: {df.columns.tolist()}")
        # Plot without population highlighting
        for label_id, species_name in species_names.items():
            mask = true_labels == label_id
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       label=species_name, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    else:
        # Create population mask
        population_mask = df[population_column] == population_value

        # Plot regular birds
        for label_id, species_name in species_names.items():
            mask = (true_labels == label_id) & (~population_mask)
            if mask.sum() > 0:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           label=f'{species_name} (other)', alpha=0.4, s=60,
                           edgecolors='gray', linewidth=0.5)

        # Highlight population birds
        for label_id, species_name in species_names.items():
            mask = (true_labels == label_id) & population_mask
            if mask.sum() > 0:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           label=f'{species_name} ({population_value})', alpha=0.95, s=150,
                           marker='*', edgecolors='black', linewidth=1.5)

        # Print summary
        print(f"\n=== {population_value} Population Highlight ===")
        print(f"Total {population_value} birds: {population_mask.sum()}")
        for label_id, species_name in species_names.items():
            mask = (true_labels == label_id) & population_mask
            print(f"{species_name} from {population_value}: {mask.sum()}")

    ax.set_xlabel(f'PC1 ({explained_variance[0] * 100:.1f}% variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained_variance[1] * 100:.1f}% variance)', fontsize=12, fontweight='bold')
    ax.set_title(f'PCA with {population_value} Population Highlighted', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Function to evaluate clustering performance
def evaluate_clustering(X_scaled, clusters, true_labels=None):
    silhouette = silhouette_score(X_scaled, clusters)
    results = {"silhouette_score": silhouette}

    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, clusters)
        results["adjusted_rand_index"] = ari

    return results


# Function to visualize clusters
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


# Function to visualize PCA with true species labels
def visualize_pca_by_species(X_pca, true_labels, species_names, explained_variance, save_path=None):
    plt.figure(figsize=(10, 6))

    # Plot each species with different colors
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


# [Rest of your existing functions remain the same - I'm keeping them to maintain compatibility]
# Function to create pairwise feature scatter plots (combined - original)
def visualize_feature_pairs(df, features, true_labels, species_names, save_path=None):
    feature_cols = features.columns.tolist()
    n_features = len(feature_cols)

    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))

    for i, feat1 in enumerate(feature_cols):
        for j, feat2 in enumerate(feature_cols):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                for label_id, species_name in species_names.items():
                    mask = true_labels == label_id
                    ax.hist(df.loc[mask, feat1], alpha=0.5, label=species_name, bins=20)
                ax.set_ylabel('Frequency')
                if i == 0:
                    ax.legend()
            else:
                # Off-diagonal: scatter plot
                for label_id, species_name in species_names.items():
                    mask = true_labels == label_id
                    ax.scatter(df.loc[mask, feat2], df.loc[mask, feat1],
                               alpha=0.5, label=species_name, s=20)

            # Labels
            if i == n_features - 1:
                ax.set_xlabel(feat2, fontsize=10)
            if j == 0:
                ax.set_ylabel(feat1, fontsize=10)

            # Remove tick labels for cleaner look
            if i < n_features - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    plt.suptitle('Pairwise Feature Distributions by Species', fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# Function to create individual pairwise scatter plots
def visualize_individual_feature_pairs(df, features, true_labels, species_names, output_dir=None):
    feature_cols = features.columns.tolist()
    n_features = len(feature_cols)

    # Create all unique pairs
    for i in range(n_features):
        for j in range(i, n_features):
            feat1 = feature_cols[i]
            feat2 = feature_cols[j]

            plt.figure(figsize=(10, 8))

            if i == j:
                # Diagonal: histogram for single feature
                for label_id, species_name in species_names.items():
                    mask = true_labels == label_id
                    plt.hist(df.loc[mask, feat1], alpha=0.6, label=species_name, bins=25, edgecolor='black')

                plt.xlabel(feat1, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.title(f'Distribution of {feat1} by Species', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)

            else:
                # Off-diagonal: scatter plot for feature pairs
                for label_id, species_name in species_names.items():
                    mask = true_labels == label_id
                    plt.scatter(df.loc[mask, feat2], df.loc[mask, feat1],
                                alpha=0.6, label=species_name, s=50, edgecolor='black', linewidth=0.5)

                plt.xlabel(feat2, fontsize=12)
                plt.ylabel(feat1, fontsize=12)
                plt.title(f'{feat1} vs {feat2} by Species', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save individual plot
            if output_dir:
                if i == j:
                    filename = f"feature_dist_{feat1}.png"
                else:
                    filename = f"feature_pair_{feat1}_vs_{feat2}.png"
                save_path = os.path.join(output_dir, filename)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    if output_dir:
        print(f"\nIndividual pairwise plots saved to: {output_dir}")


# Function to visualize feature distributions
def visualize_feature_distributions(df, features, true_labels, species_names, save_path=None):
    feature_cols = features.columns.tolist()
    n_features = len(feature_cols)

    fig, axes = plt.subplots(1, n_features, figsize=(15, 5))

    for i, feat in enumerate(feature_cols):
        ax = axes[i]

        for label_id, species_name in species_names.items():
            mask = true_labels == label_id
            ax.hist(df.loc[mask, feat], alpha=0.6, label=species_name, bins=20)

        ax.set_xlabel(feat)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feat}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# Function to create box plots for features
def visualize_feature_boxplots(df, features, true_labels, species_names, save_path=None):
    feature_cols = features.columns.tolist()
    n_features = len(feature_cols)

    fig, axes = plt.subplots(1, n_features, figsize=(15, 5))

    # Create dataframe for plotting
    plot_data = df[feature_cols].copy()
    plot_data['species'] = [species_names[label] for label in true_labels]

    for i, feat in enumerate(feature_cols):
        ax = axes[i]

        # Create box plot
        species_list = [species_names[label] for label in sorted(species_names.keys())]
        data_to_plot = [plot_data[plot_data['species'] == species][feat] for species in species_list]

        box_plot = ax.boxplot(data_to_plot, tick_labels=species_list, patch_artist=True)

        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel(feat)
        ax.set_title(f'{feat} by Species')
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x labels if needed
        ax.set_xticklabels(species_list, rotation=15, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# Function to map clusters to majority species and identify misclassifications
def map_clusters_to_species(df, clusters, true_labels, species_names):
    # Create a DataFrame for analysis
    cluster_analysis = pd.DataFrame({
        'Cluster': clusters,
        'True_Label': true_labels
    })

    # Determine majority species in each cluster
    cluster_counts = pd.crosstab(cluster_analysis['Cluster'], cluster_analysis['True_Label'])

    # Map numeric labels back to species names for better readability
    cluster_counts.columns = [species_names[col] for col in cluster_counts.columns]

    # Determine the majority class for each cluster
    majority_mapping = {}
    for cluster in cluster_counts.index:
        majority_class_id = cluster_counts.loc[cluster].idxmax()
        majority_mapping[cluster] = majority_class_id

    # Add predicted class to the analysis dataframe
    cluster_analysis['Predicted_Species'] = cluster_analysis['Cluster'].map(majority_mapping)

    # Identify misclassified points
    cluster_analysis['Correctly_Clustered'] = cluster_analysis['Predicted_Species'] == [species_names[label] for label
                                                                                        in
                                                                                        cluster_analysis['True_Label']]

    # Add original data features
    for col in df.columns:
        if col not in cluster_analysis.columns:
            cluster_analysis[col] = df[col].values

    return cluster_analysis, majority_mapping


# Function to visualize confusion matrix with mapped species labels
def visualize_confusion_matrix_with_mapping(true_labels, clusters, majority_mapping, species_names, save_path=None):
    if true_labels is None:
        print("No true labels available for confusion matrix.")
        return

    # Create predicted labels by mapping clusters to their majority species
    predicted_labels = [majority_mapping[cluster] for cluster in clusters]

    # Convert numeric labels to species names for display
    true_species = [species_names[label] for label in true_labels]
    predicted_species = predicted_labels  # already species names

    # Create a DataFrame for counting
    confusion_df = pd.DataFrame({'True': true_species, 'Predicted': predicted_species})

    # Create confusion matrix
    cm = pd.crosstab(confusion_df['True'], confusion_df['Predicted'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Species')
    plt.ylabel('True Species')
    plt.title('Confusion Matrix with Species Mapping')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    # Calculate and display accuracy
    accuracy = (confusion_df['True'] == confusion_df['Predicted']).mean() * 100
    print(f"Clustering accuracy after mapping: {accuracy:.2f}%")

    return confusion_df


# Function to visualize misclassifications
def visualize_misclassifications(X_pca, cluster_analysis, title="Misclassification Analysis", save_path=None):
    plt.figure(figsize=(10, 6))

    # Plot correctly classified points
    correct_mask = cluster_analysis['Correctly_Clustered']
    plt.scatter(X_pca[correct_mask, 0], X_pca[correct_mask, 1],
                c=cluster_analysis.loc[correct_mask, 'Cluster'],
                marker='o', alpha=0.6, label='Correctly Clustered')

    # Plot misclassified points with X markers
    incorrect_mask = ~correct_mask
    if sum(incorrect_mask) > 0:  # Check if there are any misclassified points
        plt.scatter(X_pca[incorrect_mask, 0], X_pca[incorrect_mask, 1],
                    c=cluster_analysis.loc[incorrect_mask, 'Cluster'],
                    marker='X', s=100, edgecolor='black', alpha=0.9, label='Misclassified')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    # Print summary of misclassifications
    misclassified = cluster_analysis[~cluster_analysis['Correctly_Clustered']]
    if len(misclassified) > 0:
        print(
            f"Number of misclassified points: {len(misclassified)} ({len(misclassified) / len(cluster_analysis) * 100:.2f}%)")
        print("\nSample of misclassified points:")
        display_cols = ['True_Label', 'Predicted_Species', 'Correctly_Clustered'] + list(misclassified.columns[-3:])
        print(misclassified[display_cols].head())
    else:
        print("No misclassified points found!")


# Function to create feature importance plot based on cluster centers
def visualize_feature_importance(scaler, cluster_centers, feature_names, save_path=None):
    # Transform cluster centers back to original scale
    original_centers = scaler.inverse_transform(cluster_centers)

    # Calculate differences between cluster centers for each feature
    center_diffs = np.abs(original_centers[0] - original_centers[1])

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(np.arange(len(feature_names)), center_diffs)
    plt.xlabel('Features')
    plt.ylabel('Absolute Difference Between Cluster Centers')
    plt.title('Feature Importance Based on Cluster Separation')
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45, ha='right')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# Function to calculate feature importance (returns dictionary)
def calculate_feature_importance(scaler, cluster_centers, feature_names):
    # Transform cluster centers back to original scale
    original_centers = scaler.inverse_transform(cluster_centers)

    # Calculate differences between cluster centers for each feature
    center_diffs = np.abs(original_centers[0] - original_centers[1])

    # Create dictionary of feature importance
    importance_dict = {}
    for feature, diff in zip(feature_names, center_diffs):
        importance_dict[feature] = diff

    return importance_dict


# Function to calculate pseudo-centroids for hierarchical clustering
def calculate_hierarchical_centers(X_scaled, clusters):
    # Calculate mean of each cluster
    unique_clusters = np.unique(clusters)
    centers = np.zeros((len(unique_clusters), X_scaled.shape[1]))

    for i, cluster in enumerate(unique_clusters):
        cluster_mask = clusters == cluster
        centers[i] = X_scaled[cluster_mask].mean(axis=0)

    return centers


# Function to create feature importance comparison table
def create_feature_importance_table(kmeans_imp, hierarchical_imp, gmm_imp, feature_names):
    # Create a dataframe with all feature importance values
    data = {
        'Feature': feature_names,
        'K-means_Importance': [kmeans_imp[feat] for feat in feature_names],
        'Hierarchical_Importance': [hierarchical_imp[feat] for feat in feature_names]
    }

    if gmm_imp is not None:
        data['GMM_Importance'] = [gmm_imp[feat] for feat in feature_names]

    df = pd.DataFrame(data)

    # Calculate average importance across all models
    importance_cols = [col for col in df.columns if 'Importance' in col]
    df['Average_Importance'] = df[importance_cols].mean(axis=1)

    # Calculate rank for each model
    for col in importance_cols:
        df[f'{col.replace("_Importance", "")}_Rank'] = df[col].rank(ascending=False, method='min').astype(int)

    # Sort by average importance
    df = df.sort_values('Average_Importance', ascending=False).reset_index(drop=True)

    return df


# 1. K-means Clustering
def apply_kmeans(X_scaled, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans.cluster_centers_


# 2. Hierarchical Clustering
def apply_hierarchical(X_scaled, n_clusters=2, linkage='ward'):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clusters = hierarchical.fit_predict(X_scaled)
    # Note: Hierarchical clustering doesn't provide cluster centers directly
    return clusters, None


# 3. Gaussian Mixture Model
def apply_gmm(X_scaled, n_components=2):
    gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE)
    clusters = gmm.fit_predict(X_scaled)
    return clusters, gmm.means_


# Main function to run all clustering algorithms
def run_clustering_analysis(file_path, output_dir=None, population_column=None, population_value=None):
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    df, features, true_labels, species_names = load_data(file_path)
    feature_names = features.columns.tolist()

    print("=== Data Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Features used: {feature_names}")
    if true_labels is not None:
        print(f"\nSpecies distribution:")
        for label_id, species_name in species_names.items():
            count = (true_labels == label_id).sum()
            print(f"  {species_name}: {count} samples ({count / len(df) * 100:.1f}%)")

    # Print feature statistics
    print(f"\n=== Feature Statistics ===")
    print(features.describe())

    # Standardize features
    X_scaled, scaler = preprocess_data(features)

    # Apply PCA for visualization
    X_pca, pca, explained_variance = apply_pca(X_scaled)
    print(f"\n=== PCA Analysis ===")
    print(f"PCA Component 1 explained variance: {explained_variance[0]:.3f} ({explained_variance[0] * 100:.1f}%)")
    print(f"PCA Component 2 explained variance: {explained_variance[1]:.3f} ({explained_variance[1] * 100:.1f}%)")
    print(
        f"Total variance explained by 2 components: {sum(explained_variance):.3f} ({sum(explained_variance) * 100:.1f}%)")

    # NEW: Create PCA biplot with feature vectors
    if true_labels is not None and species_names is not None:
        biplot_path = os.path.join(output_dir, "pca_biplot_with_vectors.png") if output_dir else None
        loadings_df = visualize_pca_biplot(X_pca, true_labels, species_names, pca, feature_names,
                                           explained_variance, biplot_path)

        # Save loadings to CSV
        if output_dir:
            loadings_csv = os.path.join(output_dir, "pca_component_loadings.csv")
            loadings_df.to_csv(loadings_csv)
            print(f"PCA loadings saved to: {loadings_csv}")

    # Visualize PCA by true species labels (original version)
    if true_labels is not None and species_names is not None:
        pca_species_path = os.path.join(output_dir, "pca_by_species.png") if output_dir else None
        visualize_pca_by_species(X_pca, true_labels, species_names, explained_variance, pca_species_path)

        # NEW: Identify and visualize overlap zone birds
        overlap_path = os.path.join(output_dir, "overlap_zone_birds.png") if output_dir else None
        overlap_df, overlap_birds = identify_overlap_zone_birds(X_pca, df, true_labels, species_names,
                                                                threshold_percentile=15, save_path=overlap_path)

        # NEW: Highlight specific population if specified
        if population_column and population_value:
            population_path = os.path.join(output_dir, f"pca_highlight_{population_value}.png") if output_dir else None
            highlight_population_on_pca(X_pca, df, true_labels, species_names, explained_variance,
                                        population_column, population_value, population_path)

        # Visualize feature distributions
        feature_dist_path = os.path.join(output_dir, "feature_distributions.png") if output_dir else None
        visualize_feature_distributions(df, features, true_labels, species_names, feature_dist_path)

        # Visualize feature box plots
        feature_box_path = os.path.join(output_dir, "feature_boxplots.png") if output_dir else None
        visualize_feature_boxplots(df, features, true_labels, species_names, feature_box_path)

        # Visualize pairwise feature relationships
        feature_pairs_path = os.path.join(output_dir, "feature_pairwise.png") if output_dir else None
        visualize_feature_pairs(df, features, true_labels, species_names, feature_pairs_path)

        # Visualize individual pairwise feature relationships
        visualize_individual_feature_pairs(df, features, true_labels, species_names, output_dir)

    # 1. K-means Clustering
    print("\n=== K-means Clustering ===")
    kmeans_clusters, kmeans_centers = apply_kmeans(X_scaled)
    kmeans_results = evaluate_clustering(X_scaled, kmeans_clusters, true_labels)
    print(f"Silhouette Score: {kmeans_results['silhouette_score']:.3f}")
    if 'adjusted_rand_index' in kmeans_results:
        print(f"Adjusted Rand Index: {kmeans_results['adjusted_rand_index']:.3f}")

    # Visualize K-means results
    kmeans_centers_pca = pca.transform(kmeans_centers) if kmeans_centers is not None else None
    kmeans_plot_path = os.path.join(output_dir, "kmeans_clustering.png") if output_dir else None
    visualize_clusters(X_pca, kmeans_clusters, kmeans_centers_pca, "K-means Clustering Results", kmeans_plot_path)

    if true_labels is not None and species_names is not None:
        # Map clusters to species and identify misclassifications
        kmeans_analysis, kmeans_mapping = map_clusters_to_species(df, kmeans_clusters, true_labels, species_names)

        # Visualize confusion matrix with species mapping
        print("\n=== K-means Species Mapping Analysis ===")
        kmeans_confusion_path = os.path.join(output_dir, "kmeans_confusion_matrix.png") if output_dir else None
        kmeans_confusion = visualize_confusion_matrix_with_mapping(true_labels, kmeans_clusters, kmeans_mapping,
                                                                   species_names, kmeans_confusion_path)

        # Visualize misclassified points
        kmeans_misclass_path = os.path.join(output_dir, "kmeans_misclassifications.png") if output_dir else None
        visualize_misclassifications(X_pca, kmeans_analysis, "K-means Misclassification Analysis", kmeans_misclass_path)

        # Print majority mappings
        print("\nK-means Cluster to Species Mapping:")
        for cluster, species in kmeans_mapping.items():
            print(f"Cluster {cluster} -> {species}")

    # Feature importance based on K-means
    if kmeans_centers is not None:
        kmeans_feature_path = os.path.join(output_dir, "kmeans_feature_importance.png") if output_dir else None
        visualize_feature_importance(scaler, kmeans_centers, feature_names, kmeans_feature_path)

        # Calculate feature importance for K-means
        kmeans_importance = calculate_feature_importance(scaler, kmeans_centers, feature_names)

    # 2. Hierarchical Clustering
    print("\n=== Hierarchical Clustering ===")
    hierarchical_clusters, _ = apply_hierarchical(X_scaled)
    hierarchical_results = evaluate_clustering(X_scaled, hierarchical_clusters, true_labels)
    print(f"Silhouette Score: {hierarchical_results['silhouette_score']:.3f}")
    if 'adjusted_rand_index' in hierarchical_results:
        print(f"Adjusted Rand Index: {hierarchical_results['adjusted_rand_index']:.3f}")

    # Visualize Hierarchical results
    hierarchical_plot_path = os.path.join(output_dir, "hierarchical_clustering.png") if output_dir else None
    visualize_clusters(X_pca, hierarchical_clusters, None, "Hierarchical Clustering Results", hierarchical_plot_path)

    # Calculate pseudo-centroids for hierarchical clustering
    hierarchical_centers = calculate_hierarchical_centers(X_scaled, hierarchical_clusters)
    hierarchical_importance = calculate_feature_importance(scaler, hierarchical_centers, feature_names)

    if true_labels is not None and species_names is not None:
        # Map clusters to species and identify misclassifications
        hierarchical_analysis, hierarchical_mapping = map_clusters_to_species(df, hierarchical_clusters, true_labels,
                                                                              species_names)

        # Visualize confusion matrix with species mapping
        print("\n=== Hierarchical Clustering Species Mapping Analysis ===")
        hierarchical_confusion_path = os.path.join(output_dir,
                                                   "hierarchical_confusion_matrix.png") if output_dir else None
        hierarchical_confusion = visualize_confusion_matrix_with_mapping(true_labels, hierarchical_clusters,
                                                                         hierarchical_mapping, species_names,
                                                                         hierarchical_confusion_path)

        # Visualize misclassified points
        hierarchical_misclass_path = os.path.join(output_dir,
                                                  "hierarchical_misclassifications.png") if output_dir else None
        visualize_misclassifications(X_pca, hierarchical_analysis, "Hierarchical Clustering Misclassification Analysis",
                                     hierarchical_misclass_path)

        # Print majority mappings
        print("\nHierarchical Cluster to Species Mapping:")
        for cluster, species in hierarchical_mapping.items():
            print(f"Cluster {cluster} -> {species}")

    # 3. Gaussian Mixture Model
    print("\n=== Gaussian Mixture Model ===")
    gmm_clusters, gmm_centers = apply_gmm(X_scaled)
    gmm_results = evaluate_clustering(X_scaled, gmm_clusters, true_labels)
    print(f"Silhouette Score: {gmm_results['silhouette_score']:.3f}")
    if 'adjusted_rand_index' in gmm_results:
        print(f"Adjusted Rand Index: {gmm_results['adjusted_rand_index']:.3f}")

    # Visualize GMM results
    gmm_centers_pca = pca.transform(gmm_centers) if gmm_centers is not None else None
    gmm_plot_path = os.path.join(output_dir, "gmm_clustering.png") if output_dir else None
    visualize_clusters(X_pca, gmm_clusters, gmm_centers_pca, "Gaussian Mixture Model Results", gmm_plot_path)

    # Calculate feature importance for GMM
    if gmm_centers is not None:
        gmm_importance = calculate_feature_importance(scaler, gmm_centers, feature_names)

    if true_labels is not None and species_names is not None:
        # Map clusters to species and identify misclassifications
        gmm_analysis, gmm_mapping = map_clusters_to_species(df, gmm_clusters, true_labels, species_names)

        # Visualize confusion matrix with species mapping
        print("\n=== Gaussian Mixture Model Species Mapping Analysis ===")
        gmm_confusion_path = os.path.join(output_dir, "gmm_confusion_matrix.png") if output_dir else None
        gmm_confusion = visualize_confusion_matrix_with_mapping(true_labels, gmm_clusters, gmm_mapping, species_names,
                                                                gmm_confusion_path)

        # Visualize misclassified points
        gmm_misclass_path = os.path.join(output_dir, "gmm_misclassifications.png") if output_dir else None
        visualize_misclassifications(X_pca, gmm_analysis, "GMM Misclassification Analysis", gmm_misclass_path)

        # Print majority mappings
        print("\nGMM Cluster to Species Mapping:")
        for cluster, species in gmm_mapping.items():
            print(f"Cluster {cluster} -> {species}")

    # Create combined feature importance table
    print("\n=== Feature Importance Comparison Across All Models ===")
    feature_importance_table = create_feature_importance_table(
        kmeans_importance,
        hierarchical_importance,
        gmm_importance if gmm_centers is not None else None,
        feature_names
    )

    # Save feature importance table to CSV
    importance_csv_path = os.path.join(output_dir, "feature_importance_comparison.csv") if output_dir else None
    if importance_csv_path:
        feature_importance_table.to_csv(importance_csv_path, index=False)
        print(f"\nFeature importance table saved to: {importance_csv_path}")

    print("\n" + feature_importance_table.to_string(index=False))

    return {
        'kmeans': (kmeans_clusters, kmeans_results, kmeans_analysis if true_labels is not None else None),
        'hierarchical': (
            hierarchical_clusters, hierarchical_results, hierarchical_analysis if true_labels is not None else None),
        'gmm': (gmm_clusters, gmm_results, gmm_analysis if true_labels is not None else None),
        'overlap_birds': overlap_birds if true_labels is not None else None
    }


# Function to export misclassified points to CSV
def export_misclassified_points(analysis_df, algorithm_name, output_path=None):
    if analysis_df is None:
        print(f"No analysis data available for {algorithm_name}")
        return

    # Filter for misclassified points
    misclassified = analysis_df[~analysis_df['Correctly_Clustered']]

    if len(misclassified) == 0:
        print(f"No misclassified points found for {algorithm_name}")
        return

    # Create output filename if not provided
    if output_path is None:
        output_path = f"{algorithm_name}_misclassified_points.csv"

    # Export to CSV
    misclassified.to_csv(output_path, index=False)
    print(f"Exported {len(misclassified)} misclassified points to {output_path}")


# Example usage
if __name__ == "__main__":
    # Updated file path (use raw string to avoid escape sequence warnings)
    file_path = r"D:\FYPSeagullClassification01\Clustering\dark_pixel_results_all_images.csv"

    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(__file__), "clustering_results")
    os.makedirs(output_dir, exist_ok=True)

    # Run clustering analysis with optional population highlighting
    # If your CSV has a column for location/population (e.g., 'location', 'population', 'origin'),
    # specify it here to highlight Hokkaido birds
    results = run_clustering_analysis(
        file_path,
        output_dir,
        population_column='location',  # Change this to your actual column name
        population_value='Hokkaido'  # Change this to match your data
    )

    # Compare algorithms based on silhouette scores
    print("\n=== Algorithm Comparison ===")
    algorithms = []
    scores = []

    for algo, value in results.items():
        if algo != 'overlap_birds':
            _, result, _ = value
            if 'silhouette_score' in result:
                algorithms.append(algo)
                scores.append(result['silhouette_score'])

    # Visualize comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, scores)
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Algorithm Comparison')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    # Save the comparison plot
    comparison_plot_path = os.path.join(output_dir, "algorithm_comparison.png")
    plt.savefig(comparison_plot_path)
    plt.close()

    # Export misclassified points for each algorithm
    for algo, value in results.items():
        if algo != 'overlap_birds':
            _, _, analysis = value
            if analysis is not None:
                export_misclassified_points(analysis, algo,
                                            os.path.join(output_dir, f"{algo}_misclassified_points.csv"))

    sys.stdout = sys.__stdout__
    log_file.close()