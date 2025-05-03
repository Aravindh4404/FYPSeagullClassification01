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

    features = df[['mean_wing_intensity', 'mean_wingtip_intensity', 'pct_dark_lt_60']]

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


# Function to visualize misclassified points
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
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45)

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
def run_clustering_analysis(file_path, output_dir=None):
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    df, features, true_labels, species_names = load_data(file_path)
    feature_names = features.columns.tolist()

    # Standardize features
    X_scaled, scaler = preprocess_data(features)

    # Apply PCA for visualization
    X_pca, pca, explained_variance = apply_pca(X_scaled)
    print(f"PCA explained variance: {explained_variance[0]:.3f}, {explained_variance[1]:.3f}")

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

    return {
        'kmeans': (kmeans_clusters, kmeans_results, kmeans_analysis if true_labels is not None else None),
        'hierarchical': (
            hierarchical_clusters, hierarchical_results, hierarchical_analysis if true_labels is not None else None),
        'gmm': (gmm_clusters, gmm_results, gmm_analysis if true_labels is not None else None)
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
    # Replace with your actual file path
    file_path = os.path.join(os.path.dirname(__file__), "wingtip_intensity_distribution.csv")

    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(__file__), "clustering_results")
    os.makedirs(output_dir, exist_ok=True)

    results = run_clustering_analysis(file_path, output_dir)

    # Compare algorithms based on silhouette scores
    print("\n=== Algorithm Comparison ===")
    algorithms = []
    scores = []

    for algo, (_, result, _) in results.items():
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
    for algo, (_, _, analysis) in results.items():
        if analysis is not None:
            export_misclassified_points(analysis, algo, os.path.join(output_dir, f"{algo}_misclassified_points.csv"))

    sys.stdout = sys.__stdout__
    log_file.close()
