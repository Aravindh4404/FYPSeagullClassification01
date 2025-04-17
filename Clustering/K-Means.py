from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import sys

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

# Load and prepare data
df = pd.read_csv("wingtip_intensity_distribution.csv")
features = df[['mean_wing_intensity', 'mean_wingtip_intensity', 'darker_pixel_count']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Initialize and fit K-means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Evaluation metrics
silhouette = silhouette_score(X_scaled, clusters)
ari = adjusted_rand_score(df['species'].map({'Slaty_Backed_Gull':0, 'Glaucous_Winged_Gull':1}), clusters)

print(f"Silhouette Score: {silhouette:.3f}")
print(f"Adjusted Rand Index: {ari:.3f}")

# Visualization using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster')
plt.title('K-means Clustering Results (PCA Projection)')

# Add centroids to the plot
centroids_pca = pca.transform(scaler.inverse_transform(kmeans.cluster_centers_))
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, marker='X', c='red', label='Centroids')
plt.legend()
plt.show()

# Cluster interpretation
cluster_profiles = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                               columns=features.columns)
print("\nCluster Characteristics:")
print(cluster_profiles)

# Confusion matrix to evaluate clustering accuracy
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Map species to 0 and 1 for comparison
true_labels = df['species'].map({'Slaty_Backed_Gull':0, 'Glaucous_Winged_Gull':1})

# Generate confusion matrix
cm = confusion_matrix(true_labels, clusters)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cluster 0', 'Cluster 1'],
            yticklabels=['Slaty_Backed_Gull', 'Glaucous_Winged_Gull'])
plt.xlabel('Predicted Cluster')
plt.ylabel('True Species')
plt.title('Clustering Confusion Matrix')
plt.show()
