import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load and prepare data
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)

    # Rename column if needed
    if 'method3_percentage' in df.columns:
        df = df.rename(columns={'method3_percentage': 'percentage_dark_pixels_wingtip'})

    # Extract features
    features = df[['mean_wing_intensity', 'mean_wingtip_intensity', 'percentage_dark_pixels_wingtip']]

    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Add PCA coordinates to dataframe
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    return df, X_pca, pca


# Method 1: Find birds by clicking on the plot (interactive)
def identify_birds_interactive(df, X_pca):
    """
    Interactive plot - click on points to identify birds
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create species mapping if exists
    if 'species' in df.columns:
        species_map = {'Slaty_Backed_Gull': 0, 'Glaucous_Winged_Gull': 1}
        colors = df['species'].map(species_map)
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='viridis',
                             alpha=0.6, s=60, edgecolors='black', linewidth=0.5, picker=True)
    else:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=60,
                             edgecolors='black', linewidth=0.5, picker=True)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Click on Points to Identify Birds')
    ax.grid(True, alpha=0.3)

    # Store clicked points
    selected_birds = []

    def on_pick(event):
        ind = event.ind[0]  # Get index of clicked point

        print("\n" + "=" * 60)
        print(f"BIRD #{ind} SELECTED")
        print("=" * 60)

        # Print all available information
        for col in df.columns:
            print(f"{col}: {df.iloc[ind][col]}")

        print(f"\nPCA Coordinates: PC1={df.iloc[ind]['PC1']:.3f}, PC2={df.iloc[ind]['PC2']:.3f}")
        print("=" * 60 + "\n")

        selected_birds.append(ind)

        # Highlight selected point
        ax.scatter(X_pca[ind, 0], X_pca[ind, 1], s=300, facecolors='none',
                   edgecolors='red', linewidth=3, marker='o')
        plt.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

    return selected_birds


# Method 2: Find birds within a specific region
def find_birds_in_region(df, pc1_min, pc1_max, pc2_min, pc2_max):
    """
    Find all birds within a specified rectangular region

    Parameters:
    - pc1_min, pc1_max: Range for PC1
    - pc2_min, pc2_max: Range for PC2
    """
    mask = (df['PC1'] >= pc1_min) & (df['PC1'] <= pc1_max) & \
           (df['PC2'] >= pc2_min) & (df['PC2'] <= pc2_max)

    birds_in_region = df[mask].copy()

    print(f"\n=== Found {len(birds_in_region)} birds in region ===")
    print(f"PC1 range: [{pc1_min}, {pc1_max}]")
    print(f"PC2 range: [{pc2_min}, {pc2_max}]")
    print("\nBirds:")
    print(birds_in_region.to_string())

    return birds_in_region


# Method 3: Find nearest bird to specific coordinates
def find_nearest_bird(df, target_pc1, target_pc2, n_nearest=5):
    """
    Find the n nearest birds to a specific point

    Parameters:
    - target_pc1, target_pc2: Target coordinates
    - n_nearest: Number of nearest birds to return
    """
    # Calculate distances
    distances = np.sqrt((df['PC1'] - target_pc1) ** 2 + (df['PC2'] - target_pc2) ** 2)
    df['distance_to_target'] = distances

    # Get nearest birds
    nearest = df.nsmallest(n_nearest, 'distance_to_target')

    print(f"\n=== {n_nearest} Nearest Birds to Point ({target_pc1:.2f}, {target_pc2:.2f}) ===")
    print(nearest.to_string())

    return nearest


# Method 4: Find birds you circled (by providing approximate coordinates)
def find_circled_birds(df, X_pca, circle_centers, radius=0.5):
    """
    Find birds within circles you've drawn

    Parameters:
    - circle_centers: List of (pc1, pc2) tuples representing circle centers
    - radius: Radius around each center to search
    """
    all_circled = []

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot all birds
    if 'species' in df.columns:
        for species in df['species'].unique():
            mask = df['species'] == species
            ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
                       label=species, alpha=0.5, s=60)
    else:
        ax.scatter(df['PC1'], df['PC2'], alpha=0.5, s=60)

    # For each circle
    for i, (center_pc1, center_pc2) in enumerate(circle_centers):
        # Find birds within radius
        distances = np.sqrt((df['PC1'] - center_pc1) ** 2 + (df['PC2'] - center_pc2) ** 2)
        mask = distances <= radius

        circled_birds = df[mask].copy()
        circled_birds['circle_id'] = i + 1
        circled_birds['distance_from_center'] = distances[mask]

        all_circled.append(circled_birds)

        # Highlight on plot
        ax.scatter(circled_birds['PC1'], circled_birds['PC2'],
                   s=200, facecolors='none', edgecolors='red', linewidth=2.5)

        # Draw circle
        circle = plt.Circle((center_pc1, center_pc2), radius,
                            fill=False, color='red', linestyle='--', linewidth=2)
        ax.add_patch(circle)

        # Label circle
        ax.text(center_pc1, center_pc2 + radius + 0.2, f'Circle {i + 1}',
                ha='center', fontsize=10, fontweight='bold', color='red')

        print(f"\n=== Birds in Circle {i + 1} (center: {center_pc1:.2f}, {center_pc2:.2f}) ===")
        print(f"Found {len(circled_birds)} birds")
        print(circled_birds.to_string())

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Circled Birds Highlighted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Combine all circled birds
    all_circled_df = pd.concat(all_circled, ignore_index=True)

    return all_circled_df


# Method 5: Export bird details with thumbnails/paths
def export_selected_birds(birds_df, output_path='selected_birds.csv'):
    """
    Export selected birds to CSV with all their information
    """
    birds_df.to_csv(output_path, index=False)
    print(f"\n=== Exported {len(birds_df)} birds to {output_path} ===")

    # Print image paths if available
    image_cols = [col for col in birds_df.columns if 'image' in col.lower() or 'filename' in col.lower()]
    if image_cols:
        print(f"\nImage paths (column: {image_cols[0]}):")
        for idx, row in birds_df.iterrows():
            print(f"  - {row[image_cols[0]]}")


# Main execution
if __name__ == "__main__":
    # Load your data
    file_path = r"D:\FYPSeagullClassification01\Clustering\dark_pixel_results_all_images.csv"
    df, X_pca, pca = load_and_process_data(file_path)

    print("=== Data Loaded Successfully ===")
    print(f"Total birds: {len(df)}")
    print(f"PC1 range: [{df['PC1'].min():.2f}, {df['PC1'].max():.2f}]")
    print(f"PC2 range: [{df['PC2'].min():.2f}, {df['PC2'].max():.2f}]")

    # ============================================================
    # CHOOSE YOUR METHOD:
    # ============================================================

    # METHOD 1: Interactive clicking (uncomment to use)
    # print("\n=== Starting Interactive Mode ===")
    # print("Click on birds in the plot to identify them. Close the plot when done.")
    # selected = identify_birds_interactive(df, X_pca)

    # METHOD 2: Find birds in a specific region (uncomment and adjust coordinates)
    # birds_in_region = find_birds_in_region(df, pc1_min=-1.0, pc1_max=1.0,
    #                                        pc2_min=-0.5, pc2_max=0.5)

    # METHOD 3: Find nearest bird to coordinates (uncomment and adjust coordinates)
    # nearest = find_nearest_bird(df, target_pc1=0.5, target_pc2=0.2, n_nearest=5)

    # METHOD 4: Find birds you circled (RECOMMENDED for your use case)
    # Specify the approximate center of each circle you drew
    print("\n=== Finding Circled Birds ===")
    print("Finding birds in your 4 circled overlap zones")

    # Coordinates based on your hand-drawn circles (adjusted):
    circle_centers = [
        (-1.0, 0.75),  # Circle 1 - Single blue bird top left
        (-0.9, 0.08),  # Circle 2 - Bottom center overlap (mostly blue with few orange)
        (-0.65, -0.7),  # Circle 3 - Bottom left cluster
        (0.3, -0.90),  # Circle 4 - Right side overlap zone (mix of blue/orange)
    ]

    # Smaller radius to be more selective
    circled_birds = find_circled_birds(df, X_pca, circle_centers, radius=0.05)

    # Export the circled birds
    export_selected_birds(circled_birds, 'overlap_zone_circled_birds.csv')

    # METHOD 5: Get specific bird by row number if you know it
    # print("\n=== Specific Bird Details ===")
    # bird_index = 42  # Change this to your bird's row number
    # print(df.iloc[bird_index].to_string())