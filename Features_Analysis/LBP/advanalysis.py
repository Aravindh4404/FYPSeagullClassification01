import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy import stats
import seaborn as sns

# Path to the saved LBP features
FEATURES_PATH = "../Outputs/LBP_Features/lbp_features.pkl"
OUTPUT_DIR = "../Outputs/LBP_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_features():
    """Load the previously saved LBP features"""
    with open(FEATURES_PATH, "rb") as f:
        return pickle.load(f)


def create_feature_dataframe(features):
    """
    Convert the features dictionary to a pandas DataFrame for easier analysis
    """
    rows = []
    for species_name, regions in features['individual_features'].items():
        for region_name, samples in regions.items():
            for sample_id, feature_values in samples.items():
                # Extract all numeric features (excluding histogram)
                row = {
                    'species': species_name,
                    'region': region_name,
                    'sample_id': sample_id,
                    'mean': feature_values['mean'],
                    'std': feature_values['std'],
                    'energy': feature_values['energy'],
                    'entropy': feature_values['entropy'],
                    'contrast': feature_values['contrast'],
                    'homogeneity': feature_values['homogeneity']
                }
                rows.append(row)

    return pd.DataFrame(rows)


def analyze_features(df):
    """
    Perform statistical analysis and comparison between species
    """
    # Get unique regions
    regions = df['region'].unique()
    species = df['species'].unique()
    if len(species) != 2:
        print(f"Warning: Found {len(species)} species instead of 2")

    print("=" * 80)
    print("TEXTURE ANALYSIS RESULTS")
    print("=" * 80)

    # Compare each feature for each region between species
    results = {}
    for region in regions:
        print(f"\nRegion: {region}")
        print("-" * 40)

        region_results = {}
        for feature in ['mean', 'std', 'energy', 'entropy', 'contrast', 'homogeneity']:
            # Get data for each species for this feature and region
            data_by_species = {}
            for s in species:
                data_by_species[s] = df[(df['species'] == s) & (df['region'] == region)][feature].values

            # Only perform t-test if we have data for both species
            if all(len(d) > 0 for d in data_by_species.values()):
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(
                    data_by_species[species[0]],
                    data_by_species[species[1]],
                    equal_var=False  # Welch's t-test (doesn't assume equal variance)
                )

                # Calculate means for each species
                means = {s: np.mean(data_by_species[s]) for s in species}

                # Print results
                result_str = f"{feature}: {species[0]}={means[species[0]]:.4f}, {species[1]}={means[species[1]]:.4f}, "
                result_str += f"diff={means[species[0]] - means[species[1]]:.4f}, p-value={p_value:.4f}"
                if p_value < 0.05:
                    result_str += " *"
                    if p_value < 0.01:
                        result_str += "*"
                    if p_value < 0.001:
                        result_str += "*"

                print(result_str)

                # Store results
                region_results[feature] = {
                    'means': means,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

        results[region] = region_results

    return results


def create_visualizations(df, analysis_results):
    """
    Create visualizations to compare texture features between species
    """
    species = df['species'].unique()
    regions = df['region'].unique()
    features = ['mean', 'std', 'energy', 'entropy', 'contrast', 'homogeneity']

    # 1. Box plots for each feature by region and species
    for feature in features:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='region', y=feature, hue='species', data=df)
        plt.title(f'Comparison of {feature.capitalize()} by Region and Species')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'boxplot_{feature}.png'))
        plt.close()

    # 2. Heatmap of p-values
    p_values = np.zeros((len(regions), len(features)))
    for i, region in enumerate(regions):
        for j, feature in enumerate(features):
            if region in analysis_results and feature in analysis_results[region]:
                p_values[i, j] = analysis_results[region][feature]['p_value']
            else:
                p_values[i, j] = 1.0  # Not significant if missing

    plt.figure(figsize=(10, 8))
    sns.heatmap(p_values, annot=True, cmap='coolwarm_r',
                xticklabels=features, yticklabels=regions,
                vmin=0, vmax=0.1)
    plt.title('P-values for Texture Feature Comparisons Between Species')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pvalue_heatmap.png'))
    plt.close()

    # 3. Feature differences bar chart
    for region in regions:
        if region not in analysis_results:
            continue

        differences = []
        feature_names = []
        p_values = []

        for feature in features:
            if feature in analysis_results[region]:
                result = analysis_results[region][feature]
                means = result['means']
                diff = means[species[0]] - means[species[1]]
                differences.append(diff)
                feature_names.append(feature)
                p_values.append(result['p_value'])

        # Only create plot if we have data
        if differences:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(feature_names, differences)

            # Highlight significant differences
            for i, (bar, p) in enumerate(zip(bars, p_values)):
                if p < 0.05:
                    bar.set_color('red')

            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f'Feature Differences ({species[0]} - {species[1]}) for {region}')
            plt.ylabel('Difference')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'differences_{region}.png'))
            plt.close()

    # 4. Compare mean histograms between species
    features_data = load_features()
    for region in regions:
        plt.figure(figsize=(12, 6))

        for s in species:
            if s in features_data['aggregated_stats'] and region in features_data['aggregated_stats'][s]:
                hist = features_data['aggregated_stats'][s][region]['mean_histogram']
                plt.plot(hist, label=s)

        plt.title(f'Mean LBP Histogram Comparison for {region}')
        plt.xlabel('LBP Code')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'mean_histogram_{region}.png'))
        plt.close()


def main():
    print("Starting texture analysis...")

    # Load features
    features = load_features()

    # Convert to DataFrame
    df = create_feature_dataframe(features)

    # Perform analysis
    analysis_results = analyze_features(df)

    # Create visualizations
    create_visualizations(df, analysis_results)

    # Save analysis results
    with open(os.path.join(OUTPUT_DIR, 'analysis_results.pkl'), 'wb') as f:
        pickle.dump(analysis_results, f)

    # Export to CSV for further analysis
    df.to_csv(os.path.join(OUTPUT_DIR, 'texture_features.csv'), index=False)

    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")
    print("Key files:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'texture_features.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'analysis_results.pkl')}")
    print("  - Various visualization files (.png)")


if __name__ == "__main__":
    main()