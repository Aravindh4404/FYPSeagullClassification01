import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy import stats
import seaborn as sns
from tabulate import tabulate
from matplotlib.colors import LinearSegmentedColormap

# Path to the saved LBP features
FEATURES_PATH = "../Outputs/LBP_Features/lbp_features.pkl"
OUTPUT_DIR = "../Outputs/LBP_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)
HTML_REPORT_PATH = os.path.join(OUTPUT_DIR, "texture_analysis_report.html")

# Define feature descriptions for better understanding
FEATURE_DESCRIPTIONS = {
    'mean': 'Average LBP value - indicates overall brightness/pattern type',
    'std': 'Standard deviation of LBP values - measures pattern variability',
    'variance': 'Variance of LBP values - another measure of pattern spread',
    'median': 'Median LBP value - robust measure of central tendency',
    'energy': 'Energy/Uniformity - measures textural uniformity (higher = more uniform)',
    'entropy': 'Entropy - measures randomness/complexity (higher = more random)',
    'uniformity': 'Uniformity - measures how uniform the texture is',
    'contrast': 'Contrast - measures local intensity variation (higher = more contrast)',
    'homogeneity': 'Homogeneity - measures closeness of distribution (higher = more homogeneous)',
    'dissimilarity': 'Dissimilarity - opposite of homogeneity (higher = less similar patterns)',
    'smoothness': 'Smoothness - relative smoothness of the texture (higher = smoother)',
    'skewness': 'Skewness - asymmetry of the distribution (0 = symmetric)',
    'kurtosis': 'Kurtosis - peakedness of the distribution (higher = more peaked)'
}


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
                    'sample_id': sample_id
                }
                # Add all features except histogram
                for k, v in feature_values.items():
                    if k != 'histogram':
                        row[k] = v
                rows.append(row)

    return pd.DataFrame(rows)


def analyze_features(df):
    """
    Perform statistical analysis and comparison between species
    """
    # Get unique regions and features
    regions = df['region'].unique()
    species = df['species'].unique()
    feature_cols = [col for col in df.columns if col not in ['species', 'region', 'sample_id']]

    if len(species) != 2:
        print(f"Warning: Found {len(species)} species instead of 2")

    print("=" * 80)
    print("TEXTURE ANALYSIS RESULTS")
    print("=" * 80)

    # Prepare to store results
    results = {}
    all_effect_sizes = []  # To track feature importance across regions

    for region in regions:
        print(f"\nRegion: {region}")
        print("-" * 40)

        region_results = {}
        for feature in feature_cols:
            # Get data for each species for this feature and region
            data_by_species = {}
            for s in species:
                data = df[(df['species'] == s) & (df['region'] == region)][feature].values
                # Filter out NaN values
                data_by_species[s] = data[~np.isnan(data)]

            # Only perform t-test if we have enough data for both species
            if all(len(d) >= 2 for d in data_by_species.values()):
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(
                    data_by_species[species[0]],
                    data_by_species[species[1]],
                    equal_var=False  # Welch's t-test (doesn't assume equal variance)
                )

                # Calculate means and standard deviations
                means = {s: np.mean(data_by_species[s]) for s in species}
                stds = {s: np.std(data_by_species[s]) for s in species}

                # Calculate Cohen's d effect size
                pooled_std = np.sqrt((np.var(data_by_species[species[0]]) +
                                      np.var(data_by_species[species[1]])) / 2)
                effect_size = abs(means[species[0]] - means[species[1]]) / (pooled_std + 1e-10)

                # Track effect sizes for all features
                all_effect_sizes.append({
                    'region': region,
                    'feature': feature,
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })

                # Print results
                result_str = f"{feature}: {species[0]}={means[species[0]]:.4f}±{stds[species[0]]:.4f}, " + \
                             f"{species[1]}={means[species[1]]:.4f}±{stds[species[1]]:.4f}, " + \
                             f"diff={means[species[0]] - means[species[1]]:.4f}, " + \
                             f"effect={effect_size:.2f}, p={p_value:.4f}"
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
                    'stds': stds,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05
                }

        results[region] = region_results

    # Rank features by effect size
    effect_size_df = pd.DataFrame(all_effect_sizes)
    return results, effect_size_df


def create_visualizations(df, analysis_results, effect_size_df):
    """
    Create visualizations to compare texture features between species
    """
    species = df['species'].unique()
    regions = df['region'].unique()
    features = [col for col in df.columns if col not in ['species', 'region', 'sample_id']]

    # 1. Box plots for the most distinguishing features
    # Find top features by effect size (that are significant)
    significant_features = effect_size_df[effect_size_df['significant']].sort_values(
        by='effect_size', ascending=False
    )

    # Plot top 5 most distinguishing features across regions
    if len(significant_features) > 0:
        top_n = min(5, len(significant_features))
        top_features = significant_features.head(top_n)

        for _, row in top_features.iterrows():
            feature = row['feature']
            plt.figure(figsize=(12, 6))
            ax = sns.boxplot(x='region', y=feature, hue='species', data=df)
            plt.title(
                f'Most Distinguishing Feature: {feature.capitalize()}\nEffect Size: {row["effect_size"]:.2f}, p-value: {row["p_value"]:.4f}')
            if feature in FEATURE_DESCRIPTIONS:
                plt.figtext(0.5, 0.01, FEATURE_DESCRIPTIONS[feature], ha='center',
                            bbox={'facecolor': 'orange', 'alpha': 0.1, 'pad': 5})
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'top_feature_{feature}.png'))
            plt.close()

    # 2. Feature importance heatmap
    pivot_effect = effect_size_df.pivot(index='region', columns='feature', values='effect_size')
    pivot_p = effect_size_df.pivot(index='region', columns='feature', values='p_value')

    # Create a masked heatmap where only significant differences are colored
    mask = pivot_p > 0.05

    plt.figure(figsize=(16, 10))
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', '#FFFFCC', '#FFEDA0', '#FED976',
                                                             '#FEB24C', '#FD8D3C', '#FC4E2A',
                                                             '#E31A1C', '#BD0026', '#800026'])

    ax = sns.heatmap(pivot_effect, mask=mask, cmap=cmap,
                     linewidths=.5, annot=True, fmt='.2f',
                     vmin=0, vmax=max(2.0, pivot_effect.max().max()))
    plt.title('Feature Importance (Effect Size) by Region\n(Only Significant Differences Shown)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_heatmap.png'))
    plt.close()

    # 3. Compare mean histograms for regions with the most significant differences
    features_data = load_features()

    # Find regions with the most significant differences
    region_significance = effect_size_df.groupby('region')['effect_size'].sum().sort_values(ascending=False)

    for region in region_significance.index[:min(3, len(region_significance))]:
        plt.figure(figsize=(12, 6))

        for s in species:
            if s in features_data['aggregated_stats'] and region in features_data['aggregated_stats'][s]:
                hist = features_data['aggregated_stats'][s][region]['mean_histogram']
                plt.plot(hist, label=s)

        plt.title(f'Mean LBP Histogram: {region} (Highly Distinguishing Region)')
        plt.xlabel('LBP Code')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'mean_histogram_{region}.png'))
        plt.close()


def generate_comparison_report(analysis_results, effect_size_df):
    """
    Generate an extensive report on the texture feature differences
    """
    # Sort features by effect size
    top_features = effect_size_df[effect_size_df['significant']].sort_values(
        by='effect_size', ascending=False
    )

    # Create region-specific reports
    region_reports = {}
    for region in effect_size_df['region'].unique():
        region_data = effect_size_df[effect_size_df['region'] == region]
        region_significant = region_data[region_data['significant']].sort_values(
            by='effect_size', ascending=False
        )

        if not region_significant.empty:
            region_reports[region] = region_significant

    # Generate a comprehensive text report
    report = []
    report.append("# Texture Analysis Comparison Report")
    report.append("\n## Overview")
    report.append("\nThis analysis compares texture features between two bird species across different regions.")

    # Overall most distinguishing features
    report.append("\n## Most Distinguishing Features Overall")
    if not top_features.empty:
        report.append("\nThese features show the most significant differences between species:")
        for i, (_, row) in enumerate(top_features.head(5).iterrows(), 1):
            desc = FEATURE_DESCRIPTIONS.get(row['feature'], "")
            report.append(
                f"\n{i}. **{row['feature'].capitalize()}** (Effect Size: {row['effect_size']:.2f}, p-value: {row['p_value']:.4f})")
            report.append(f"   - {desc}")
    else:
        report.append("\nNo statistically significant differences were found between the species.")

    # Region-specific analyses
    report.append("\n## Region-Specific Analysis")
    for region, region_data in region_reports.items():
        report.append(f"\n### {region} Region")
        if not region_data.empty:
            report.append("\nSignificant differentiating features:")
            feature_rows = []
            for _, row in region_data.iterrows():
                feature_rows.append([
                    row['feature'],
                    f"{row['effect_size']:.2f}",
                    f"{row['p_value']:.4f}",
                    '*' * min(3, int(-np.log10(row['p_value']))) if row['p_value'] < 0.05 else ''
                ])

            report.append("\n```")
            report.append(tabulate(
                feature_rows,
                headers=["Feature", "Effect Size", "p-value", "Significance"],
                tablefmt="simple"
            ))
            report.append("```")

            # Get the top feature for this region
            if len(region_data) > 0:
                top_feature = region_data.iloc[0]['feature']
                report.append(f"\nThe most distinguishing feature for {region} is **{top_feature}**.")
                if top_feature in FEATURE_DESCRIPTIONS:
                    report.append(f"\n{FEATURE_DESCRIPTIONS[top_feature]}")
        else:
            report.append("\nNo statistically significant differences were found in this region.")

    # Conclusion
    report.append("\n## Conclusion")
    if not top_features.empty:
        top_regions = effect_size_df.groupby('region')['effect_size'].sum().sort_values(ascending=False)

        report.append("\nBased on the texture analysis:")

        # Top distinguishing features
        top_feature = top_features.iloc[0]['feature']
        report.append(f"\n1. **{top_feature}** is the most differentiating texture feature between the species")
        if top_feature in FEATURE_DESCRIPTIONS:
            report.append(f"   - {FEATURE_DESCRIPTIONS[top_feature]}")

        # Top differentiating regions
        if not top_regions.empty:
            top_region = top_regions.index[0]
            report.append(f"\n2. The **{top_region}** region shows the most significant texture differences")

            # Features in the top region
            region_features = effect_size_df[
                (effect_size_df['region'] == top_region) &
                (effect_size_df['significant'])
                ].sort_values(by='effect_size', ascending=False)

            if not region_features.empty:
                region_top_feature = region_features.iloc[0]['feature']
                report.append(f"   - Primarily in the **{region_top_feature}** feature")

        # Pattern differences observed
        report.append("\n3. Overall pattern differences:")
        textures = []
        if 'contrast' in top_features['feature'].values:
            contrast_effect = top_features[top_features['feature'] == 'contrast'].iloc[0]['effect_size']
            textures.append(f"contrast (effect size: {contrast_effect:.2f})")
        if 'uniformity' in top_features['feature'].values:
            uniformity_effect = top_features[top_features['feature'] == 'uniformity'].iloc[0]['effect_size']
            textures.append(f"uniformity (effect size: {uniformity_effect:.2f})")
        if 'entropy' in top_features['feature'].values:
            entropy_effect = top_features[top_features['feature'] == 'entropy'].iloc[0]['effect_size']
            textures.append(f"pattern complexity (effect size: {entropy_effect:.2f})")

        if textures:
            report.append(f"   - The species differ significantly in {', '.join(textures)}")
    else:
        report.append("\nNo statistically significant texture differences were found between the species.")

    return "\n".join(report)


def create_html_report(markdown_report, df, analysis_results, effect_size_df):
    """
    Create an HTML report with interactive elements
    """
    import markdown

    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_report)

    # Create HTML structure
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Texture Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            code {{ background-color: #f8f8f8; padding: 2px 5px; border-radius: 3px; }}
            pre {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            .image-container {{ margin: 20px 0; text-align: center; }}
            .image-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .feature-description {{ background-color: #fff8e1; padding: 10px; border-left: 5px solid #ffc107; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Texture Analysis Report</h1>
        <div>{html_content}</div>

        <div class="image-container">
            <h2>Feature Importance Visualization</h2>
            <img src="feature_importance_heatmap.png" alt="Feature Importance Heatmap">
        </div>
    </body>
    </html>
    """

    # Write to file
    with open(HTML_REPORT_PATH, 'w') as f:
        f.write(html)

    print(f"HTML report generated: {HTML_REPORT_PATH}")


def main():
    print("Starting enhanced texture analysis...")

    # Load features
    features = load_features()

    # Convert to DataFrame
    df = create_feature_dataframe(features)

    # Perform analysis
    analysis_results, effect_size_df = analyze_features(df)

    # Create visualizations
    create_visualizations(df, analysis_results, effect_size_df)

    # Generate report
    report = generate_comparison_report(analysis_results, effect_size_df)

    # Save report as markdown
    with open(os.path.join(OUTPUT_DIR, 'texture_analysis_report.md'), 'w') as f:
        f.write(report)

    # Create HTML report
    create_html_report(report, df, analysis_results, effect_size_df)

    # Export to CSV for further analysis
    df.to_csv(os.path.join(OUTPUT_DIR, 'texture_features.csv'), index=False)
    effect_size_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}")
    print("Key files:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'texture_features.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'feature_importance.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'texture_analysis_report.md')}")
    print(f"  - {HTML_REPORT_PATH}")


if __name__ == "__main__":
    main()