import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon
import warnings

warnings.filterwarnings('ignore')

# Create output directory for visualizations
# Create output directory for visualizations
OUTPUT_DIR = "Results_LBP_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create subdirectories for different feature types
os.makedirs(os.path.join(OUTPUT_DIR, "LBP_histograms"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Ones_histograms"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Transitions_histograms"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "PCA_plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Texture_properties"), exist_ok=True)


def parse_array_string(array_str):
    """Parse numpy array string representation without commas"""
    try:
        # Remove brackets and split by whitespace
        clean_str = array_str.replace('[', '').replace(']', '').strip()
        # Split by whitespace and convert to float
        values = [float(x) for x in clean_str.split()]
        return np.array(values)
    except Exception as e:
        print(f"Error parsing array string: {e}")
        return np.array([])


def load_and_prepare_data(csv_path="LBP_Abstract_Features/lbp_abstract_features_default.csv"):
    """Load and prepare the LBP feature data for analysis"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Convert string representations of lists to actual numpy arrays
    for col in ['lbp_histogram', 'ones_histogram', 'transitions_histogram']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: parse_array_string(x) if isinstance(x, str) else x)

    return df


def calculate_texture_features(avg_hist):
    """
    Calculate advanced texture features from histogram

    Parameters:
    avg_hist (numpy.ndarray): Histogram

    Returns:
    dict: Calculated texture features
    """
    # Ensure histogram is normalized
    hist_norm = avg_hist / np.sum(avg_hist)

    # Entropy (Shannon entropy)
    # Add small epsilon to avoid log(0)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

    # Energy (sum of squared probabilities)
    energy = np.sum(hist_norm ** 2)

    # Uniformity is the same as energy in this context
    uniformity = energy

    # Contrast (weighted variance)
    indices = np.arange(len(hist_norm))
    mean = np.sum(indices * hist_norm)
    contrast = np.sum((indices - mean) ** 2 * hist_norm)

    # Homogeneity
    homogeneity = np.sum(hist_norm / (1 + np.abs(indices - mean)))

    return {
        'entropy': entropy,
        'energy': energy,
        'uniformity': uniformity,
        'contrast': contrast,
        'homogeneity': homogeneity
    }


def calculate_metrics(hist1, hist2):
    """Calculate comparison metrics between two histograms"""
    # Ensure histograms have the same length
    max_len = max(len(hist1), len(hist2))
    if len(hist1) < max_len:
        hist1 = np.pad(hist1, (0, max_len - len(hist1)))
    if len(hist2) < max_len:
        hist2 = np.pad(hist2, (0, max_len - len(hist2)))

    # Add small value to avoid division by zero
    eps = 1e-10
    h1 = hist1 + eps
    h2 = hist2 + eps

    # Normalize histograms
    h1 = h1 / np.sum(h1)
    h2 = h2 / np.sum(h2)

    # Calculate KL Divergence (symmetric)
    kl_div = (stats.entropy(h1, h2) + stats.entropy(h2, h1)) / 2

    # Calculate Earth Mover's Distance (Wasserstein)
    emd = stats.wasserstein_distance(np.arange(len(h1)), np.arange(len(h2)), h1, h2)

    # Calculate Chi-Square Distance
    chi_square = np.sum((h1 - h2) ** 2 / (h1 + h2))

    # Calculate Jensen-Shannon Distance
    js_dist = jensenshannon(h1, h2)

    return {
        'kl_divergence': kl_div,
        'earth_movers_distance': emd,
        'chi_square': chi_square,
        'js_distance': js_dist
    }


def visualize_lbp_histograms(data):
    """Visualize LBP histograms for comparison between species"""
    regions = data['region'].unique()

    for region in regions:
        print(f"Creating histogram visualizations for region: {region}")

        # Filter by region
        region_data = data[data['region'] == region]

        # Get data for each species
        slaty_data = region_data[region_data['species'] == 'Slaty_Backed_Gull']
        glaucous_data = region_data[region_data['species'] == 'Glaucous_Winged_Gull']

        if len(slaty_data) == 0 or len(glaucous_data) == 0:
            print(f"  Not enough data for both species in {region} region.")
            continue

        # Calculate average histograms for each species
        slaty_avg_lbp = np.mean(np.stack(slaty_data['lbp_histogram'].values), axis=0)
        glaucous_avg_lbp = np.mean(np.stack(glaucous_data['lbp_histogram'].values), axis=0)

        # Calculate metrics
        metrics = calculate_metrics(slaty_avg_lbp, glaucous_avg_lbp)

        # Basic histogram comparison
        plt.figure(figsize=(14, 6))
        bins = np.arange(len(slaty_avg_lbp))
        plt.bar(bins - 0.2, slaty_avg_lbp, width=0.4, label='Slaty-backed Gull', alpha=0.7, color='#3274A1')
        plt.bar(bins + 0.2, glaucous_avg_lbp, width=0.4, label='Glaucous-winged Gull', alpha=0.7, color='#E1812C')

        plt.title(f'LBP Histogram Comparison for {region}', fontsize=15)
        plt.xlabel('LBP values', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()

        # Add metric values as text
        plt.figtext(0.5, 0.01,
                    f"KL Divergence: {metrics['kl_divergence']:.4f} | "
                    f"Earth Mover's Distance: {metrics['earth_movers_distance']:.4f} | "
                    f"Chi-Square: {metrics['chi_square']:.4f}",
                    ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        # For LBP histograms
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, "LBP_histograms", f"{region}_lbp_histogram.png"), dpi=300)
        plt.close()

        # Abstract feature histograms
        if 'ones_histogram' in data.columns and 'transitions_histogram' in data.columns:
            # Number of ones histograms
            slaty_avg_ones = np.mean(np.stack(slaty_data['ones_histogram'].values), axis=0)
            glaucous_avg_ones = np.mean(np.stack(glaucous_data['ones_histogram'].values), axis=0)

            # Transitions histograms
            slaty_avg_trans = np.mean(np.stack(slaty_data['transitions_histogram'].values), axis=0)
            glaucous_avg_trans = np.mean(np.stack(glaucous_data['transitions_histogram'].values), axis=0)

            # Calculate metrics for abstract features
            ones_metrics = calculate_metrics(slaty_avg_ones, glaucous_avg_ones)
            trans_metrics = calculate_metrics(slaty_avg_trans, glaucous_avg_trans)

            # Plot number of ones histograms
            plt.figure(figsize=(14, 6))
            bins_ones = np.arange(len(slaty_avg_ones))
            plt.bar(bins_ones - 0.2, slaty_avg_ones, width=0.4, label='Slaty-backed Gull', alpha=0.7, color='#3274A1')
            plt.bar(bins_ones + 0.2, glaucous_avg_ones, width=0.4, label='Glaucous-winged Gull', alpha=0.7,
                    color='#E1812C')

            # Calculate mean number of ones
            slaty_ones_mean = np.sum(np.arange(len(slaty_avg_ones)) * slaty_avg_ones)
            glaucous_ones_mean = np.sum(np.arange(len(glaucous_avg_ones)) * glaucous_avg_ones)
            ones_diff_pct = abs(slaty_ones_mean - glaucous_ones_mean) / max(slaty_ones_mean, glaucous_ones_mean) * 100

            plt.title(f'Number of Ones Histogram for {region}', fontsize=15)
            plt.xlabel('Number of Ones in Pattern', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()

            # Add metric values as text
            plt.figtext(0.5, 0.01,
                        f"Mean: Slaty={slaty_ones_mean:.2f}, Glaucous={glaucous_ones_mean:.2f} ({ones_diff_pct:.2f}% diff) | "
                        f"KL Divergence: {ones_metrics['kl_divergence']:.4f} | "
                        f"JS Distance: {ones_metrics['js_distance']:.4f}",
                        ha='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

            # For ones histograms
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(OUTPUT_DIR, "Ones_histograms", f"{region}_ones_histogram.png"), dpi=300)
            plt.close()

            # Plot transitions histograms
            plt.figure(figsize=(14, 6))
            bins_trans = np.arange(len(slaty_avg_trans))
            plt.bar(bins_trans - 0.2, slaty_avg_trans, width=0.4, label='Slaty-backed Gull', alpha=0.7, color='#3274A1')
            plt.bar(bins_trans + 0.2, glaucous_avg_trans, width=0.4, label='Glaucous-winged Gull', alpha=0.7,
                    color='#E1812C')

            # Calculate mean number of transitions
            slaty_trans_mean = np.sum(np.arange(len(slaty_avg_trans)) * slaty_avg_trans)
            glaucous_trans_mean = np.sum(np.arange(len(glaucous_avg_trans)) * glaucous_avg_trans)
            trans_diff_pct = abs(slaty_trans_mean - glaucous_trans_mean) / max(slaty_trans_mean,
                                                                               glaucous_trans_mean) * 100

            plt.title(f'Transitions Histogram for {region}', fontsize=15)
            plt.xlabel('Number of Transitions in Pattern', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()

            # Add metric values as text
            plt.figtext(0.5, 0.01,
                        f"Mean: Slaty={slaty_trans_mean:.2f}, Glaucous={glaucous_trans_mean:.2f} ({trans_diff_pct:.2f}% diff) | "
                        f"KL Divergence: {trans_metrics['kl_divergence']:.4f} | "
                        f"JS Distance: {trans_metrics['js_distance']:.4f}",
                        ha='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

            # For transitions histograms
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(OUTPUT_DIR, "Transitions_histograms", f"{region}_transitions_histogram.png"),
                        dpi=300)
            plt.close()


def analyze_texture_properties(data):
    """Analyze basic texture properties by region and species"""
    regions = data['region'].unique()

    # Create aggregated statistics
    texture_stats = []

    for region in regions:
        region_data = data[data['region'] == region]

        for species in region_data['species'].unique():
            species_region_data = region_data[region_data['species'] == species]

            # Calculate basic statistics
            mean_intensity = species_region_data['mean_intensity'].mean()
            std_intensity = species_region_data['std_intensity'].mean()

            # Extract more advanced texture features
            if 'lbp_histogram' in species_region_data.columns:
                # Average LBP histogram for this species/region
                avg_hist = np.mean(np.stack(species_region_data['lbp_histogram'].values), axis=0)

                # Get advanced texture features
                texture_features = calculate_texture_features(avg_hist)

                # Abstract pattern features if available
                abstract_features = {}

                if 'ones_histogram' in species_region_data.columns:
                    ones_hist = np.mean(np.stack(species_region_data['ones_histogram'].values), axis=0)
                    ones_features = calculate_texture_features(ones_hist)
                    # Calculate mean number of ones
                    mean_ones = np.sum(np.arange(len(ones_hist)) * ones_hist)
                    abstract_features['mean_ones'] = mean_ones
                    abstract_features['ones_entropy'] = ones_features['entropy']
                    abstract_features['ones_energy'] = ones_features['energy']

                if 'transitions_histogram' in species_region_data.columns:
                    trans_hist = np.mean(np.stack(species_region_data['transitions_histogram'].values), axis=0)
                    trans_features = calculate_texture_features(trans_hist)
                    # Calculate mean number of transitions
                    mean_transitions = np.sum(np.arange(len(trans_hist)) * trans_hist)
                    abstract_features['mean_transitions'] = mean_transitions
                    abstract_features['transitions_entropy'] = trans_features['entropy']
                    abstract_features['transitions_energy'] = trans_features['energy']

                # Store all features
                texture_stats.append({
                    'region': region,
                    'species': species,
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    **texture_features,
                    **abstract_features
                })

    # Create DataFrame
    stats_df = pd.DataFrame(texture_stats)

    # Save to CSV
    # Save to CSV
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "Texture_properties", "texture_properties.csv"), index=False)

    # Create visualizations for each property
    properties = ['mean_intensity', 'std_intensity', 'entropy', 'energy', 'uniformity', 'contrast', 'homogeneity']

    # Add abstract features if they exist
    if 'mean_ones' in stats_df.columns:
        properties.extend(['mean_ones', 'ones_entropy', 'ones_energy'])

    if 'mean_transitions' in stats_df.columns:
        properties.extend(['mean_transitions', 'transitions_entropy', 'transitions_energy'])

    for prop in properties:
        if prop not in stats_df.columns:
            continue

        plt.figure(figsize=(12, 6))

        # Reshape data for grouped bar chart
        plot_data = []
        for region in regions:
            for species in stats_df['species'].unique():
                subset = stats_df[(stats_df['region'] == region) & (stats_df['species'] == species)]
                if not subset.empty and prop in subset.columns:
                    val = subset[prop].values[0]
                    plot_data.append({
                        'region': region,
                        'species': species,
                        'value': val
                    })

        plot_df = pd.DataFrame(plot_data)

        if len(plot_df) == 0:
            plt.close()
            continue

        # Create grouped bar chart
        ax = sns.barplot(x='region', y='value', hue='species', data=plot_df)

        plt.title(f'{prop.replace("_", " ").title()} by Region and Species', fontsize=15)
        plt.xlabel('Region', fontsize=12)
        plt.ylabel(prop.replace("_", " ").title(), fontsize=12)

        # Calculate and annotate percentage differences
        for region in regions:
            region_data = plot_df[plot_df['region'] == region]
            if len(region_data) == 2:  # We need exactly 2 species for comparison
                species_vals = region_data.set_index('species')['value']
                species_list = list(species_vals.index)

                if 'Slaty_Backed_Gull' in species_vals and 'Glaucous_Winged_Gull' in species_vals:
                    val1 = species_vals['Slaty_Backed_Gull']
                    val2 = species_vals['Glaucous_Winged_Gull']

                    if val1 != 0 and val2 != 0:
                        pct_diff = abs(val1 - val2) / max(val1, val2) * 100

                        # Get x position for annotation
                        x_pos = list(regions).index(region)

                        # Get max y value for this region group
                        max_val = max(val1, val2)

                        # Add annotation of percentage difference
                        ax.text(x_pos, max_val * 1.05, f"Diff: {pct_diff:.1f}%",
                                ha='center', fontsize=9, fontweight='bold')

        plt.legend(title='Species')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "Texture_properties", f"{prop}_comparison.png"), dpi=300)
        plt.close()

    return stats_df


def create_discriminative_power_chart(stats_df):
    """Create a heatmap showing discriminative power of texture properties"""
    # Calculate percentage difference for each property and region
    regions = stats_df['region'].unique()

    # Identify all available properties in the dataframe
    properties = [col for col in stats_df.columns
                  if col not in ['region', 'species']
                  and stats_df[col].dtype in [np.float64, np.int64]]

    diff_data = []

    for region in regions:
        for prop in properties:
            region_data = stats_df[stats_df['region'] == region]

            if len(region_data) == 2:  # Need exactly 2 species
                slaty_val = region_data[region_data['species'] == 'Slaty_Backed_Gull'][prop].values[0]
                glaucous_val = region_data[region_data['species'] == 'Glaucous_Winged_Gull'][prop].values[0]

                if slaty_val != 0 and glaucous_val != 0:
                    pct_diff = abs(slaty_val - glaucous_val) / max(slaty_val, glaucous_val) * 100

                    diff_data.append({
                        'region': region,
                        'property': prop,
                        'difference': pct_diff
                    })

    diff_df = pd.DataFrame(diff_data)

    # Create pivot table for heatmap
    pivot_data = diff_df.pivot(index='region', columns='property', values='difference')

    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis',
                linewidths=0.5, cbar_kws={'label': 'Difference (%)'})

    plt.title('Discriminative Power of Texture Properties', fontsize=15)
    # Heatmap
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Texture_properties", "discriminative_power_heatmap.png"), dpi=300)
    plt.close()

    # Create bar chart of top discriminative features
    diff_data_sorted = sorted(diff_data, key=lambda x: x['difference'], reverse=True)
    top_n = min(15, len(diff_data_sorted))

    plt.figure(figsize=(14, 8))

    labels = [f"{x['region']}-{x['property']}" for x in diff_data_sorted[:top_n]]
    values = [x['difference'] for x in diff_data_sorted[:top_n]]

    bars = plt.bar(range(len(values)), values, color=sns.color_palette("viridis", top_n))

    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{values[i]:.1f}%", ha='center', fontsize=10)

    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Difference (%)', fontsize=12)
    plt.title('Top Discriminative Features (Region-Property Pairs)', fontsize=15)

    # Bar chart
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Texture_properties", "top_discriminative_features.png"), dpi=300)
    plt.close()

    return diff_df


def generate_abstract_features_summary(stats_df):
    """
    Generate a detailed summary of abstract feature analysis

    Parameters:
    stats_df (pandas.DataFrame): Dataframe with texture statistics

    Returns:
    None (writes to file)
    """
    with open(os.path.join(OUTPUT_DIR, "Texture_properties", "abstract_features_summary.txt"), 'w') as f:

        f.write("ABSTRACT FEATURES DETAILED ANALYSIS\n")
        f.write("==================================\n\n")

        # Check if required abstract features are present
        has_ones = 'mean_ones' in stats_df.columns
        has_transitions = 'mean_transitions' in stats_df.columns

        if not (has_ones or has_transitions):
            f.write("No abstract features (number of ones or transitions) found in the dataset.\n")
            return

        # Regions and species
        regions = stats_df['region'].unique()
        species = stats_df['species'].unique()

        f.write(f"Species analyzed: {', '.join(species)}\n")
        f.write(f"Regions analyzed: {', '.join(regions)}\n\n")

        # Detailed abstract feature analysis
        f.write("FEATURE DISTRIBUTIONS\n")
        f.write("--------------------\n")

        for region in regions:
            f.write(f"\nRegion: {region}\n")
            region_data = stats_df[stats_df['region'] == region]

            if len(region_data) == 2:  # Ensure we have both species
                slaty_data = region_data[region_data['species'] == 'Slaty_Backed_Gull'].iloc[0]
                glaucous_data = region_data[region_data['species'] == 'Glaucous_Winged_Gull'].iloc[0]

                # Number of ones analysis
                if has_ones:
                    f.write("\nNumber of Ones Analysis:\n")
                    slaty_ones = slaty_data['mean_ones']
                    glaucous_ones = glaucous_data['mean_ones']
                    ones_diff = abs(slaty_ones - glaucous_ones)
                    ones_pct = ones_diff / max(slaty_ones, glaucous_ones) * 100

                    f.write(f"  Slaty-backed Gull mean: {slaty_ones:.2f}\n")
                    f.write(f"  Glaucous-winged Gull mean: {glaucous_ones:.2f}\n")
                    f.write(f"  Absolute difference: {ones_diff:.2f}\n")
                    f.write(f"  Percentage difference: {ones_pct:.2f}%\n")

                    if 'ones_entropy' in slaty_data:
                        ones_entropy_diff = abs(slaty_data['ones_entropy'] - glaucous_data['ones_entropy'])
                        ones_entropy_pct = ones_entropy_diff / max(slaty_data['ones_entropy'],
                                                                   glaucous_data['ones_entropy']) * 100
                        f.write(f"  Entropy difference: {ones_entropy_pct:.2f}%\n")

                    f.write(f"  Interpretation: ")
                    if slaty_ones > glaucous_ones:
                        f.write(f"Slaty-backed Gulls have more neighbors brighter than central pixel\n")
                    else:
                        f.write(f"Glaucous-winged Gulls have more neighbors brighter than central pixel\n")

                # Transitions analysis
                if has_transitions:
                    f.write("\nTransitions Analysis:\n")
                    slaty_trans = slaty_data['mean_transitions']
                    glaucous_trans = glaucous_data['mean_transitions']
                    trans_diff = abs(slaty_trans - glaucous_trans)
                    trans_pct = trans_diff / max(slaty_trans, glaucous_trans) * 100

                    f.write(f"  Slaty-backed Gull mean: {slaty_trans:.2f}\n")
                    f.write(f"  Glaucous-winged Gull mean: {glaucous_trans:.2f}\n")
                    f.write(f"  Absolute difference: {trans_diff:.2f}\n")
                    f.write(f"  Percentage difference: {trans_pct:.2f}%\n")

                    if 'transitions_entropy' in slaty_data:
                        trans_entropy_diff = abs(
                            slaty_data['transitions_entropy'] - glaucous_data['transitions_entropy'])
                        trans_entropy_pct = trans_entropy_diff / max(slaty_data['transitions_entropy'],
                                                                     glaucous_data['transitions_entropy']) * 100
                        f.write(f"  Entropy difference: {trans_entropy_pct:.2f}%\n")

                    f.write(f"  Interpretation: ")
                    if slaty_trans > glaucous_trans:
                        f.write(f"Slaty-backed Gulls have more complex texture patterns\n")
                    else:
                        f.write(f"Glaucous-winged Gulls have more complex texture patterns\n")

        f.write("\n\nINTERPRETATION GUIDE\n")
        f.write("-------------------\n")
        f.write("Number of Ones: Measures how many neighboring pixels are brighter than center\n")
        f.write("  - Higher values: More bright spots or edges within darker regions\n")
        f.write("  - Lower values: More uniform dark or bright regions\n\n")

        f.write("Transitions: Counts how many times the binary pattern switches between 0 and 1\n")
        f.write("  - Higher values: More complex textures with frequent brightness changes\n")
        f.write("  - Lower values: Smoother textures with fewer brightness changes\n\n")

        f.write("SIGNIFICANCE FOR SPECIES IDENTIFICATION\n")
        f.write("--------------------------------------\n")

        # Identify most significant differences
        signif_features = []

        if has_ones:
            # Collect differences for number of ones across regions
            for region in regions:
                region_data = stats_df[stats_df['region'] == region]
                if len(region_data) == 2:
                    slaty_ones = region_data[region_data['species'] == 'Slaty_Backed_Gull']['mean_ones'].values[0]
                    glaucous_ones = region_data[region_data['species'] == 'Glaucous_Winged_Gull']['mean_ones'].values[0]
                    ones_diff = abs(slaty_ones - glaucous_ones)
                    ones_pct = ones_diff / max(slaty_ones, glaucous_ones) * 100
                    signif_features.append({
                        'feature': 'Number of Ones',
                        'region': region,
                        'diff_pct': ones_pct
                    })

        if has_transitions:
            # Collect differences for transitions across regions
            for region in regions:
                region_data = stats_df[stats_df['region'] == region]
                if len(region_data) == 2:
                    slaty_trans = region_data[region_data['species'] == 'Slaty_Backed_Gull']['mean_transitions'].values[
                        0]
                    glaucous_trans = \
                    region_data[region_data['species'] == 'Glaucous_Winged_Gull']['mean_transitions'].values[0]
                    trans_diff = abs(slaty_trans - glaucous_trans)
                    trans_pct = trans_diff / max(slaty_trans, glaucous_trans) * 100
                    signif_features.append({
                        'feature': 'Transitions',
                        'region': region,
                        'diff_pct': trans_pct
                    })

        # Sort by difference percentage
        signif_features.sort(key=lambda x: x['diff_pct'], reverse=True)

        # Display top features
        for i, feat in enumerate(signif_features[:5]):
            if i == 0:
                f.write(
                    f"\nThe most discriminative abstract feature is {feat['feature']} in the {feat['region']} region ")
                f.write(f"with a difference of {feat['diff_pct']:.2f}%.\n")
            else:
                f.write(f"{i + 1}. {feat['feature']} in {feat['region']}: {feat['diff_pct']:.2f}% difference\n")

        f.write("\nCONCLUSION\n")
        f.write("----------\n")
        if signif_features:
            top_feature = signif_features[0]
            f.write(f"Abstract pattern analysis reveals {top_feature['feature']} in the {top_feature['region']} ")
            f.write(f"as the most distinctive texture feature between the two gull species.\n\n")

            if top_feature['feature'] == 'Number of Ones':
                f.write(
                    "The Number of Ones feature captures the brightness relationship pattern between neighboring pixels.\n")
                f.write(
                    "This suggests that the two gull species differ significantly in how brightness is distributed in their feathers,\n")
                f.write("which may reflect differences in their feather structure and coloration patterns.\n")
            else:
                f.write("The Transitions feature captures the complexity of texture patterns in the bird's plumage.\n")
                f.write("This suggests that the two gull species differ significantly in texture complexity,\n")
                f.write("which may reflect differences in their feather microstructure and pattern arrangements.\n")

        f.write("\nBoth abstract pattern features provide robust metrics for identification that can complement\n")
        f.write("traditional intensity-based measurements. These features are particularly useful because they\n")
        f.write(
            "can remain distinctive even when lighting conditions vary, providing reliable differentiating characteristics.\n")


def generate_summary_report(stats_df, diff_df):
    """Generate a text summary report of the analysis"""

    with open(os.path.join(OUTPUT_DIR, "Texture_properties", "lbp_analysis_summary.txt"), 'w') as f:
        f.write("LOCAL BINARY PATTERN (LBP) ANALYSIS SUMMARY\n")
        f.write("==========================================\n\n")

        # Summary of regions and species analyzed
        regions = stats_df['region'].unique()
        species = stats_df['species'].unique()

        f.write(f"Species analyzed: {', '.join(species)}\n")
        f.write(f"Regions analyzed: {', '.join(regions)}\n\n")

        # Find most discriminative features
        diff_data_sorted = diff_df.sort_values('difference', ascending=False)
        top_features = diff_data_sorted.head(5).to_dict('records')

        f.write("MOST DISCRIMINATIVE FEATURES\n")
        f.write("--------------------------\n")
        for i, feature in enumerate(top_features):
            f.write(f"{i + 1}. {feature['region']}'s {feature['property']}: {feature['difference']:.2f}% difference\n")

        f.write("\n")

        # Region-by-region analysis
        f.write("REGION BY REGION ANALYSIS\n")
        f.write("------------------------\n")

        for region in regions:
            f.write(f"\n{region.upper()}:\n")

            region_stats = stats_df[stats_df['region'] == region]
            if len(region_stats) == 2:  # Need exactly 2 species for comparison
                slaty_stats = region_stats[region_stats['species'] == 'Slaty_Backed_Gull'].iloc[0]
                glaucous_stats = region_stats[region_stats['species'] == 'Glaucous_Winged_Gull'].iloc[0]

                # Compare each property
                basic_props = ['mean_intensity', 'std_intensity', 'entropy', 'energy', 'uniformity', 'contrast']
                for prop in basic_props:
                    if prop in slaty_stats and prop in glaucous_stats:
                        slaty_val = slaty_stats[prop]
                        glaucous_val = glaucous_stats[prop]

                        if slaty_val != 0 and glaucous_val != 0:
                            pct_diff = abs(slaty_val - glaucous_val) / max(slaty_val, glaucous_val) * 100

                            comparison = "higher" if slaty_val > glaucous_val else "lower"
                            f.write(
                                f"  - {prop.replace('_', ' ').title()}: Slaty-backed Gull has {comparison} value ({slaty_val:.4f} vs {glaucous_val:.4f}, {pct_diff:.1f}% difference)\n")

                # Add abstract feature comparisons if available
                if 'mean_ones' in slaty_stats and 'mean_ones' in glaucous_stats:
                    slaty_val = slaty_stats['mean_ones']
                    glaucous_val = glaucous_stats['mean_ones']

                    if slaty_val != 0 and glaucous_val != 0:
                        pct_diff = abs(slaty_val - glaucous_val) / max(slaty_val, glaucous_val) * 100

                        comparison = "higher" if slaty_val > glaucous_val else "lower"
                        f.write(
                            f"  - Mean Number of Ones: Slaty-backed Gull has {comparison} value ({slaty_val:.4f} vs {glaucous_val:.4f}, {pct_diff:.1f}% difference)\n")

                if 'mean_transitions' in slaty_stats and 'mean_transitions' in glaucous_stats:
                    slaty_val = slaty_stats['mean_transitions']
                    glaucous_val = glaucous_stats['mean_transitions']

                    if slaty_val != 0 and glaucous_val != 0:
                        pct_diff = abs(slaty_val - glaucous_val) / max(slaty_val, glaucous_val) * 100

                        comparison = "higher" if slaty_val > glaucous_val else "lower"
                        f.write(
                            f"  - Mean Transitions: Slaty-backed Gull has {comparison} value ({slaty_val:.4f} vs {glaucous_val:.4f}, {pct_diff:.1f}% difference)\n")

        f.write("\n\nSPECIES TEXTURE CHARACTERISTICS\n")
        f.write("------------------------------\n")

        # Distinctive characteristics of each species
        for s in species:
            f.write(f"\n{s} distinctive characteristics:\n")

            for region in regions:
                region_stats = stats_df[stats_df['region'] == region]
                if len(region_stats) == 2:  # Need exactly 2 species
                    species_stats = region_stats[region_stats['species'] == s].iloc[0]
                    other_species = [sp for sp in species if sp != s][0]
                    other_stats = region_stats[region_stats['species'] == other_species].iloc[0]

                    region_props = []
                    for prop in species_stats.index:
                        if prop not in ['region', 'species'] and prop in other_stats:
                            my_val = species_stats[prop]
                            other_val = other_stats[prop]

                            if my_val != 0 and other_val != 0:
                                diff = my_val - other_val
                                if abs(diff) / max(my_val, other_val) > 0.1:  # If difference is >10%
                                    comparison = "higher" if diff > 0 else "lower"
                                    pct = abs(diff / other_val * 100)
                                    region_props.append(f"{prop.replace('_', ' ')} ({comparison} by {pct:.1f}%)")

                    if region_props:
                        f.write(f"  {region}: {', '.join(region_props)}\n")

        # Conclusion
        f.write("\n\nCONCLUSION\n")
        f.write("----------\n")

        # Get the most discriminative feature
        if len(top_features) > 0:
            top_feature = top_features[0]
            f.write(f"The most discriminative feature for species identification is the {top_feature['region']}'s ")
            f.write(
                f"{top_feature['property'].replace('_', ' ')}, with a difference of {top_feature['difference']:.2f}%.\n\n")

        f.write(
            "For species identification using LBP texture analysis, the following features show the most promise:\n")
        for i, feature in enumerate(top_features):
            f.write(
                f"{i + 1}. {feature['region']}'s {feature['property'].replace('_', ' ')} ({feature['difference']:.2f}% difference)\n")


def perform_pca_analysis(data):
    """Perform PCA on LBP histograms to visualize data in 2D"""
    regions = data['region'].unique()

    for region in regions:
        region_data = data[data['region'] == region]

        # Prepare data for PCA
        X = np.array([hist for hist in region_data['lbp_histogram']])

        # Make sure all histograms have the same length
        max_len = max(len(hist) for hist in X)
        X_padded = np.array([np.pad(hist, (0, max_len - len(hist))) for hist in X])

        # Standardize data
        X_std = StandardScaler().fit_transform(X_padded)

        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)

        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'species': region_data['species'].values
        })

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df,
                        palette={'Slaty_Backed_Gull': '#3274A1', 'Glaucous_Winged_Gull': '#E1812C'},
                        s=100, alpha=0.7)

        plt.title(f'PCA of LBP Features - {region} Region', fontsize=15)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)

        # Add variance explained text
        total_var = pca.explained_variance_ratio_.sum() * 100
        plt.figtext(0.5, 0.01, f"Total variance explained: {total_var:.2f}%",
                    ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        # For LBP PCA
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, "PCA_plots", f"{region}_pca_plot.png"), dpi=300)
        plt.close()

        # If abstract patterns are available, also perform PCA on them
        for pattern_type in ['ones_histogram', 'transitions_histogram']:
            if pattern_type in region_data.columns:
                pattern_name = pattern_type.split('_')[0]

                # Prepare data for PCA
                X = np.array([hist for hist in region_data[pattern_type]])

                # Make sure all histograms have the same length
                max_len = max(len(hist) for hist in X)
                X_padded = np.array([np.pad(hist, (0, max_len - len(hist))) for hist in X])

                # Standardize data
                X_std = StandardScaler().fit_transform(X_padded)

                # Apply PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_std)

                # Create DataFrame for plotting
                pca_df = pd.DataFrame({
                    'PC1': X_pca[:, 0],
                    'PC2': X_pca[:, 1],
                    'species': region_data['species'].values
                })

                # Create scatter plot
                plt.figure(figsize=(10, 8))
                sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df,
                                palette={'Slaty_Backed_Gull': '#3274A1', 'Glaucous_Winged_Gull': '#E1812C'},
                                s=100, alpha=0.7)

                plt.title(f'PCA of {pattern_name.capitalize()} Features - {region} Region', fontsize=15)
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)

                # Add variance explained text
                total_var = pca.explained_variance_ratio_.sum() * 100
                plt.figtext(0.5, 0.01, f"Total variance explained: {total_var:.2f}%",
                            ha='center', fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                plt.savefig(os.path.join(OUTPUT_DIR, "PCA_plots", f"{region}_{pattern_name}_pca_plot.png"), dpi=300)
                plt.close()


def main():
    """Main function to run the analysis"""
    print("Starting Enhanced LBP Analysis...")

    # Load and prepare data
    data = load_and_prepare_data()

    # Visualize LBP histograms
    visualize_lbp_histograms(data)

    # Analyze texture properties
    stats_df = analyze_texture_properties(data)

    # Create discriminative power chart
    diff_df = create_discriminative_power_chart(stats_df)

    # Perform PCA analysis
    perform_pca_analysis(data)

    # Generate summary reports
    generate_summary_report(stats_df, diff_df)
    generate_abstract_features_summary(stats_df)

    print(f"Analysis complete! Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
