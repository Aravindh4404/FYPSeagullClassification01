import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create output directory
OUTPUT_DIR = "Number_of_Ones_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_array_string(array_str):
    """Parse numpy array string representation"""
    try:
        clean_str = array_str.replace('[', '').replace(']', '').strip()
        values = [float(x) for x in clean_str.split()]
        return np.array(values)
    except Exception as e:
        print(f"Error parsing array string: {e}")
        return np.array([])


def load_and_prepare_data(csv_path):
    """Load and prepare the LBP feature data for analysis"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Convert string representations of lists to actual numpy arrays
    if 'ones_histogram' in df.columns:
        df['ones_histogram'] = df['ones_histogram'].apply(lambda x: parse_array_string(x) if isinstance(x, str) else x)

    return df


def analyze_ones_patterns(data, lbp_method):
    """Analyze and visualize number of ones patterns"""
    regions = data['region'].unique()

    # Calculate mean number of ones for each image
    data['mean_ones'] = data['ones_histogram'].apply(lambda x: np.sum(np.arange(len(x)) * x))

    # Create summary statistics
    summary_stats = []

    for region in regions:
        print(f"Analyzing {region} region...")
        region_data = data[data['region'] == region]

        # Get data for each species
        slaty_data = region_data[region_data['species'] == 'Slaty_Backed_Gull']
        glaucous_data = region_data[region_data['species'] == 'Glaucous_Winged_Gull']

        if len(slaty_data) == 0 or len(glaucous_data) == 0:
            print(f"  Not enough data for both species in {region} region.")
            continue

        # Calculate average histograms for each species
        slaty_ones = np.mean(np.stack(slaty_data['ones_histogram'].values), axis=0)
        glaucous_ones = np.mean(np.stack(glaucous_data['ones_histogram'].values), axis=0)

        # Calculate central tendencies
        slaty_ones_mean = np.sum(np.arange(len(slaty_ones)) * slaty_ones)
        glaucous_ones_mean = np.sum(np.arange(len(glaucous_ones)) * glaucous_ones)

        # Calculate percentage differences
        mean_diff = abs(slaty_ones_mean - glaucous_ones_mean) / max(slaty_ones_mean, glaucous_ones_mean) * 100

        # Calculate KL divergence
        kl_div = (stats.entropy(slaty_ones + 1e-10, glaucous_ones + 1e-10) +
                  stats.entropy(glaucous_ones + 1e-10, slaty_ones + 1e-10)) / 2

        # Add to summary statistics
        summary_stats.append({
            'region': region,
            'slaty_mean_ones': slaty_ones_mean,
            'glaucous_mean_ones': glaucous_ones_mean,
            'mean_diff_pct': mean_diff,
            'kl_divergence': kl_div
        })

        # Plot number of ones histograms
        plt.figure(figsize=(12, 6))
        bins = np.arange(len(slaty_ones))
        plt.bar(bins - 0.2, slaty_ones, width=0.4, label='Slaty-backed Gull', alpha=0.7, color='#3274A1')
        plt.bar(bins + 0.2, glaucous_ones, width=0.4, label='Glaucous-winged Gull', alpha=0.7, color='#E1812C')
        plt.title(f'Number of Ones Histogram - {region} ({lbp_method.capitalize()})', fontsize=14)
        plt.xlabel('Number of Ones in Pattern', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()

        # Add statistics as text
        plt.figtext(0.5, 0.01,
                    f"Mean: Slaty={slaty_ones_mean:.2f}, Glaucous={glaucous_ones_mean:.2f} ({mean_diff:.2f}% diff) | "
                    f"KL Div: {kl_div:.4f}",
                    ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, f"{lbp_method}_{region}_ones_histogram.png"), dpi=300)
        plt.close()

        # Plot relationship between mean ones and intensity
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='mean_ones', y='mean_intensity', hue='species', data=region_data)
        plt.title(f'Mean Number of Ones vs. Mean Intensity - {region}', fontsize=14)
        plt.xlabel('Mean Number of Ones', fontsize=12)
        plt.ylabel('Mean Intensity', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{lbp_method}_{region}_ones_vs_intensity.png"), dpi=300)
        plt.close()

    # Create summary dataframe
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, f"{lbp_method}_ones_summary.csv"), index=False)

    return summary_df


def main():
    """Main function to analyze both uniform and default LBP data for number of ones"""
    # Analyze uniform LBP data
    # uniform_data = load_and_prepare_data("../LBP_Abstract_Features/lbp_abstract_features_uniform.csv")
    # if uniform_data is not None:
    #     print("Analyzing number of ones in uniform LBP data...")
    #     analyze_ones_patterns(uniform_data, lbp_method="uniform")

    # Analyze default LBP data
    default_data = load_and_prepare_data("../LBP_Abstract_Features/lbp_abstract_features_default.csv")
    if default_data is not None:
        print("Analyzing number of ones in default LBP data...")
        analyze_ones_patterns(default_data, lbp_method="default")

    print(f"Number of Ones analysis complete! Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
