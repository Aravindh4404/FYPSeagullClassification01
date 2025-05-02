import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create output directory
OUTPUT_DIR = "Transitions_Analysis"
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
    if 'transitions_histogram' in df.columns:
        df['transitions_histogram'] = df['transitions_histogram'].apply(
            lambda x: parse_array_string(x) if isinstance(x, str) else x)

    return df


def analyze_transitions_patterns(data, lbp_method):
    """Analyze and visualize transitions patterns"""
    regions = data['region'].unique()

    # Calculate mean transitions for each image
    data['mean_transitions'] = data['transitions_histogram'].apply(lambda x: np.sum(np.arange(len(x)) * x))

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
        slaty_trans = np.mean(np.stack(slaty_data['transitions_histogram'].values), axis=0)
        glaucous_trans = np.mean(np.stack(glaucous_data['transitions_histogram'].values), axis=0)

        # Calculate central tendencies
        slaty_trans_mean = np.sum(np.arange(len(slaty_trans)) * slaty_trans)
        glaucous_trans_mean = np.sum(np.arange(len(glaucous_trans)) * glaucous_trans)

        # Calculate percentage differences
        mean_diff = abs(slaty_trans_mean - glaucous_trans_mean) / max(slaty_trans_mean, glaucous_trans_mean) * 100

        # Calculate KL divergence
        kl_div = (stats.entropy(slaty_trans + 1e-10, glaucous_trans + 1e-10) +
                  stats.entropy(glaucous_trans + 1e-10, slaty_trans + 1e-10)) / 2

        # Add to summary statistics
        summary_stats.append({
            'region': region,
            'slaty_mean_transitions': slaty_trans_mean,
            'glaucous_mean_transitions': glaucous_trans_mean,
            'mean_diff_pct': mean_diff,
            'kl_divergence': kl_div
        })

        # Plot transitions histograms
        plt.figure(figsize=(12, 6))
        bins = np.arange(len(slaty_trans))
        plt.bar(bins - 0.2, slaty_trans, width=0.4, label='Slaty-backed Gull', alpha=0.7, color='#3274A1')
        plt.bar(bins + 0.2, glaucous_trans, width=0.4, label='Glaucous-winged Gull', alpha=0.7, color='#E1812C')
        plt.title(f'Transitions Histogram - {region} ({lbp_method.capitalize()})', fontsize=14)
        plt.xlabel('Number of Transitions in Pattern', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()

        # Add statistics as text
        plt.figtext(0.5, 0.01,
                    f"Mean: Slaty={slaty_trans_mean:.2f}, Glaucous={glaucous_trans_mean:.2f} ({mean_diff:.2f}% diff) | "
                    f"KL Div: {kl_div:.4f}",
                    ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, f"{lbp_method}_{region}_transitions_histogram.png"), dpi=300)
        plt.close()

        # Plot relationship between mean transitions and texture complexity (std_intensity)
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='mean_transitions', y='std_intensity', hue='species', data=region_data)
        plt.title(f'Mean Transitions vs. Texture Complexity - {region}', fontsize=14)
        plt.xlabel('Mean Transitions', fontsize=12)
        plt.ylabel('Standard Deviation of Intensity', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{lbp_method}_{region}_transitions_vs_complexity.png"), dpi=300)
        plt.close()

    # Create summary dataframe
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, f"{lbp_method}_transitions_summary.csv"), index=False)

    return summary_df


def main():
    """Main function to analyze both uniform and default LBP data for transitions"""
    # Analyze uniform LBP data
    # uniform_data = load_and_prepare_data("../LBP_Abstract_Features/lbp_abstract_features_uniform.csv")
    # if uniform_data is not None:
    #     print("Analyzing transitions in uniform LBP data...")
    #     analyze_transitions_patterns(uniform_data, lbp_method="uniform")

    # Analyze default LBP data
    default_data = load_and_prepare_data("../LBP_Abstract_Features/lbp_abstract_features_default.csv")
    if default_data is not None:
        print("Analyzing transitions in default LBP data...")
        analyze_transitions_patterns(default_data, lbp_method="default")

    print(f"Transitions analysis complete! Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
