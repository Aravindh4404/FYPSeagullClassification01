import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create output directory
OUTPUT_DIR = "Abstract_Pattern_Analysis"
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
    for col in ['lbp_histogram', 'ones_histogram', 'transitions_histogram']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: parse_array_string(x) if isinstance(x, str) else x)

    return df


def analyze_abstract_patterns(data, output_prefix=""):
    """Analyze and visualize abstract patterns (ones and transitions)"""
    regions = data['region'].unique()

    # Create summary statistics for abstract patterns
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

        slaty_trans = np.mean(np.stack(slaty_data['transitions_histogram'].values), axis=0)
        glaucous_trans = np.mean(np.stack(glaucous_data['transitions_histogram'].values), axis=0)

        # Calculate statistics for number of ones
        ones_kl_div = (stats.entropy(slaty_ones + 1e-10, glaucous_ones + 1e-10) +
                       stats.entropy(glaucous_ones + 1e-10, slaty_ones + 1e-10)) / 2

        # Calculate mean number of ones
        slaty_ones_mean = np.sum(np.arange(len(slaty_ones)) * slaty_ones)
        glaucous_ones_mean = np.sum(np.arange(len(glaucous_ones)) * glaucous_ones)
        ones_mean_diff = abs(slaty_ones_mean - glaucous_ones_mean) / max(slaty_ones_mean, glaucous_ones_mean) * 100

        # Calculate statistics for transitions
        trans_kl_div = (stats.entropy(slaty_trans + 1e-10, glaucous_trans + 1e-10) +
                        stats.entropy(glaucous_trans + 1e-10, slaty_trans + 1e-10)) / 2

        # Calculate mean transitions
        slaty_trans_mean = np.sum(np.arange(len(slaty_trans)) * slaty_trans)
        glaucous_trans_mean = np.sum(np.arange(len(glaucous_trans)) * glaucous_trans)
        trans_mean_diff = abs(slaty_trans_mean - glaucous_trans_mean) / max(slaty_trans_mean, glaucous_trans_mean) * 100

        # Add to summary statistics
        summary_stats.append({
            'region': region,
            'ones_kl_divergence': ones_kl_div,
            'ones_mean_slaty': slaty_ones_mean,
            'ones_mean_glaucous': glaucous_ones_mean,
            'ones_mean_diff_pct': ones_mean_diff,
            'trans_kl_divergence': trans_kl_div,
            'trans_mean_slaty': slaty_trans_mean,
            'trans_mean_glaucous': glaucous_trans_mean,
            'trans_mean_diff_pct': trans_mean_diff
        })

        # Plot number of ones histograms
        plt.figure(figsize=(12, 6))
        bins = np.arange(len(slaty_ones))
        plt.bar(bins - 0.2, slaty_ones, width=0.4, label='Slaty-backed Gull', alpha=0.7, color='#3274A1')
        plt.bar(bins + 0.2, glaucous_ones, width=0.4, label='Glaucous-winged Gull', alpha=0.7, color='#E1812C')
        plt.title(f'Number of Ones Histogram - {region}', fontsize=14)
        plt.xlabel('Number of Ones in Pattern', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()

        # Add statistics as text
        plt.figtext(0.5, 0.01,
                    f"KL Divergence: {ones_kl_div:.4f} | "
                    f"Mean Values: Slaty={slaty_ones_mean:.2f}, Glaucous={glaucous_ones_mean:.2f} ({ones_mean_diff:.2f}% diff)",
                    ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}{region}_ones_histogram.png"), dpi=300)
        plt.close()

        # Similar code for transitions histograms
        plt.figure(figsize=(12, 6))
        bins = np.arange(len(slaty_trans))
        plt.bar(bins - 0.2, slaty_trans, width=0.4, label='Slaty-backed Gull', alpha=0.7, color='#3274A1')
        plt.bar(bins + 0.2, glaucous_trans, width=0.4, label='Glaucous-winged Gull', alpha=0.7, color='#E1812C')
        plt.title(f'Transitions Histogram - {region}', fontsize=14)
        plt.xlabel('Number of Transitions in Pattern', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()

        plt.figtext(0.5, 0.01,
                    f"KL Divergence: {trans_kl_div:.4f} | "
                    f"Mean Values: Slaty={slaty_trans_mean:.2f}, Glaucous={glaucous_trans_mean:.2f} ({trans_mean_diff:.2f}% diff)",
                    ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}{region}_transitions_histogram.png"), dpi=300)
        plt.close()

    # Create summary dataframe
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, f"{output_prefix}abstract_patterns_summary.csv"), index=False)

    # Create bar chart of most discriminative features
    if len(summary_stats) > 0:
        plt.figure(figsize=(12, 8))

        # Create the data for the bar chart
        regions = summary_df['region'].tolist()
        ones_diff = summary_df['ones_mean_diff_pct'].tolist()
        trans_diff = summary_df['trans_mean_diff_pct'].tolist()

        # Combine and sort
        feature_data = []
        for i, region in enumerate(regions):
            feature_data.append(('Number of 1s', region, ones_diff[i]))
            feature_data.append(('Transitions', region, trans_diff[i]))

        df_plot = pd.DataFrame(feature_data, columns=['Feature', 'Region', 'Difference'])
        df_plot = df_plot.sort_values('Difference', ascending=False)

        # Plot
        ax = sns.barplot(x='Region', y='Difference', hue='Feature', data=df_plot)
        plt.title('Percentage Difference in Abstract Features by Region', fontsize=14)
        plt.xlabel('Region', fontsize=12)
        plt.ylabel('Percentage Difference (%)', fontsize=12)
        plt.legend(title='Feature')

        # Add values on bars
        for i, p in enumerate(ax.patches):
            ax.annotate(f"{p.get_height():.1f}%",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{output_prefix}feature_diff_barchart.png"), dpi=300)
        plt.close()

    return summary_df


def main():
    """Main function to analyze both uniform and default LBP data"""
    # Analyze uniform LBP data
    # uniform_data = load_and_prepare_data("../LBP_Abstract_Features/lbp_abstract_features_uniform.csv")
    # if uniform_data is not None:
    #     print("Analyzing uniform LBP data...")
    #     analyze_abstract_patterns(uniform_data, output_prefix="uniform_")

    # Analyze default LBP data
    default_data = load_and_prepare_data("../LBP_Abstract_Features/lbp_abstract_features_default.csv")
    if default_data is not None:
        print("Analyzing default LBP data...")
        analyze_abstract_patterns(default_data, output_prefix="default_")

    print(f"Analysis complete! Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
