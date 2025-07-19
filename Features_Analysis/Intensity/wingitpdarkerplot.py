import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define consistent color scheme for species (same as your original code)
SPECIES_COLORS = {
    'Glaucous_Winged_Gull': '#3274A1',  # Blue
    'Slaty_Backed_Gull': '#E1812C'  # Orange
}

# Set plotting style for professional-looking visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def load_darkness_analysis_data():
    """
    Load the darkness analysis results from CSV file.
    """
    csv_path = "Wingtip_Darkness_Analysis/wingtip_darkness_analysis.csv"

    if not os.path.exists(csv_path):
        print(f"Darkness analysis data not found at {csv_path}")
        print("Please run 'wingtip_darkness_analyzer.py' first to generate the data.")
        return None

    df = pd.read_csv(csv_path)
    print(f"Loaded darkness analysis data for {len(df)} images")
    return df


def save_darkness_statistics_to_txt(results_df, output_dir):
    """
    Save detailed darkness percentage statistics to a text file.
    """
    txt_path = os.path.join(output_dir, 'darkness_percentage_statistics.txt')

    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("WINGTIP DARKNESS PERCENTAGE ANALYSIS STATISTICS\n")
        f.write("=" * 60 + "\n\n")

        # Overall statistics
        f.write("OVERALL DATASET STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Images Analyzed: {len(results_df)}\n")
        f.write(f"Overall Mean Darkness: {results_df['darkness_percentage'].mean():.2f}%\n")
        f.write(f"Overall Std Darkness: {results_df['darkness_percentage'].std():.2f}%\n")
        f.write(f"Overall Min Darkness: {results_df['darkness_percentage'].min():.2f}%\n")
        f.write(f"Overall Max Darkness: {results_df['darkness_percentage'].max():.2f}%\n")
        f.write(f"Overall Median Darkness: {results_df['darkness_percentage'].median():.2f}%\n\n")

        # Species-specific statistics
        f.write("SPECIES-SPECIFIC STATISTICS:\n")
        f.write("-" * 40 + "\n")

        for species in results_df['species'].unique():
            species_data = results_df[results_df['species'] == species]['darkness_percentage']
            display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

            f.write(f"\n{display_name}:\n")
            f.write(f"  Sample Size: {len(species_data)} images\n")
            f.write(f"  Mean: {species_data.mean():.2f}%\n")
            f.write(f"  Standard Deviation: {species_data.std():.2f}%\n")
            f.write(f"  Minimum: {species_data.min():.2f}%\n")
            f.write(f"  Maximum: {species_data.max():.2f}%\n")
            f.write(f"  Median: {species_data.median():.2f}%\n")
            f.write(f"  25th Percentile: {species_data.quantile(0.25):.2f}%\n")
            f.write(f"  75th Percentile: {species_data.quantile(0.75):.2f}%\n")
            f.write(f"  Interquartile Range: {species_data.quantile(0.75) - species_data.quantile(0.25):.2f}%\n")

        # Distribution analysis
        f.write("\n\nDISTRIBUTION ANALYSIS:\n")
        f.write("-" * 40 + "\n")

        # Create bins for analysis
        bins = np.arange(0, results_df['darkness_percentage'].max() + 5, 5)

        for species in results_df['species'].unique():
            species_data = results_df[results_df['species'] == species]['darkness_percentage']
            display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

            # Calculate histogram
            counts, bin_edges = np.histogram(species_data, bins=bins)

            f.write(f"\n{display_name} - Frequency Distribution (5% bins):\n")
            for i in range(len(counts)):
                if counts[i] > 0:
                    f.write(
                        f"  {bin_edges[i]:.0f}-{bin_edges[i + 1]:.0f}%: {counts[i]} images ({counts[i] / len(species_data) * 100:.1f}%)\n")

        # Statistical comparison
        f.write("\n\nSTATISTICAL COMPARISON BETWEEN SPECIES:\n")
        f.write("-" * 40 + "\n")

        species_list = results_df['species'].unique()
        if len(species_list) == 2:
            sp1_data = results_df[results_df['species'] == species_list[0]]['darkness_percentage']
            sp2_data = results_df[results_df['species'] == species_list[1]]['darkness_percentage']

            sp1_name = species_list[0].replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')
            sp2_name = species_list[1].replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

            mean_diff = sp1_data.mean() - sp2_data.mean()

            f.write(f"Mean Difference ({sp1_name} - {sp2_name}): {mean_diff:.2f}%\n")
            f.write(f"{sp1_name} has {'higher' if mean_diff > 0 else 'lower'} average darkness percentage\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Analysis completed successfully.\n")

    print(f"Statistics saved to: {txt_path}")


def create_darkness_distribution_chart(results_df):
    """
    Create a distribution/frequency chart showing darkness percentages by species.
    """
    # Create output directory
    output_dir = "Darkness_Distribution_Chart"
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Define bins for the histogram (5% intervals)
    bins = np.arange(0, results_df['darkness_percentage'].max() + 5, 5)

    # Create histogram for each species
    for species in results_df['species'].unique():
        species_data = results_df[results_df['species'] == species]['darkness_percentage']
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Create histogram with borders
        ax.hist(species_data, bins=bins, alpha=0.7, color=color,
                edgecolor='black', linewidth=1.2, label=display_name)

    # Customize the chart
    ax.set_title('Distribution of Wingtip Darkness Percentage by Species',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Darkness Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Images)', fontsize=12, fontweight='bold')

    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)

    # Customize grid and layout
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add subtle border around the plot
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # Set x-axis ticks to show bin edges
    ax.set_xticks(bins[::2])  # Show every other bin edge to avoid crowding

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'darkness_percentage_distribution.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Save statistics to text file
    save_darkness_statistics_to_txt(results_df, output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("DARKNESS PERCENTAGE DISTRIBUTION SUMMARY")
    print("=" * 50)

    for species in results_df['species'].unique():
        species_data = results_df[results_df['species'] == species]['darkness_percentage']
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')
        print(f"{display_name}: {len(species_data)} images")
        print(f"  Range: {species_data.min():.1f}% - {species_data.max():.1f}%")
        print(f"  Mean ± Std: {species_data.mean():.1f}% ± {species_data.std():.1f}%")
        print()


def main():
    """
    Main function to create the darkness percentage distribution chart.
    """
    print("Loading darkness analysis data...")

    # Load the analysis results
    results_df = load_darkness_analysis_data()

    if results_df is None:
        return

    print(f"Loaded data for {len(results_df)} images across {results_df['species'].nunique()} species")

    # Create distribution chart
    print("\nCreating darkness percentage distribution chart...")
    create_darkness_distribution_chart(results_df)

    print("\nDistribution chart and statistics saved to Darkness_Distribution_Chart directory")
    print("- PNG chart: darkness_percentage_distribution.png")
    print("- Text statistics: darkness_percentage_statistics.txt")


if __name__ == "__main__":
    main()