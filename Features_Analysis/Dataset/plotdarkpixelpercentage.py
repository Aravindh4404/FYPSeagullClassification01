import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
# Use only dark blue and orange colors
DARK_BLUE = '#1f77b4'  # Default matplotlib blue
DARK_ORANGE = '#ff7f0e'  # Default matplotlib orange


# Create color palette cycling between dark blue and orange only
def get_blue_orange_colors(n):
    """Generate alternating dark blue and orange colors"""
    colors = []
    for i in range(n):
        if i % 2 == 0:
            colors.append(DARK_BLUE)
        else:
            colors.append(DARK_ORANGE)
    return colors


def load_and_prepare_data(file_path):
    """Load the Excel file and prepare data for plotting"""
    try:
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Check if required columns exist
        required_cols = ['species', 'method1_percentage']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            return None

        # Clean species names for better display
        df['species_clean'] = df['species'].str.replace('_', ' ').str.title()

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_comprehensive_plot(df, output_dir="plots"):
    """Create comprehensive visualization of Method 1 percentage by species"""

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # Main title
    fig.suptitle('Wingtip Dark Pixel Analysis: Method 1 (< Mean Wing Intensity) by Species',
                 fontsize=16, fontweight='bold', y=0.95)

    # Plot 1: Box plot (main visualization)
    ax1 = plt.subplot(2, 2, (1, 2))  # Takes top row

    # Create box plot
    box_plot = df.boxplot(column='method1_percentage', by='species_clean',
                          ax=ax1, patch_artist=True, return_type='dict')

    # Customize box plot colors - only dark blue and orange
    colors = get_blue_orange_colors(len(df['species_clean'].unique()))
    for patch, color in zip(box_plot['method1_percentage']['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Dark Pixel Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title(
        'Distribution of Dark Pixel Percentages by Species\n(Method 1: Pixels darker than mean wing intensity)',
        fontsize=11, fontweight='bold', pad=20)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Violin plot (bottom left)
    ax2 = plt.subplot(2, 2, 3)

    # Create custom palette with only dark blue and orange for violin plot
    n_species = len(df['species_clean'].unique())
    violin_colors = get_blue_orange_colors(n_species)

    sns.violinplot(data=df, x='species_clean', y='method1_percentage', ax=ax2, palette=violin_colors)
    ax2.set_xlabel('Species', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Dark Pixel Percentage (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Density Distribution', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Summary statistics (bottom right)
    ax3 = plt.subplot(2, 2, 4)

    # Calculate summary statistics
    summary_stats = df.groupby('species_clean')['method1_percentage'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)

    # Create bar plot of means with error bars (std)
    species_names = summary_stats.index
    means = summary_stats['mean']
    stds = summary_stats['std']

    bars = ax3.bar(range(len(species_names)), means, yerr=stds,
                   capsize=5, alpha=0.7, color=colors,
                   edgecolor='black', linewidth=0.5)

    ax3.set_xlabel('Species', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Mean Dark Pixel Percentage (%)', fontsize=10, fontweight='bold')
    ax3.set_title('Mean ± Standard Deviation', fontsize=11, fontweight='bold')
    ax3.set_xticks(range(len(species_names)))
    ax3.set_xticklabels(species_names, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + std_val + 0.5,
                 f'{mean_val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "wingtip_dark_pixels_method1_by_species.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved as: {output_path}")

    plt.show()

    return summary_stats


def create_statistical_summary_plot(df, summary_stats, output_dir="plots"):
    """Create a detailed statistical summary visualization"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Statistical Summary: Dark Pixel Percentages by Species (Method 1)',
                 fontsize=14, fontweight='bold')

    species_names = summary_stats.index
    n_species = len(species_names)
    colors = get_blue_orange_colors(n_species)

    # Plot 1: Sample sizes
    ax1.bar(range(len(species_names)), summary_stats['count'], alpha=0.8,
            color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_title('Sample Size by Species', fontweight='bold')
    ax1.set_ylabel('Number of Images')
    ax1.set_xticks(range(len(species_names)))
    ax1.set_xticklabels(species_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for i, count in enumerate(summary_stats['count']):
        ax1.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')

    # Plot 2: Mean vs Median
    x_pos = np.arange(len(species_names))
    width = 0.35

    # Use dark blue and orange for mean vs median
    ax2.bar(x_pos - width / 2, summary_stats['mean'], width, label='Mean',
            alpha=0.8, color=DARK_BLUE, edgecolor='black', linewidth=0.5)
    ax2.bar(x_pos + width / 2, summary_stats['median'], width, label='Median',
            alpha=0.8, color=DARK_ORANGE, edgecolor='black', linewidth=0.5)
    ax2.set_title('Mean vs Median Comparison', fontweight='bold')
    ax2.set_ylabel('Dark Pixel Percentage (%)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(species_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Range (Min-Max)
    ax3.bar(range(len(species_names)), summary_stats['max'] - summary_stats['min'],
            bottom=summary_stats['min'], alpha=0.8, color=colors,
            edgecolor='black', linewidth=0.5)
    ax3.set_title('Range (Min to Max)', fontweight='bold')
    ax3.set_ylabel('Dark Pixel Percentage (%)')
    ax3.set_xticks(range(len(species_names)))
    ax3.set_xticklabels(species_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Coefficient of Variation
    cv = (summary_stats['std'] / summary_stats['mean']) * 100
    ax4.bar(range(len(species_names)), cv, alpha=0.8, color=colors,
            edgecolor='black', linewidth=0.5)
    ax4.set_title('Coefficient of Variation (%)', fontweight='bold')
    ax4.set_ylabel('CV (%)')
    ax4.set_xticks(range(len(species_names)))
    ax4.set_xticklabels(species_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    # Add value labels for CV
    for i, cv_val in enumerate(cv):
        ax4.text(i, cv_val + 0.5, f'{cv_val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save the statistical summary plot
    output_path = Path(output_dir) / "wingtip_dark_pixels_statistical_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Statistical summary plot saved as: {output_path}")

    plt.show()


def print_detailed_statistics(df, summary_stats):
    """Print detailed statistical information"""

    print("\n" + "=" * 80)
    print("DETAILED STATISTICAL ANALYSIS")
    print("=" * 80)

    print(f"\nOverall Dataset Summary:")
    print(f"Total images analyzed: {len(df)}")
    print(f"Number of species: {df['species'].nunique()}")
    print(f"Overall mean dark pixel percentage: {df['method1_percentage'].mean():.2f}%")
    print(f"Overall median dark pixel percentage: {df['method1_percentage'].median():.2f}%")
    print(f"Overall standard deviation: {df['method1_percentage'].std():.2f}%")

    print(f"\nPer-Species Statistics:")
    print("-" * 80)
    print(summary_stats)

    print(f"\nSpecies Ranking (by mean dark pixel percentage):")
    print("-" * 50)
    ranked_species = summary_stats.sort_values('mean', ascending=False)
    for i, (species, stats) in enumerate(ranked_species.iterrows(), 1):
        print(f"{i:2d}. {species:20s}: {stats['mean']:6.2f}% (±{stats['std']:5.2f}%)")

    # Statistical significance testing
    print(f"\nSpecies with highest variability (Coefficient of Variation):")
    print("-" * 60)
    cv_data = []
    for species, stats in summary_stats.iterrows():
        cv = (stats['std'] / stats['mean']) * 100
        cv_data.append((species, cv))

    cv_data.sort(key=lambda x: x[1], reverse=True)
    for i, (species, cv) in enumerate(cv_data[:3], 1):
        print(f"{i}. {species}: {cv:.1f}% CV")


def main():
    """Main function to run the analysis"""

    # File path - update this to match your file location
    file_path = "cleaned_for_plotting.xlsx"  # Update this path as needed

    print("Loading data...")
    df = load_and_prepare_data(file_path)

    if df is None:
        print("Failed to load data. Please check the file path and format.")
        return

    print(f"Successfully loaded data with {len(df)} records for {df['species'].nunique()} species")

    # Create comprehensive plot
    print("\nCreating comprehensive visualization...")
    summary_stats = create_comprehensive_plot(df)

    # Create statistical summary plot
    print("\nCreating statistical summary...")
    create_statistical_summary_plot(df, summary_stats)

    # Print detailed statistics
    print_detailed_statistics(df, summary_stats)

    print("\n✅ Analysis complete! Check the 'plots' directory for output files.")


if __name__ == "__main__":
    main()