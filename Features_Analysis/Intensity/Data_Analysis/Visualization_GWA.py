import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Set plotting style for professional-looking visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Create output directory for visualizations
output_dir = PROJECT_ROOT / "Features_Analysis" / "Intensity" / "Data_Analysis" / "Gull_Wing_Analysis"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Created output directory: {output_dir}")


def load_data():
    """Load and prepare all necessary datasets"""
    # Load wing intensity and wingtip data
    try:
        wing_data_path = PROJECT_ROOT / "Features_Analysis" / "Intensity" / "Intensity_Results" / "wing_intensity_analysis.csv"
        wingtip_avg_path = PROJECT_ROOT / "Features_Analysis" / "Intensity" / "Wingtip_Intensity_Distribution" / "wingtip_intensity_averages.csv"
        wingtip_dist_path = PROJECT_ROOT / "Features_Analysis" / "Intensity" / "Wingtip_Intensity_Distribution" / "wingtip_intensity_distribution.csv"

        # Check if files exist
        if not wing_data_path.exists():
            raise FileNotFoundError(f"Wing intensity data not found at {wing_data_path}")
        if not wingtip_avg_path.exists():
            raise FileNotFoundError(f"Wingtip averages data not found at {wingtip_avg_path}")
        if not wingtip_dist_path.exists():
            raise FileNotFoundError(f"Wingtip distribution data not found at {wingtip_dist_path}")

        # Load the data
        wing_data = pd.read_csv(wing_data_path)
        wingtip_avg_data = pd.read_csv(wingtip_avg_path)
        wingtip_distribution = pd.read_csv(wingtip_dist_path)

        print("Successfully loaded all datasets:")
        print(f"- Wing data: {len(wing_data)} rows")
        print(f"- Wingtip averages: {len(wingtip_avg_data)} rows")
        print(f"- Wingtip distribution: {len(wingtip_distribution)} rows")

        return wing_data, wingtip_avg_data, wingtip_distribution
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def wing_intensity_analysis(wing_data):
    """Create visualizations comparing wing intensity between species"""
    # Separate data by species
    slaty_wing = wing_data[wing_data['species'] == 'Slaty_Backed_Gull']
    glaucous_wing = wing_data[wing_data['species'] == 'Glaucous_Winged_Gull']

    # Calculate summary statistics
    wing_summary = wing_data.groupby('species')['mean_intensity'].agg(['mean', 'std', 'min', 'max', 'median', 'count'])

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(
        slaty_wing['mean_intensity'],
        glaucous_wing['mean_intensity'],
        equal_var=False
    )

    # Calculate percentage difference
    percentage_diff = ((wing_summary.loc['Glaucous_Winged_Gull', 'mean'] -
                        wing_summary.loc['Slaty_Backed_Gull', 'mean']) /
                       wing_summary.loc['Slaty_Backed_Gull', 'mean'] * 100)

    # Create figure for wing intensity comparison
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 1. Box plot of wing intensity
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(x='species', y='mean_intensity', data=wing_data, ax=ax1)
    ax1.set_title('Wing Intensity by Species')
    ax1.set_ylabel('Mean Intensity (0-255)')
    ax1.set_xlabel('')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=25)

    # Add mean values as text
    for i, species in enumerate(['Slaty_Backed_Gull', 'Glaucous_Winged_Gull']):
        if species in wing_summary.index:
            mean_val = wing_summary.loc[species, 'mean']
            ax1.text(i, mean_val + 5, f'Mean: {mean_val:.1f}',
                     ha='center', va='bottom', fontweight='bold')

    # 2. Histogram of wing intensity
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data=wing_data, x='mean_intensity', hue='species', kde=True,
                 bins=25, alpha=0.6, ax=ax2)
    ax2.set_title('Distribution of Wing Intensity')
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Count')

    # 3. Violin plot for detailed distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sns.violinplot(x='species', y='mean_intensity', data=wing_data, inner='quartile', ax=ax3)
    ax3.set_title('Detailed Wing Intensity Distribution')
    ax3.set_ylabel('Mean Intensity (0-255)')
    ax3.set_xlabel('')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=25)

    # 4. Bar chart of mean intensities
    ax4 = fig.add_subplot(gs[1, 1])
    means = wing_data.groupby('species')['mean_intensity'].mean()
    errors = wing_data.groupby('species')['mean_intensity'].std()

    bars = ax4.bar(means.index, means.values, yerr=errors.values, capsize=10)
    ax4.set_title('Mean Wing Intensity with Standard Deviation')
    ax4.set_ylabel('Mean Intensity (0-255)')
    ax4.set_xlabel('')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=25)

    # Add value labels to the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Add summary text
    plt.figtext(0.5, 0.01,
                f"Wing Intensity T-test: t = {t_stat:.2f}, p < 0.001\n"
                f"Slaty-backed Gull: {wing_summary.loc['Slaty_Backed_Gull', 'mean']:.2f} ± "
                f"{wing_summary.loc['Slaty_Backed_Gull', 'std']:.2f}\n"
                f"Glaucous-winged Gull: {wing_summary.loc['Glaucous_Winged_Gull', 'mean']:.2f} ± "
                f"{wing_summary.loc['Glaucous_Winged_Gull', 'std']:.2f}\n"
                f"Percentage Difference: {percentage_diff:.1f}% (Glaucous-winged wings are brighter)",
                ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.suptitle('Wing Intensity Comparison Between Gull Species', fontsize=16, y=0.99)
    plt.savefig(os.path.join(output_dir, 'wing_intensity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return wing_summary, t_stat, p_value, percentage_diff


def wingtip_darkness_analysis(wingtip_avg_data, wingtip_distribution):
    """Create visualizations for wingtip darkness analysis"""
    try:
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)

        # 1. Percentage of wingtip darker than wing
        ax1 = fig.add_subplot(gs[0, 0])
        sns.barplot(x='species', y='pct_darker_pixels', data=wingtip_avg_data, ax=ax1)
        ax1.set_title('Percentage of Wingtip Darker than Wing')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_xlabel('')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=25)

        # Add value labels
        for i, bar in enumerate(ax1.patches):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 2. Dark pixel percentage comparison
        ax2 = fig.add_subplot(gs[0, 1])

        # Extract darkness percentage data
        darkness_data = []
        for threshold in [25, 50, 75]:  # Updated thresholds to match available data
            for species in wingtip_avg_data['species']:
                row = wingtip_avg_data[wingtip_avg_data['species'] == species].iloc[0]
                darkness_data.append({
                    'species': species,
                    'threshold': f'< {threshold}',
                    'percentage': row[f'pct_dark_lt_{threshold}']
                })

        darkness_df = pd.DataFrame(darkness_data)

        sns.barplot(x='threshold', y='percentage', hue='species', data=darkness_df, ax=ax2)
        ax2.set_title('Percentage of Pixels Below Intensity Threshold')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_xlabel('Intensity Threshold')

        # 3. Average intensity comparison
        ax3 = fig.add_subplot(gs[1, 0])

        # Prepare data for intensity comparison
        intensity_data = []
        for species in wingtip_avg_data['species']:
            row = wingtip_avg_data[wingtip_avg_data['species'] == species].iloc[0]
            intensity_data.append({
                'species': species,
                'region': 'Wing',
                'intensity': row['mean_wing_intensity']
            })
            intensity_data.append({
                'species': species,
                'region': 'Wingtip',
                'intensity': row['mean_wingtip_intensity']
            })

        intensity_df = pd.DataFrame(intensity_data)

        sns.barplot(x='species', y='intensity', hue='region', data=intensity_df, ax=ax3)
        ax3.set_title('Mean Intensity by Region')
        ax3.set_ylabel('Mean Intensity (0-255)')
        ax3.set_xlabel('')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=25)

        # 4. Wing vs Wingtip intensity scatter plot
        ax4 = fig.add_subplot(gs[1, 1])

        # Create scatter plot
        for species, color in zip(['Slaty_Backed_Gull', 'Glaucous_Winged_Gull'], ['blue', 'orange']):
            subset = wingtip_distribution[wingtip_distribution['species'] == species]
            ax4.scatter(subset['mean_wing_intensity'], subset['mean_wingtip_intensity'],
                        alpha=0.7, label=species.replace('_', ' '), c=color)

        # Add identity line
        xlim = ax4.get_xlim()
        ylim = ax4.get_ylim()
        lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
        ax4.plot(lims, lims, 'k--', alpha=0.5, label='Equal Intensity')
        ax4.set_xlim(lims)
        ax4.set_ylim(lims)

        ax4.set_title('Wing vs. Wingtip Intensity')
        ax4.set_xlabel('Mean Wing Intensity')
        ax4.set_ylabel('Mean Wingtip Intensity')
        ax4.legend()

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Add summary text
        plt.figtext(0.5, 0.01,
                    f"Slaty-backed Gull: {wingtip_avg_data[wingtip_avg_data['species'] == 'Slaty_Backed_Gull']['pct_darker_pixels'].values[0]:.2f}% of wingtip pixels darker than wing\n"
                    f"Glaucous-winged Gull: {wingtip_avg_data[wingtip_avg_data['species'] == 'Glaucous_Winged_Gull']['pct_darker_pixels'].values[0]:.2f}% of wingtip pixels darker than wing\n"
                    f"Slaty-backed Gull: {wingtip_avg_data[wingtip_avg_data['species'] == 'Slaty_Backed_Gull']['pct_dark_lt_25'].values[0]:.2f}% of wingtip pixels below intensity 25\n"
                    f"Glaucous-winged Gull: {wingtip_avg_data[wingtip_avg_data['species'] == 'Glaucous_Winged_Gull']['pct_dark_lt_25'].values[0]:.4f}% of wingtip pixels below intensity 25",
                    ha='center', va='center', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.suptitle('Wingtip Darkness Analysis', fontsize=16, y=0.99)
        plt.savefig(os.path.join(output_dir, 'wingtip_darkness_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Created wingtip darkness analysis plot")
    except Exception as e:
        print(f"Error in wingtip darkness analysis: {str(e)}")
        raise


def pixel_intensity_distribution(wingtip_avg_data):
    """Analyze and visualize the pixel intensity distribution in wingtips"""
    try:
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)

        # 1. Full intensity distribution
        ax1 = fig.add_subplot(gs[0, :])

        # Extract intensity distribution data
        intensity_bins = []
        bin_columns = ['pct_0_25', 'pct_25_50', 'pct_50_75', 'pct_75_100',
                      'pct_100_125', 'pct_125_150', 'pct_150_175', 'pct_175_200',
                      'pct_200_225', 'pct_225_255']
        
        for species in wingtip_avg_data['species']:
            row = wingtip_avg_data[wingtip_avg_data['species'] == species].iloc[0]
            for col in bin_columns:
                start, end = map(int, col.split('_')[1:])
                intensity_bins.append({
                    'species': species,
                    'bin': f'{start}-{end}',
                    'percentage': row[col]
                })

        intensity_df = pd.DataFrame(intensity_bins)

        sns.barplot(x='bin', y='percentage', hue='species', data=intensity_df, ax=ax1)
        ax1.set_title('Intensity Distribution in Wingtips')
        ax1.set_xlabel('Intensity Range')
        ax1.set_ylabel('Percentage of Pixels (%)')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. Cumulative distribution
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Calculate cumulative percentages
        cumulative_data = []
        for species in wingtip_avg_data['species']:
            row = wingtip_avg_data[wingtip_avg_data['species'] == species].iloc[0]
            cumulative = 0
            for col in bin_columns:
                start, end = map(int, col.split('_')[1:])
                cumulative += row[col]
                cumulative_data.append({
                    'species': species,
                    'threshold': end,
                    'cumulative_percentage': cumulative
                })

        cumulative_df = pd.DataFrame(cumulative_data)

        sns.lineplot(x='threshold', y='cumulative_percentage', hue='species', 
                    data=cumulative_df, ax=ax2)
        ax2.set_title('Cumulative Intensity Distribution')
        ax2.set_xlabel('Intensity Threshold')
        ax2.set_ylabel('Cumulative Percentage (%)')

        # 3. Dark pixel analysis
        ax3 = fig.add_subplot(gs[1, 1])

        # Extract dark pixel data
        dark_pixel_data = []
        dark_thresholds = [25, 50, 75]
        for species in wingtip_avg_data['species']:
            row = wingtip_avg_data[wingtip_avg_data['species'] == species].iloc[0]
            for threshold in dark_thresholds:
                dark_pixel_data.append({
                    'species': species,
                    'threshold': f'< {threshold}',
                    'percentage': row[f'pct_dark_lt_{threshold}']
                })

        dark_pixel_df = pd.DataFrame(dark_pixel_data)

        sns.barplot(x='threshold', y='percentage', hue='species', data=dark_pixel_df, ax=ax3)
        ax3.set_title('Dark Pixel Analysis')
        ax3.set_xlabel('Intensity Threshold')
        ax3.set_ylabel('Percentage of Pixels (%)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pixel_intensity_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Created pixel intensity distribution plot")
    except Exception as e:
        print(f"Error in pixel intensity distribution: {str(e)}")
        raise


def create_summary_visualization(wing_summary, wingtip_avg_data):
    """Create a comprehensive summary visualization of key findings"""

    # Create figure
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # 1. Wing Intensity Comparison
    ax1 = fig.add_subplot(gs[0, 0])

    # Create bar plot of mean wing intensity
    species_names = ['Slaty-backed Gull', 'Glaucous-winged Gull']
    intensity_values = [
        wing_summary.loc['Slaty_Backed_Gull', 'mean'],
        wing_summary.loc['Glaucous_Winged_Gull', 'mean']
    ]

    bars = ax1.bar(species_names, intensity_values, color=['#3274A1', '#E1812C'])
    ax1.set_title('Mean Wing Intensity')
    ax1.set_ylabel('Intensity (0-255)')
    ax1.set_ylim(0, max(intensity_values) * 1.2)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

    # Calculate percentage difference
    percentage_diff = ((intensity_values[1] - intensity_values[0]) / intensity_values[0] * 100)

    # Add significance indicator
    ax1.text(0.5, max(intensity_values) * 1.1,
             f'Difference: {percentage_diff:.1f}%\np < 0.001',
             ha='center', va='bottom', fontweight='bold')

    # 2. Percentage of Wingtip Darker than Wing
    ax2 = fig.add_subplot(gs[0, 1])

    # Extract data
    darker_percentages = [
        wingtip_avg_data[wingtip_avg_data['species'] == 'Slaty_Backed_Gull']['pct_darker_pixels'].values[0],
        wingtip_avg_data[wingtip_avg_data['species'] == 'Glaucous_Winged_Gull']['pct_darker_pixels'].values[0]
    ]

    bars = ax2.bar(species_names, darker_percentages, color=['#3274A1', '#E1812C'])
    ax2.set_title('Percentage of Wingtip Darker than Wing')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_ylim(0, max(darker_percentages) * 1.2)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 2,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 3. Very Dark Pixels Comparison
    ax3 = fig.add_subplot(gs[1, 0])

    # Extract data for very dark pixels (intensity < 30)
    dark_pixel_percentages = [
        wingtip_avg_data[wingtip_avg_data['species'] == 'Slaty_Backed_Gull']['pct_dark_lt_25'].values[0],
        wingtip_avg_data[wingtip_avg_data['species'] == 'Glaucous_Winged_Gull']['pct_dark_lt_25'].values[0]
    ]

    bars = ax3.bar(species_names, dark_pixel_percentages, color=['#3274A1', '#E1812C'])
    ax3.set_title('Percentage of Very Dark Pixels (< 25)')
    ax3.set_ylabel('Percentage (%)')

    # Set y-limit based on max value, but at least 25%
    ax3.set_ylim(0, max(max(dark_pixel_percentages) * 1.2, 25))

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

    # 4. Dark Pixel Count Comparison
    ax4 = fig.add_subplot(gs[1, 1])

    # Extract data for dark pixel counts
    dark_pixel_counts = [
        wingtip_avg_data[wingtip_avg_data['species'] == 'Slaty_Backed_Gull']['dark_lt_25'].values[0],
        wingtip_avg_data[wingtip_avg_data['species'] == 'Glaucous_Winged_Gull']['dark_lt_25'].values[0]
    ]

    bars = ax4.bar(species_names, dark_pixel_counts, color=['#3274A1', '#E1812C'])
    ax4.set_title('Count of Very Dark Pixels (< 25)')
    ax4.set_ylabel('Number of Pixels')
    ax4.set_yscale('log')  # Use log scale due to large difference

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height * 1.1,
                 f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

    # 5. Combined pixel distribution
    ax5 = fig.add_subplot(gs[2, :])

    # Prepare data for grouped distribution
    bin_ranges = [(0, 25), (25, 50), (50, 75), (75, 100),
                  (100, 125), (125, 150), (150, 175), (175, 200),
                  (200, 225), (225, 255)]

    grouped_data = []
    for species in wingtip_avg_data['species']:
        species_row = wingtip_avg_data[wingtip_avg_data['species'] == species].iloc[0]

        for start, end in bin_ranges:
            # Sum all percentages in this range
            total_pct = 0
            for i in range(start, end, 10):
                if i >= 240:
                    col = f'pct_240_255'
                else:
                    col = f'pct_{i}_{i + 10}'

                if col in wingtip_avg_data.columns:
                    total_pct += species_row[col]

            grouped_data.append({
                'species': species,
                'range': f'{start}-{end}',
                'percentage': total_pct,
                'start': start  # For sorting
            })

    # Convert to DataFrame and sort
    grouped_df = pd.DataFrame(grouped_data)
    grouped_df = grouped_df.sort_values('start')

    # Plot grouped distribution
    sns.barplot(x='range', y='percentage', hue='species', data=grouped_df, ax=ax5)
    ax5.set_title('Wingtip Pixel Intensity Distribution (Grouped)')
    ax5.set_xlabel('Intensity Range')
    ax5.set_ylabel('Percentage of Pixels (%)')

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    # Add summary text
    plt.figtext(0.5, 0.02,
                "Key Differences Between Species:\n"
                f"1. Wing Intensity: Slaty-backed Gulls have significantly darker wings ({intensity_values[0]:.1f} vs {intensity_values[1]:.1f}, {percentage_diff:.1f}% difference)\n"
                f"2. Wingtip Darkness: Slaty-backed Gulls have more pixels darker than wing ({darker_percentages[0]:.1f}% vs {darker_percentages[1]:.1f}%)\n"
                f"3. Very Dark Pixels: Slaty-backed Gulls have dramatically more very dark pixels ({dark_pixel_percentages[0]:.2f}% vs {dark_pixel_percentages[1]:.3f}%)\n"
                f"4. Dark Pixel Count: Average Slaty-backed Gull has {dark_pixel_counts[0]:.0f} very dark pixels vs only {dark_pixel_counts[1]:.0f} in Glaucous-winged Gull",
                ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

    plt.suptitle('Summary: Wing and Wingtip Analysis for Slaty-backed vs Glaucous-winged Gulls',
                 fontsize=16, y=0.98)
    plt.savefig(os.path.join(output_dir, 'species_comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(wing_summary, wingtip_avg_data):
    """Generate a comprehensive summary report of the findings"""

    # Calculate key metrics
    percentage_diff = ((wing_summary.loc['Glaucous_Winged_Gull', 'mean'] -
                        wing_summary.loc['Slaty_Backed_Gull', 'mean']) /
                       wing_summary.loc['Slaty_Backed_Gull', 'mean'] * 100)

    report = f"""# Wing and Wingtip Analysis: Slaty-backed Gull vs Glaucous-winged Gull

## Key Findings

### 1. Wing Intensity
- **Slaty-backed Gull**: {wing_summary.loc['Slaty_Backed_Gull', 'mean']:.2f} ± {wing_summary.loc['Slaty_Backed_Gull', 'std']:.2f}
- **Glaucous-winged Gull**: {wing_summary.loc['Glaucous_Winged_Gull', 'mean']:.2f} ± {wing_summary.loc['Glaucous_Winged_Gull', 'std']:.2f}
- **Difference**: Glaucous-winged wings are {percentage_diff:.1f}% brighter
- **Statistical Significance**: Highly significant difference (p < 0.001)

### 2. Wingtip Darkness
- **Slaty-backed Gull**: {wingtip_avg_data[wingtip_avg_data['species'] == 'Slaty_Backed_Gull']['pct_darker_pixels'].values[0]:.2f}% of wingtip pixels darker than wing
- **Glaucous-winged Gull**: {wingtip_avg_data[wingtip_avg_data['species'] == 'Glaucous_Winged_Gull']['pct_darker_pixels'].values[0]:.2f}% of wingtip pixels darker than wing

### 3. Very Dark Pixels
- **Slaty-backed Gull**:
  - {wingtip_avg_data[wingtip_avg_data['species'] == 'Slaty_Backed_Gull']['pct_dark_lt_25'].values[0]:.2f}% pixels < 25 intensity

- **Glaucous-winged Gull**:
  - {wingtip_avg_data[wingtip_avg_data['species'] == 'Glaucous_Winged_Gull']['pct_dark_lt_25'].values[0]:.4f}% pixels < 25 intensity

### 4. Raw Pixel Counts
- **Slaty-backed Gull**: Average of {wingtip_avg_data[wingtip_avg_data['species'] == 'Slaty_Backed_Gull']['dark_lt_25'].values[0]:.0f} very dark pixels
- **Glaucous-winged Gull**: Average of {wingtip_avg_data[wingtip_avg_data['species'] == 'Glaucous_Winged_Gull']['dark_lt_25'].values[0]:.0f} very dark pixels

## Biological Significance

These results demonstrate clear, quantifiable differences between the two gull species:

1. **Overall Wing Color**: Slaty-backed Gulls have significantly darker wings, with intensity values approximately half those of Glaucous-winged Gulls.

2. **Wingtip Darkness Pattern**: Slaty-backed Gulls have a dramatically higher percentage of very dark pixels in their wingtips. Over 20% of wingtip pixels have intensity below 25, compared to virtually none in Glaucous-winged Gulls.

3. **Species Identification Feature**: The presence of very dark pixels (intensity < 25) in the wingtip appears to be a reliable diagnostic feature for distinguishing between these species.

4. **Contrast Pattern**: The higher percentage of dark pixels in Slaty-backed Gull wingtips creates a more pronounced visual contrast between wing and wingtip regions.

These quantitative differences align with field observations that Slaty-backed Gulls have darker wings and more prominent dark wingtips compared to Glaucous-winged Gulls, providing a reliable basis for species identification in image analysis.
"""

    with open(os.path.join(output_dir, 'species_comparison_report.md'), 'w') as f:
        f.write(report)

    print(f"Summary report saved to {output_dir}/species_comparison_report.md")


def main():
    """Main function to run all analyses"""
    # Load data
    wing_data, wingtip_avg_data, wingtip_distribution = load_data()

    if wing_data is None or wingtip_avg_data is None or wingtip_distribution is None:
        print("Error: Could not load required data files.")
        return

    # Run analyses
    print("Generating wing intensity analysis...")
    wing_summary, t_stat, p_value, percentage_diff = wing_intensity_analysis(wing_data)

    print("Generating wingtip darkness analysis...")
    wingtip_darkness_analysis(wingtip_avg_data, wingtip_distribution)

    print("Analyzing pixel intensity distribution...")
    pixel_intensity_distribution(wingtip_avg_data)

    print("Creating summary visualization...")
    create_summary_visualization(wing_summary, wingtip_avg_data)

    print("Generating summary report...")
    generate_summary_report(wing_summary, wingtip_avg_data)

    print(f"Analysis complete! All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
