import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Set plotting style
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.figsize'] = (16, 10)


def process_wing_intensity_data(file_path):
    """
    Process the wing intensity data and generate statistics and visualizations.
    """
    print(f"Loading wing intensity data from: {file_path}")

    data = pd.read_csv(file_path)

    # Calculate summary statistics grouped by species
    summary_stats = data.groupby('species').agg({
        'mean_intensity': ['mean', 'std', 'min', 'max', 'median'],
        'std_intensity': ['mean', 'std', 'min', 'max'],
        'skewness': ['mean', 'std'],
        'kurtosis': ['mean', 'std'],
        'pixel_count': ['sum', 'mean', 'count']
    })

    # Perform t-test between species for mean intensity
    slaty_data = data[data['species'] == 'Slaty_Backed_Gull']['mean_intensity']
    glaucous_data = data[data['species'] == 'Glaucous_Winged_Gull']['mean_intensity']

    t_stat, p_value = ttest_ind(slaty_data, glaucous_data, equal_var=False)

    # Calculate percentage difference
    slaty_mean = summary_stats[('mean_intensity', 'mean')]['Slaty_Backed_Gull']
    glaucous_mean = summary_stats[('mean_intensity', 'mean')]['Glaucous_Winged_Gull']
    percentage_diff = ((glaucous_mean - slaty_mean) / slaty_mean) * 100

    # Print summary statistics
    print("\nWing Intensity Summary Statistics:")
    print(
        f"  Slaty-backed Gull: {slaty_mean:.2f} ± {summary_stats[('mean_intensity', 'std')]['Slaty_Backed_Gull']:.2f}")
    print(
        f"  Glaucous-winged Gull: {glaucous_mean:.2f} ± {summary_stats[('mean_intensity', 'std')]['Glaucous_Winged_Gull']:.2f}")
    print(f"  Difference: {percentage_diff:.2f}% (Glaucous-winged wings are brighter)")
    print(f"  T-test: t={t_stat:.4f}, p={p_value:.8f}")

    # Create figure for wing intensity visualizations
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # 1. Histogram of Mean Intensity by Species
    ax1 = fig.add_subplot(gs[0, 0:2])
    sns.histplot(data=data, x='mean_intensity', hue='species', kde=True, bins=20, alpha=0.6, ax=ax1)
    ax1.set_title('Distribution of Wing Mean Intensity by Species', fontsize=14)
    ax1.set_xlabel('Wing Intensity (0-255 scale)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)

    # 2. Box Plot of Mean Intensity by Species
    ax2 = fig.add_subplot(gs[0, 2])
    sns.boxplot(data=data, x='species', y='mean_intensity', ax=ax2)
    ax2.set_title('Box Plot of Wing Mean Intensity by Species', fontsize=14)
    ax2.set_ylabel('Wing Intensity (0-255 scale)', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25)

    # 3. Bar Plot of Mean Intensities by Species
    ax3 = fig.add_subplot(gs[1, 0])
    species_means = data.groupby('species')['mean_intensity'].mean().reset_index()
    species_std = data.groupby('species')['mean_intensity'].std().reset_index()

    bars = sns.barplot(data=species_means, x='species', y='mean_intensity', ax=ax3,
                       yerr=species_std['mean_intensity'], capsize=10)
    ax3.set_title('Mean Wing Intensity by Species', fontsize=14)
    ax3.set_ylabel('Mean Intensity (0-255 scale)', fontsize=12)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=25)

    # Add value labels to bars
    for i, p in enumerate(bars.patches):
        ax3.annotate(f'{p.get_height():.1f}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=12)

    # 4. Standard Deviation Plot
    ax4 = fig.add_subplot(gs[1, 1])
    species_std = data.groupby('species')['std_intensity'].mean().reset_index()
    species_std_err = data.groupby('species')['std_intensity'].std().reset_index()

    bars = sns.barplot(data=species_std, x='species', y='std_intensity', ax=ax4,
                       yerr=species_std_err['std_intensity'], capsize=10)
    ax4.set_title('Mean Standard Deviation of Wing Intensity', fontsize=14)
    ax4.set_ylabel('Standard Deviation', fontsize=12)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=25)

    # Add value labels to bars
    for i, p in enumerate(bars.patches):
        ax4.annotate(f'{p.get_height():.1f}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=12)

    # 5. Scatter plot of mean vs std intensity
    ax5 = fig.add_subplot(gs[1, 2])
    sns.scatterplot(data=data, x='mean_intensity', y='std_intensity', hue='species', alpha=0.7, ax=ax5)
    ax5.set_title('Mean vs. Standard Deviation of Wing Intensity', fontsize=14)
    ax5.set_xlabel('Mean Intensity (0-255 scale)', fontsize=12)
    ax5.set_ylabel('Standard Deviation', fontsize=12)

    # Add T-test result and stats as text
    significance = "Statistically Significant" if p_value < 0.05 else "Not Statistically Significant"
    stats_text = (f"T-test Results: t = {t_stat:.4f}, p = {p_value:.8f}\n"
                  f"Result: {significance}\n"
                  f"Slaty-backed: {slaty_mean:.2f} ± {summary_stats[('mean_intensity', 'std')]['Slaty_Backed_Gull']:.2f}\n"
                  f"Glaucous-winged: {glaucous_mean:.2f} ± {summary_stats[('mean_intensity', 'std')]['Glaucous_Winged_Gull']:.2f}\n"
                  f"Difference: {percentage_diff:.2f}%")

    plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=12,
                bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle('Wing Intensity Comparison: Slaty-backed Gull vs Glaucous-winged Gull', fontsize=18, y=0.98)

    # Save the figure
    output_file = "Wing_Intensity_Analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Wing intensity visualization saved as: {output_file}")

    return data, summary_stats


def process_wingtip_darkness_data(file_path):
    """
    Process the wingtip darkness data and generate statistics and visualizations.
    """
    print(f"Loading wingtip darkness data from: {file_path}")

    data = pd.read_csv(file_path)

    # Calculate summary statistics for wingtip darkness
    metrics = ['percentage_darker', 'mean_wing_intensity',
               'mean_wingtip_intensity', 'mean_darker_wingtip_intensity']

    # Comprehensive statistics calculation
    summary_stats = {}
    for metric in metrics:
        summary_stats[metric] = data.groupby('species')[metric].agg(['mean', 'std', 'min', 'max', 'median', 'count'])

    # Perform t-tests for all metrics
    t_test_results = {}
    for metric in metrics:
        slaty_data = data[data['species'] == 'Slaty_Backed_Gull'][metric]
        glaucous_data = data[data['species'] == 'Glaucous_Winged_Gull'][metric]

        t_stat, p_value = ttest_ind(slaty_data, glaucous_data, equal_var=False)

        t_test_results[metric] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    # Print summary information
    print("\nWingtip Darkness Summary Statistics:")
    for metric in metrics:
        stat = summary_stats[metric]
        slaty_mean = stat.loc['Slaty_Backed_Gull', 'mean']
        slaty_std = stat.loc['Slaty_Backed_Gull', 'std']
        glaucous_mean = stat.loc['Glaucous_Winged_Gull', 'mean']
        glaucous_std = stat.loc['Glaucous_Winged_Gull', 'std']

        print(f"\n  {metric.replace('_', ' ').title()}:")
        print(f"    Slaty-backed Gull: {slaty_mean:.2f} ± {slaty_std:.2f}")
        print(f"    Glaucous-winged Gull: {glaucous_mean:.2f} ± {glaucous_std:.2f}")

        if metric == 'percentage_darker':
            diff = slaty_mean - glaucous_mean
            print(
                f"    Difference: {diff:.2f}% (Slaty-backed has {abs(diff):.2f}% {'more' if diff > 0 else 'fewer'} darker pixels)")

        t_stat = t_test_results[metric]['t_statistic']
        p_value = t_test_results[metric]['p_value']
        significance = "Significant" if p_value < 0.05 else "Not significant"
        print(f"    T-test: t={t_stat:.4f}, p={p_value:.8f} ({significance})")

    # Create visualizations for wingtip darkness data
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # 1. Percentage of wingtip darker than wing mean
    ax1 = fig.add_subplot(gs[0, 0])
    species_means = data.groupby('species')['percentage_darker'].mean().reset_index()
    species_std = data.groupby('species')['percentage_darker'].std().reset_index()

    bars = sns.barplot(x='species', y='percentage_darker', data=species_means,
                       yerr=species_std['percentage_darker'], capsize=10, ax=ax1)
    ax1.set_title(f'Percentage of Wingtip Darker than Wing\n(p = {t_test_results["percentage_darker"]["p_value"]:.5f})',
                  fontsize=14)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25)

    # Add value labels
    for i, p in enumerate(bars.patches):
        ax1.annotate(f'{p.get_height():.1f}%',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=12)

    # 2. Compare wing, wingtip and darker wingtip intensities
    ax2 = fig.add_subplot(gs[0, 1])

    # Prepare data for grouped bar chart
    intensity_data = []
    for species in ['Slaty_Backed_Gull', 'Glaucous_Winged_Gull']:
        species_data = data[data['species'] == species]
        wing_mean = species_data['mean_wing_intensity'].mean()
        wing_std = species_data['mean_wing_intensity'].std()
        wingtip_mean = species_data['mean_wingtip_intensity'].mean()
        wingtip_std = species_data['mean_wingtip_intensity'].std()
        darker_mean = species_data['mean_darker_wingtip_intensity'].mean()
        darker_std = species_data['mean_darker_wingtip_intensity'].std()

        intensity_data.append({
            'species': species,
            'region': 'Wing',
            'intensity': wing_mean,
            'std': wing_std
        })
        intensity_data.append({
            'species': species,
            'region': 'Wingtip',
            'intensity': wingtip_mean,
            'std': wingtip_std
        })
        intensity_data.append({
            'species': species,
            'region': 'Darker Wingtip',
            'intensity': darker_mean,
            'std': darker_std
        })

    intensity_df = pd.DataFrame(intensity_data)

    # Plot grouped bar chart
    bars = sns.barplot(x='species', y='intensity', hue='region', data=intensity_df, ax=ax2)
    ax2.set_title('Mean Intensity Comparison by Region', fontsize=14)
    ax2.set_ylabel('Intensity (0-255 scale)', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25)

    # 3. Distribution of percentage darker
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(data=data, x='percentage_darker', hue='species', kde=True, bins=20, alpha=0.6, ax=ax3)
    ax3.set_title('Distribution of Percentage Darker', fontsize=14)
    ax3.set_xlabel('Percentage Darker (%)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)

    # 4. Relationship between wing and wingtip intensity
    ax4 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(data=data, x='mean_wing_intensity', y='mean_wingtip_intensity',
                    hue='species', size='percentage_darker', sizes=(20, 200), alpha=0.7, ax=ax4)
    ax4.set_title('Wing vs Wingtip Intensity', fontsize=14)
    ax4.set_xlabel('Wing Intensity (0-255 scale)', fontsize=12)
    ax4.set_ylabel('Wingtip Intensity (0-255 scale)', fontsize=12)

    # Add a diagonal line representing equal intensity
    lims = [
        np.min([ax4.get_xlim(), ax4.get_ylim()]),  # min of both axes
        np.max([ax4.get_xlim(), ax4.get_ylim()]),  # max of both axes
    ]
    ax4.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

    # Add legend for diagonal
    handles, labels = ax4.get_legend_handles_labels()
    handles.append(Line2D([0], [0], linestyle='--', color='k', alpha=0.5))
    labels.append('Equal Intensity')
    ax4.legend(handles=handles, labels=labels, title='Species & Darkness %', loc='best')

    # Add overall statistics
    plt.figtext(0.5, 0.01,
                f"Percentage Darker: t={t_test_results['percentage_darker']['t_statistic']:.4f}, "
                f"p={t_test_results['percentage_darker']['p_value']:.8f} "
                f"({'Significant' if t_test_results['percentage_darker']['significant'] else 'Not significant'})",
                ha="center", fontsize=12,
                bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle('Wingtip Darkness Analysis: Slaty-backed Gull vs Glaucous-winged Gull', fontsize=18, y=0.98)

    # Save the figure
    output_file = "Wingtip_Darkness_Analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Wingtip darkness visualization saved as: {output_file}")

    return data, summary_stats, t_test_results


def generate_combined_report(wing_data, wing_stats, darkness_data, darkness_stats, darkness_t_tests):
    """
    Generate a comprehensive report combining wing intensity and wingtip darkness analysis.
    """
    # Create a summary figure with key findings
    plt.figure(figsize=(12, 10))

    # 1. Bar plot comparing key metrics between species
    plt.subplot(2, 1, 1)

    # Prepare data for the comparison
    comparison_data = []

    # Add wing intensity data
    slaty_intensity = wing_stats[('mean_intensity', 'mean')]['Slaty_Backed_Gull']
    glaucous_intensity = wing_stats[('mean_intensity', 'mean')]['Glaucous_Winged_Gull']

    comparison_data.append({
        'metric': 'Wing Intensity',
        'Slaty_Backed_Gull': slaty_intensity,
        'Glaucous_Winged_Gull': glaucous_intensity
    })

    # Add wingtip darkness data
    slaty_darkness = darkness_data[darkness_data['species'] == 'Slaty_Backed_Gull']['percentage_darker'].mean()
    glaucous_darkness = darkness_data[darkness_data['species'] == 'Glaucous_Winged_Gull']['percentage_darker'].mean()

    comparison_data.append({
        'metric': 'Wingtip Darkness %',
        'Slaty_Backed_Gull': slaty_darkness,
        'Glaucous_Winged_Gull': glaucous_darkness
    })

    # Add contrast ratio (darker wingtip vs wing)
    slaty_contrast = (darkness_data[darkness_data['species'] == 'Slaty_Backed_Gull']['mean_wing_intensity'] /
                      darkness_data[darkness_data['species'] == 'Slaty_Backed_Gull'][
                          'mean_darker_wingtip_intensity']).mean()

    glaucous_contrast = (darkness_data[darkness_data['species'] == 'Glaucous_Winged_Gull']['mean_wing_intensity'] /
                         darkness_data[darkness_data['species'] == 'Glaucous_Winged_Gull'][
                             'mean_darker_wingtip_intensity']).mean()

    comparison_data.append({
        'metric': 'Wing/Darker Wingtip Ratio',
        'Slaty_Backed_Gull': slaty_contrast,
        'Glaucous_Winged_Gull': glaucous_contrast
    })

    # Convert to DataFrame and reshape for plotting
    comp_df = pd.DataFrame(comparison_data)
    plot_df = pd.melt(comp_df, id_vars=['metric'],
                      value_vars=['Slaty_Backed_Gull', 'Glaucous_Winged_Gull'],
                      var_name='species', value_name='value')

    sns.barplot(x='metric', y='value', hue='species', data=plot_df)
    plt.title('Key Metrics Comparison Between Species', fontsize=14)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=15)

    # 2. Text summary of findings
    plt.subplot(2, 1, 2)
    plt.axis('off')

    wing_t_stat = ttest_ind(
        wing_data[wing_data['species'] == 'Slaty_Backed_Gull']['mean_intensity'],
        wing_data[wing_data['species'] == 'Glaucous_Winged_Gull']['mean_intensity'],
        equal_var=False
    )

    summary_text = """
    ## Summary of Findings

    ### Wing Intensity
    * Slaty-backed Gull wings are SIGNIFICANTLY DARKER than Glaucous-winged Gull wings
    * Mean wing intensity difference is statistically significant (p < 0.00001)
    * Glaucous-winged Gull wings are approximately 2× brighter

    ### Wingtip Darkness
    * Slaty-backed Gull wingtips have MORE dark pixels relative to their wing intensity
    * A larger percentage of the Slaty-backed Gull wingtip area is darker than its wing
    * The contrast between wing and darker wingtip areas is greater in Slaty-backed Gulls

    ### Key Distinguishing Features
    1. Wing intensity is the most reliable differentiator between species
    2. Wingtip darkness percentage provides additional distinguishing information
    3. The combination of both features enables more accurate species identification
    """

    plt.text(0.5, 0.95, 'Summary of Findings', horizontalalignment='center',
             fontsize=16, fontweight='bold')

    plt.text(0.1, 0.85, f"Wing Intensity:", fontsize=14, fontweight='bold')
    plt.text(0.15, 0.80,
             f"• Slaty-backed Gull: {slaty_intensity:.1f} ± {wing_stats[('mean_intensity', 'std')]['Slaty_Backed_Gull']:.1f}",
             fontsize=12)
    plt.text(0.15, 0.75,
             f"• Glaucous-winged Gull: {glaucous_intensity:.1f} ± {wing_stats[('mean_intensity', 'std')]['Glaucous_Winged_Gull']:.1f}",
             fontsize=12)
    plt.text(0.15, 0.70,
             f"• Difference: {((glaucous_intensity - slaty_intensity) / slaty_intensity * 100):.1f}% (p={wing_t_stat[1]:.8f})",
             fontsize=12)

    plt.text(0.1, 0.60, f"Wingtip Darkness:", fontsize=14, fontweight='bold')
    plt.text(0.15, 0.55,
             f"• Slaty-backed Gull: {slaty_darkness:.1f}% ± {darkness_data[darkness_data['species'] == 'Slaty_Backed_Gull']['percentage_darker'].std():.1f}%",
             fontsize=12)
    plt.text(0.15, 0.50,
             f"• Glaucous-winged Gull: {glaucous_darkness:.1f}% ± {darkness_data[darkness_data['species'] == 'Glaucous_Winged_Gull']['percentage_darker'].std():.1f}%",
             fontsize=12)
    plt.text(0.15, 0.45,
             f"• Difference: {slaty_darkness - glaucous_darkness:.1f}% (p={darkness_t_tests['percentage_darker']['p_value']:.8f})",
             fontsize=12)

    plt.text(0.1, 0.35, f"Contrast Ratio (Wing/Darker Wingtip):", fontsize=14, fontweight='bold')
    plt.text(0.15, 0.30, f"• Slaty-backed Gull: {slaty_contrast:.2f}", fontsize=12)
    plt.text(0.15, 0.25, f"• Glaucous-winged Gull: {glaucous_contrast:.2f}", fontsize=12)

    plt.text(0.1, 0.15, f"Conclusion:", fontsize=14, fontweight='bold')
    plt.text(0.15, 0.10, "Slaty-backed Gulls have darker wings overall and more contrast in wingtips", fontsize=12)
    plt.text(0.15, 0.05, "Both features can be used together for improved species identification", fontsize=12)

    plt.tight_layout()

    # Save the report
    output_file = "Gull_Wing_Analysis_Summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Summary report saved as: {output_file}")


def main():
    """
    Main function to run the complete analysis.
    """
    # Create output directories if needed
    os.makedirs("Analysis_Results", exist_ok=True)

    # Define input files
    wing_intensity_file = "wing_intensity_analysis.csv"
    wingtip_darkness_file = "Darkness_Analysis_Results/wingtip_darkness_analysis.csv"

    # Check if input files exist
    if not os.path.exists(wing_intensity_file):
        print(f"Error: Wing intensity file '{wing_intensity_file}' not found.")
        return

    if not os.path.exists(wingtip_darkness_file):
        print(f"Error: Wingtip darkness file '{wingtip_darkness_file}' not found.")
        return

    # Process wing intensity data
    wing_data, wing_stats = process_wing_intensity_data(wing_intensity_file)

    # Process wingtip darkness data
    darkness_data, darkness_stats, darkness_t_tests = process_wingtip_darkness_data(wingtip_darkness_file)

    # Generate combined report
    generate_combined_report(wing_data, wing_stats, darkness_data, darkness_stats, darkness_t_tests)

    print("\nAnalysis complete! Output files saved to current directory.")


if __name__ == "__main__":
    main()
