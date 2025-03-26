import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


def analyze_wingtip_darkness_data():
    """
    Analyze the wingtip darkness data that's already stored in Darkness_Analysis_Results
    and generate comprehensive visualizations and statistical comparisons.
    """
    # Ensure results directory exists
    results_dir = "../Darkness_Analysis_Results"

    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return

    # Load the existing analysis results
    darkness_file = os.path.join(results_dir, "wingtip_darkness_analysis.csv")

    if not os.path.exists(darkness_file):
        print(f"Error: File '{darkness_file}' not found.")
        return

    print(f"Loading data from {darkness_file}...")
    df = pd.read_csv(darkness_file)

    # Calculate standard deviations for the key metrics
    metrics = ['percentage_darker', 'mean_wing_intensity',
               'mean_wingtip_intensity', 'mean_darker_wingtip_intensity']

    # Summarize data by species
    species_summary = df.groupby('species')[metrics].agg(['mean', 'std', 'min', 'max']).reset_index()

    # Save the comprehensive summary
    summary_file = os.path.join(results_dir, "wingtip_darkness_comprehensive_summary.csv")
    species_summary.to_csv(summary_file)

    # Conduct t-tests for statistical comparisons
    slaty_data = df[df['species'] == 'Slaty_Backed_Gull']
    glaucous_data = df[df['species'] == 'Glaucous_Winged_Gull']

    t_test_results = {}

    for metric in metrics:
        t_stat, p_val = ttest_ind(
            slaty_data[metric],
            glaucous_data[metric],
            equal_var=False  # Welch's t-test for unequal variances
        )

        t_test_results[metric] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }

    # Save t-test results
    t_test_df = pd.DataFrame([
        {
            'metric': metric,
            't_statistic': results['t_statistic'],
            'p_value': results['p_value'],
            'significant': results['significant']
        }
        for metric, results in t_test_results.items()
    ])

    t_test_file = os.path.join(results_dir, "wingtip_darkness_t_tests.csv")
    t_test_df.to_csv(t_test_file, index=False)

    # Create enhanced visualizations
    create_enhanced_visualizations(df, results_dir, t_test_results)

    # Generate a comprehensive report
    generate_report(df, species_summary, t_test_results, results_dir)

    print(f"Analysis complete. Results saved to {results_dir}:")
    print(f"- Comprehensive summary: {summary_file}")
    print(f"- Statistical tests: {t_test_file}")
    print(f"- Enhanced visualizations saved to {results_dir}")


def create_enhanced_visualizations(df, results_dir, t_test_results):
    """
    Create enhanced visualizations for the wingtip darkness analysis
    """
    # Set a consistent style
    sns.set(style="whitegrid", font_scale=1.1)

    # Calculate species averages for barplots
    species_avg = df.groupby('species')[
        ['percentage_darker', 'mean_wing_intensity',
         'mean_wingtip_intensity', 'mean_darker_wingtip_intensity']
    ].mean().reset_index()

    species_std = df.groupby('species')[
        ['percentage_darker', 'mean_wing_intensity',
         'mean_wingtip_intensity', 'mean_darker_wingtip_intensity']
    ].std().reset_index()

    # Figure 1: Comparative Bar Charts
    plt.figure(figsize=(14, 10))

    # Plot 1: Percentage Darker
    plt.subplot(2, 2, 1)
    ax = sns.barplot(x='species', y='percentage_darker', data=species_avg,
                     yerr=species_std['percentage_darker'], capsize=10)
    plt.title(f'Percentage of Wingtip Darker than Wing\n(p = {t_test_results["percentage_darker"]["p_value"]:.5f})')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=25)
    # Add value labels to bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}%',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')

    # Plot 2: Wing Intensity
    plt.subplot(2, 2, 2)
    ax = sns.barplot(x='species', y='mean_wing_intensity', data=species_avg,
                     yerr=species_std['mean_wing_intensity'], capsize=10)
    plt.title(f'Mean Wing Intensity\n(p = {t_test_results["mean_wing_intensity"]["p_value"]:.5f})')
    plt.ylabel('Intensity (0-255)')
    plt.xticks(rotation=25)
    # Add value labels to bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')

    # Plot 3: Wingtip Intensity
    plt.subplot(2, 2, 3)
    ax = sns.barplot(x='species', y='mean_wingtip_intensity', data=species_avg,
                     yerr=species_std['mean_wingtip_intensity'], capsize=10)
    plt.title(f'Mean Wingtip Intensity\n(p = {t_test_results["mean_wingtip_intensity"]["p_value"]:.5f})')
    plt.ylabel('Intensity (0-255)')
    plt.xticks(rotation=25)
    # Add value labels to bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')

    # Plot 4: Darker Wingtip Intensity
    plt.subplot(2, 2, 4)
    ax = sns.barplot(x='species', y='mean_darker_wingtip_intensity', data=species_avg,
                     yerr=species_std['mean_darker_wingtip_intensity'], capsize=10)
    plt.title(f'Mean Darker Wingtip Intensity\n(p = {t_test_results["mean_darker_wingtip_intensity"]["p_value"]:.5f})')
    plt.ylabel('Intensity (0-255)')
    plt.xticks(rotation=25)
    # Add value labels to bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wingtip_darkness_barplots.png'), dpi=300)

    # Figure 2: Distribution Comparisons
    plt.figure(figsize=(14, 10))

    # Plot 1: Distribution of Percentage Darker
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='percentage_darker', hue='species', kde=True, bins=20, alpha=0.6)
    plt.title('Distribution of Percentage Darker')
    plt.xlabel('Percentage Darker (%)')

    # Plot 2: Distribution of Wing Intensity
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='mean_wing_intensity', hue='species', kde=True, bins=20, alpha=0.6)
    plt.title('Distribution of Wing Intensity')
    plt.xlabel('Wing Intensity (0-255)')

    # Plot 3: Distribution of Wingtip Intensity
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='mean_wingtip_intensity', hue='species', kde=True, bins=20, alpha=0.6)
    plt.title('Distribution of Wingtip Intensity')
    plt.xlabel('Wingtip Intensity (0-255)')

    # Plot 4: Distribution of Darker Wingtip Intensity
    plt.subplot(2, 2, 4)
    sns.histplot(data=df, x='mean_darker_wingtip_intensity', hue='species', kde=True, bins=20, alpha=0.6)
    plt.title('Distribution of Darker Wingtip Intensity')
    plt.xlabel('Darker Wingtip Intensity (0-255)')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wingtip_darkness_distributions.png'), dpi=300)

    # Figure 3: Box Plots
    plt.figure(figsize=(14, 10))

    # Plot 1: Box Plot of Percentage Darker
    plt.subplot(2, 2, 1)
    sns.boxplot(x='species', y='percentage_darker', data=df)
    plt.title('Percentage of Wingtip Darker than Wing')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=25)

    # Plot 2: Box Plot of Wing Intensity
    plt.subplot(2, 2, 2)
    sns.boxplot(x='species', y='mean_wing_intensity', data=df)
    plt.title('Wing Intensity')
    plt.ylabel('Intensity (0-255)')
    plt.xticks(rotation=25)

    # Plot 3: Box Plot of Wingtip Intensity
    plt.subplot(2, 2, 3)
    sns.boxplot(x='species', y='mean_wingtip_intensity', data=df)
    plt.title('Wingtip Intensity')
    plt.ylabel('Intensity (0-255)')
    plt.xticks(rotation=25)

    # Plot 4: Box Plot of Darker Wingtip Intensity
    plt.subplot(2, 2, 4)
    sns.boxplot(x='species', y='mean_darker_wingtip_intensity', data=df)
    plt.title('Darker Wingtip Intensity')
    plt.ylabel('Intensity (0-255)')
    plt.xticks(rotation=25)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wingtip_darkness_boxplots.png'), dpi=300)

    # Figure 4: Scatter Plots
    plt.figure(figsize=(14, 10))

    # Plot 1: Wing Intensity vs Percentage Darker
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='mean_wing_intensity', y='percentage_darker', hue='species', alpha=0.7)
    plt.title('Wing Intensity vs Percentage Darker')
    plt.xlabel('Wing Intensity (0-255)')
    plt.ylabel('Percentage Darker (%)')

    # Plot 2: Wingtip Intensity vs Percentage Darker
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='mean_wingtip_intensity', y='percentage_darker', hue='species', alpha=0.7)
    plt.title('Wingtip Intensity vs Percentage Darker')
    plt.xlabel('Wingtip Intensity (0-255)')
    plt.ylabel('Percentage Darker (%)')

    # Plot 3: Wing Intensity vs Wingtip Intensity
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='mean_wing_intensity', y='mean_wingtip_intensity', hue='species', alpha=0.7)
    plt.title('Wing Intensity vs Wingtip Intensity')
    plt.xlabel('Wing Intensity (0-255)')
    plt.ylabel('Wingtip Intensity (0-255)')

    # Plot 4: Wing Intensity vs Darker Wingtip Intensity
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='mean_wing_intensity', y='mean_darker_wingtip_intensity', hue='species', alpha=0.7)
    plt.title('Wing Intensity vs Darker Wingtip Intensity')
    plt.xlabel('Wing Intensity (0-255)')
    plt.ylabel('Darker Wingtip Intensity (0-255)')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wingtip_darkness_scatterplots.png'), dpi=300)

    # Figure 5: Region Intensity Comparison
    plt.figure(figsize=(10, 8))

    # Melt the dataframe for easier plotting
    intensity_data = pd.melt(
        df,
        id_vars=['species'],
        value_vars=['mean_wing_intensity', 'mean_wingtip_intensity', 'mean_darker_wingtip_intensity'],
        var_name='Region',
        value_name='Intensity'
    )

    # Create a boxplot
    sns.boxplot(x='Region', y='Intensity', hue='species', data=intensity_data)
    plt.title('Intensity Comparison by Region and Species')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'wingtip_darkness_region_comparison.png'), dpi=300)


def generate_report(df, species_summary, t_test_results, results_dir):
    """Generate a comprehensive report with statistical insights"""
    # Code for report generation
    # ...


if __name__ == "__main__":
    analyze_wingtip_darkness_data()
