import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set the style for all plots
plt.style.use('ggplot')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def create_output_dir():
    """Create directory for visualizations"""
    output_dir = Path("Wing_Analysis_Visualizations")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def load_data():
    """Load both datasets"""
    try:
        wing_df = pd.read_csv(r"../../Wing_Intensity_Results_New/wing_intensity_analysis.csv")
        print(f"Loaded wing intensity data: {wing_df.shape[0]} samples")
    except FileNotFoundError:
        print("Wing intensity CSV not found!")
        wing_df = None

    try:
        tip_df = pd.read_csv("../../Wingtip_Darkness_Results_New/wingtip_darkness_analysis.csv")
        print(f"Loaded wingtip darkness data: {tip_df.shape[0]} samples")
    except FileNotFoundError:
        print("Wingtip darkness CSV not found!")
        tip_df = None

    return wing_df, tip_df


def plot_intensity_distribution(wing_df, output_dir):
    """Plot the distribution of wing intensity by species"""
    if wing_df is None:
        return

    plt.figure(figsize=(14, 8))

    # Create violin plot with boxplot inside
    ax = sns.violinplot(x="species", y="mean_intensity", data=wing_df, inner="box", cut=0)

    # Add individual points
    sns.stripplot(x="species", y="mean_intensity", data=wing_df,
                  size=4, color=".3", alpha=0.4)

    # Add statistics
    species_names = wing_df['species'].unique()
    for i, species in enumerate(species_names):
        species_data = wing_df[wing_df['species'] == species]['mean_intensity']
        mean_val = species_data.mean()
        median_val = species_data.median()
        std_val = species_data.std()

        # Add text with statistics
        plt.text(i, wing_df['mean_intensity'].min() - 5,
                 f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}",
                 ha='center', fontsize=10)

    plt.title("Wing Intensity Distribution by Species", fontsize=16)
    plt.ylabel("Mean Wing Intensity", fontsize=14)
    plt.xlabel("Species", fontsize=14)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / "wing_intensity_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved wing intensity distribution plot")


def plot_intensity_histograms(wing_df, output_dir):
    """Plot histograms of wing intensity by species"""
    if wing_df is None:
        return

    plt.figure(figsize=(14, 8))

    species_names = wing_df['species'].unique()
    colors = sns.color_palette("colorblind", len(species_names))

    for i, species in enumerate(species_names):
        species_data = wing_df[wing_df['species'] == species]['mean_intensity']
        plt.hist(species_data, alpha=0.7, bins=20, color=colors[i],
                 label=f"{species} (n={len(species_data)})")

        # Add vertical line for mean
        plt.axvline(species_data.mean(), color=colors[i], linestyle='dashed',
                    linewidth=2, label=f"{species} mean: {species_data.mean():.2f}")

    plt.title("Histogram of Wing Intensity by Species", fontsize=16)
    plt.xlabel("Mean Wing Intensity", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / "wing_intensity_histograms.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved wing intensity histograms plot")


def plot_wingtip_darkness(tip_df, output_dir):
    """Plot wingtip darkness metrics"""
    if tip_df is None:
        return

    plt.figure(figsize=(14, 8))

    # Create violin plot with box plot inside
    ax = sns.violinplot(x="species", y="percentage_darker", data=tip_df, inner="box", cut=0)

    # Add individual points
    sns.stripplot(x="species", y="percentage_darker", data=tip_df,
                  size=4, color=".3", alpha=0.4)

    # Add statistics
    species_names = tip_df['species'].unique()
    for i, species in enumerate(species_names):
        species_data = tip_df[tip_df['species'] == species]['percentage_darker']
        mean_val = species_data.mean()
        median_val = species_data.median()
        std_val = species_data.std()

        # Add text with statistics
        plt.text(i, 5,
                 f"Mean: {mean_val:.2f}%\nMedian: {median_val:.2f}%\nStd: {std_val:.2f}%",
                 ha='center', fontsize=10)

    plt.title("Percentage of Wingtip Darker than Wing Mean", fontsize=16)
    plt.ylabel("Percentage of Darker Pixels (%)", fontsize=14)
    plt.xlabel("Species", fontsize=14)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / "wingtip_darkness_percentage.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved wingtip darkness percentage plot")


def plot_wing_vs_wingtip(tip_df, output_dir):
    """Plot wing vs wingtip intensity"""
    if tip_df is None:
        return

    plt.figure(figsize=(14, 8))

    # Scatter plot with regression line for each species
    species_names = tip_df['species'].unique()
    colors = sns.color_palette("colorblind", len(species_names))
    markers = ['o', 's']

    for i, species in enumerate(species_names):
        species_data = tip_df[tip_df['species'] == species]

        plt.scatter(species_data['mean_wing_intensity'],
                    species_data['mean_wingtip_intensity'],
                    alpha=0.7, s=50, color=colors[i], marker=markers[i],
                    label=f"{species} (n={len(species_data)})")

        # Add regression line
        sns.regplot(x='mean_wing_intensity', y='mean_wingtip_intensity',
                    data=species_data, scatter=False,
                    line_kws={'color': colors[i], 'linestyle': '--'})

    # Add diagonal line (y=x)
    min_val = min(tip_df['mean_wing_intensity'].min(), tip_df['mean_wingtip_intensity'].min())
    max_val = max(tip_df['mean_wing_intensity'].max(), tip_df['mean_wingtip_intensity'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3,
             label='Equal intensity (y=x)')

    plt.title("Wing vs Wingtip Intensity by Species", fontsize=16)
    plt.xlabel("Mean Wing Intensity", fontsize=14)
    plt.ylabel("Mean Wingtip Intensity", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / "wing_vs_wingtip_intensity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved wing vs wingtip intensity plot")


def plot_intensity_ranges(wing_df, output_dir):
    """Plot intensity range distribution by species"""
    if wing_df is None:
        return

    # Extract intensity range columns
    range_cols = [col for col in wing_df.columns if col.startswith('pct_') and col.count('_') == 2]

    if not range_cols:
        print("No intensity range columns found in wing data")
        return

    # Create a new dataframe for plotting
    plot_data = []
    for _, row in wing_df.iterrows():
        for col in range_cols:
            # Extract the range from column name (pct_0_25 -> (0,25))
            range_parts = col.split('_')[1:]
            range_label = f"{range_parts[0]}-{range_parts[1]}"

            plot_data.append({
                'species': row['species'],
                'intensity_range': range_label,
                'percentage': row[col]
            })

    plot_df = pd.DataFrame(plot_data)

    # Convert range labels to proper order for plotting
    def range_sorter(range_str):
        start = int(range_str.split('-')[0])
        return start

    unique_ranges = sorted(plot_df['intensity_range'].unique(), key=range_sorter)

    plt.figure(figsize=(16, 10))

    # Create grouped bar chart
    ax = sns.barplot(x='intensity_range', y='percentage', hue='species', data=plot_df,
                     order=unique_ranges)

    plt.title("Intensity Range Distribution by Species", fontsize=16)
    plt.xlabel("Intensity Range", fontsize=14)
    plt.ylabel("Percentage of Wing Pixels (%)", fontsize=14)
    plt.legend(title="Species")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / "intensity_range_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved intensity range distribution plot")


def plot_feature_boxplots(wing_df, tip_df, output_dir):
    """Create a grid of boxplots for multiple features"""
    features_wing = ['mean_intensity', 'std_intensity', 'skewness', 'kurtosis']
    features_tip = ['percentage_darker', 'mean_darker_wingtip_intensity']

    # Plot wing features
    if wing_df is not None and len(wing_df) > 0:
        plt.figure(figsize=(16, 12))

        for i, feature in enumerate(features_wing):
            plt.subplot(2, 2, i + 1)

            sns.boxplot(x='species', y=feature, data=wing_df)
            sns.stripplot(x='species', y=feature, data=wing_df,
                          size=4, color=".3", alpha=0.4)

            plt.title(f"{feature.replace('_', ' ').title()}")
            if i >= 2:  # Only add x-label for bottom plots
                plt.xlabel("Species")
            else:
                plt.xlabel("")

        plt.tight_layout()
        plt.savefig(output_dir / "wing_feature_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved wing feature boxplots")

    # Plot wingtip features
    if tip_df is not None and len(tip_df) > 0:
        # Get threshold columns
        threshold_cols = [col for col in tip_df.columns if col.startswith('pct_diff_gt_')]
        features_thresholds = threshold_cols + features_tip

        plt.figure(figsize=(18, 12))

        for i, feature in enumerate(features_thresholds):
            plt.subplot(3, 3, i + 1) if i < 9 else plt.subplot(3, 3, 9)

            sns.boxplot(x='species', y=feature, data=tip_df)
            sns.stripplot(x='species', y=feature, data=tip_df,
                          size=4, color=".3", alpha=0.4)

            # Format the title
            if feature.startswith('pct_diff_gt_'):
                threshold = feature.split('_')[-1]
                title = f"% Pixels > {threshold} Darker than Wing"
            else:
                title = feature.replace('_', ' ').title()

            plt.title(title)
            if i >= 6:  # Only add x-label for bottom plots
                plt.xlabel("Species")
            else:
                plt.xlabel("")

        plt.tight_layout()
        plt.savefig(output_dir / "wingtip_feature_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved wingtip feature boxplots")


def create_statistical_summary(wing_df, tip_df, output_dir):
    """Create statistical summary table with p-values"""
    summary_data = []

    if wing_df is not None and len(wing_df) > 0:
        species_names = wing_df['species'].unique()
        if len(species_names) == 2:
            wing_features = ['mean_intensity', 'std_intensity', 'skewness', 'kurtosis']

            for feature in wing_features:
                species1_data = wing_df[wing_df['species'] == species_names[0]][feature]
                species2_data = wing_df[wing_df['species'] == species_names[1]][feature]

                # Calculate t-test p-value
                from scipy.stats import ttest_ind
                t_stat, p_val = ttest_ind(species1_data, species2_data, equal_var=False)

                summary_data.append({
                    'Feature': feature,
                    'Source': 'Wing',
                    f'{species_names[0]} Mean': species1_data.mean(),
                    f'{species_names[0]} Std': species1_data.std(),
                    f'{species_names[1]} Mean': species2_data.mean(),
                    f'{species_names[1]} Std': species2_data.std(),
                    'p-value': p_val,
                    'Significant': p_val < 0.05
                })

    if tip_df is not None and len(tip_df) > 0:
        species_names = tip_df['species'].unique()
        if len(species_names) == 2:
            tip_features = ['percentage_darker', 'mean_wingtip_intensity',
                            'mean_darker_wingtip_intensity']
            threshold_cols = [col for col in tip_df.columns if col.startswith('pct_diff_gt_')]

            for feature in tip_features + threshold_cols:
                species1_data = tip_df[tip_df['species'] == species_names[0]][feature]
                species2_data = tip_df[tip_df['species'] == species_names[1]][feature]

                # Calculate t-test p-value
                from scipy.stats import ttest_ind
                t_stat, p_val = ttest_ind(species1_data, species2_data, equal_var=False)

                # Format feature name
                display_name = feature
                if feature.startswith('pct_diff_gt_'):
                    threshold = feature.split('_')[-1]
                    display_name = f"% Pixels > {threshold} Darker than Wing"

                summary_data.append({
                    'Feature': display_name,
                    'Source': 'Wingtip',
                    f'{species_names[0]} Mean': species1_data.mean(),
                    f'{species_names[0]} Std': species1_data.std(),
                    f'{species_names[1]} Mean': species2_data.mean(),
                    f'{species_names[1]} Std': species2_data.std(),
                    'p-value': p_val,
                    'Significant': p_val < 0.05
                })

    # Create summary dataframe
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(by=['Significant', 'p-value'],
                                            ascending=[False, True])

        # Save to CSV
        summary_df.to_csv(output_dir / "statistical_summary.csv", index=False)
        print("Saved statistical summary to CSV")

        # Create a heatmap of p-values for significant features
        if len(summary_df) > 0:
            significant_df = summary_df[summary_df['Significant'] == True].copy()

            if len(significant_df) > 0:
                # Prepare data for heatmap
                significant_df['-log10(p)'] = -np.log10(significant_df['p-value'])

                plt.figure(figsize=(12, max(6, len(significant_df) * 0.4)))

                # Create horizontal bar chart
                features = significant_df['Feature'].tolist()
                sources = significant_df['Source'].tolist()
                labels = [f"{f} ({s})" for f, s in zip(features, sources)]

                # Sort by p-value
                idx = np.argsort(significant_df['p-value'].values)
                labels = [labels[i] for i in idx]
                p_values = significant_df['p-value'].values[idx]
                log_p = significant_df['-log10(p)'].values[idx]

                # Create bar chart
                bars = plt.barh(range(len(labels)), log_p, color='skyblue')

                # Add p-values as text
                for i, (bar, p) in enumerate(zip(bars, p_values)):
                    plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                             f"p = {p:.6f}", va='center')

                plt.yticks(range(len(labels)), labels)
                plt.xlabel('-log10(p-value)')
                plt.title('Significant Features (p < 0.05)')
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()

                plt.savefig(output_dir / "significant_features.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("Saved significant features plot")


def main():
    output_dir = create_output_dir()
    wing_df, tip_df = load_data()

    # Generate visualizations
    plot_intensity_distribution(wing_df, output_dir)
    plot_intensity_histograms(wing_df, output_dir)
    plot_intensity_ranges(wing_df, output_dir)
    plot_wingtip_darkness(tip_df, output_dir)
    plot_wing_vs_wingtip(tip_df, output_dir)
    plot_feature_boxplots(wing_df, tip_df, output_dir)
    create_statistical_summary(wing_df, tip_df, output_dir)

    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()