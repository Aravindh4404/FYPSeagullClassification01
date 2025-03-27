import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

# Create output directory
output_dir = "Wingtip_Diff_Analysis"
os.makedirs(output_dir, exist_ok=True)

wing_data = pd.read_csv('../Intensity_Results/wing_intensity_analysis.csv')
wingtip_data = pd.read_csv('../Wingtip_Intensity_Distribution/wingtip_intensity_averages.csv')
detailed_data = pd.read_csv('../Wingtip_Intensity_Distribution/wingtip_intensity_distribution.csv')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12


def visualize_intensity_differences():
    """Create visualizations showing the differences between wing and wingtip intensity"""

    # Figure 1: Visualize the percentage of pixels with differences greater than thresholds
    plt.figure(figsize=(14, 10))

    # Extract difference threshold data
    diff_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    diff_data = []

    for species in wingtip_data['species']:
        for threshold in diff_thresholds:
            col = f'pct_diff_gt_{threshold}'
            if col in wingtip_data.columns:
                value = wingtip_data[wingtip_data['species'] == species][col].values[0]
                diff_data.append({
                    'species': species.replace('_', '-'),
                    'threshold': threshold,
                    'percentage': value
                })

    diff_df = pd.DataFrame(diff_data)

    # Create a bar plot
    ax = sns.barplot(x='threshold', y='percentage', hue='species', data=diff_df)

    # Add labels and title
    plt.title('Percentage of Wingtip Pixels Darker Than Wing by Threshold', fontsize=16)
    plt.xlabel('Intensity Difference Threshold', fontsize=14)
    plt.ylabel('Percentage of Pixels (%)', fontsize=14)

    # Add value labels to the bars
    for i, threshold in enumerate(diff_thresholds):
        for j, species in enumerate(['Slaty-Backed-Gull', 'Glaucous-Winged-Gull']):
            species_data = diff_df[(diff_df['species'] == species) & (diff_df['threshold'] == threshold)]
            if not species_data.empty:
                percentage = species_data['percentage'].values[0]
                x_pos = i - 0.2 + j * 0.4  # Adjust position based on bar width
                plt.text(x_pos, percentage + 1, f'{percentage:.1f}%',
                         ha='center', va='bottom', fontsize=9, rotation=90)

    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust legend
    plt.legend(title='Species', fontsize=12)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wingtip_darkness_thresholds.png'), dpi=300)
    plt.close()

    # Figure 2: Create a visualization showing the pixel darkness distribution
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)

    # 1. Line plot showing the percentage at each threshold
    ax1 = fig.add_subplot(gs[0, :])
    sns.lineplot(data=diff_df, x='threshold', y='percentage', hue='species',
                 marker='o', markersize=10, linewidth=3, ax=ax1)

    # Add labels
    ax1.set_title('Wing-Wingtip Intensity Difference Distribution', fontsize=16)
    ax1.set_xlabel('Intensity Difference Threshold', fontsize=14)
    ax1.set_ylabel('Percentage of Pixels (%)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Highlight the difference at key thresholds
    for threshold in [30, 50, 70]:
        slaty_data = diff_df[(diff_df['species'] == 'Slaty-Backed-Gull') & (diff_df['threshold'] == threshold)]
        glaucous_data = diff_df[(diff_df['species'] == 'Glaucous-Winged-Gull') & (diff_df['threshold'] == threshold)]

        if not slaty_data.empty and not glaucous_data.empty:
            slaty_pct = slaty_data['percentage'].values[0]
            glaucous_pct = glaucous_data['percentage'].values[0]
            diff_pct = slaty_pct - glaucous_pct

            ax1.annotate(f'Diff: {diff_pct:.1f}%',
                         xy=(threshold, (slaty_pct + glaucous_pct) / 2),
                         xytext=(threshold + 5, (slaty_pct + glaucous_pct) / 2 + 5),
                         arrowprops=dict(arrowstyle='->'),
                         fontsize=12)

    # 2. Bar chart showing relative darkness differences at key thresholds
    ax2 = fig.add_subplot(gs[1, 0])

    # Select key thresholds for clearer visualization
    key_thresholds = [30, 50, 70]
    key_data = diff_df[diff_df['threshold'].isin(key_thresholds)]

    sns.barplot(data=key_data, x='threshold', y='percentage', hue='species', ax=ax2)
    ax2.set_title('Significant Darkness Difference Thresholds', fontsize=16)
    ax2.set_xlabel('Intensity Difference Threshold', fontsize=14)
    ax2.set_ylabel('Percentage of Pixels (%)', fontsize=14)

    # Add value labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f%%', fontsize=11)

    # 3. Ratio comparison chart
    ax3 = fig.add_subplot(gs[1, 1])

    # Calculate ratios
    ratio_data = []
    for threshold in diff_thresholds:
        slaty_data = diff_df[(diff_df['species'] == 'Slaty-Backed-Gull') & (diff_df['threshold'] == threshold)]
        glaucous_data = diff_df[(diff_df['species'] == 'Glaucous-Winged-Gull') & (diff_df['threshold'] == threshold)]

        if not slaty_data.empty and not glaucous_data.empty:
            slaty_pct = slaty_data['percentage'].values[0]
            glaucous_pct = glaucous_data['percentage'].values[0]

            # Avoid division by zero
            if glaucous_pct > 0:
                ratio = slaty_pct / glaucous_pct
            else:
                ratio = float('inf') if slaty_pct > 0 else 0

            ratio_data.append({
                'threshold': threshold,
                'ratio': min(ratio, 10)  # Cap for visualization purposes
            })

    ratio_df = pd.DataFrame(ratio_data)

    # Create bar chart for ratio
    sns.barplot(data=ratio_df, x='threshold', y='ratio', color='purple', ax=ax3)
    ax3.set_title('Ratio of Slaty-backed to Glaucous-winged Darkness', fontsize=16)
    ax3.set_xlabel('Intensity Difference Threshold', fontsize=14)
    ax3.set_ylabel('Ratio (Slaty/Glaucous)', fontsize=14)

    # Add reference line at ratio=1
    ax3.axhline(y=1, color='red', linestyle='--')

    # Add value labels
    for i, p in enumerate(ax3.patches):
        if p.get_height() >= 10:
            ax3.text(p.get_x() + p.get_width() / 2., p.get_height() - 0.5, '>10x',
                     ha='center', fontsize=11)
        else:
            ax3.text(p.get_x() + p.get_width() / 2., p.get_height() + 0.2, f'{p.get_height():.1f}x',
                     ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wingtip_darkness_analysis.png'), dpi=300)
    plt.close()

    # Figure 3: Create a heatmap visualization of wing vs wingtip intensity patterns
    # This combines both diff_gt and dark_lt metrics
    plt.figure(figsize=(16, 8))

    # Extract relevant metrics
    metrics = ['pct_diff_gt_10', 'pct_diff_gt_30', 'pct_diff_gt_50', 'pct_diff_gt_70',
               'pct_dark_lt_30', 'pct_dark_lt_40', 'pct_dark_lt_50', 'pct_dark_lt_60']

    metrics_data = wingtip_data[['species'] + metrics].set_index('species')

    # Rename for readability
    metrics_data.columns = [
        'Darker than wing by >10', 'Darker than wing by >30',
        'Darker than wing by >50', 'Darker than wing by >70',
        'Very dark pixels (<30)', 'Dark pixels (<40)',
        'Dark pixels (<50)', 'Dark pixels (<60)'
    ]

    # Create heatmap
    sns.heatmap(metrics_data, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, cbar_kws={'label': 'Percentage (%)'})

    plt.title('Comparison of Wing-Wingtip Contrast and Absolute Darkness', fontsize=16)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wingtip_darkness_heatmap.png'), dpi=300)
    plt.close()

    # Figure 4: Create a scatterplot of wing intensity vs. darkness differences
    if 'mean_wing_intensity' in detailed_data.columns and 'pct_dark_lt_30' in detailed_data.columns:
        plt.figure(figsize=(12, 10))

        # Create a scatterplot
        sns.scatterplot(data=detailed_data,
                        x='mean_wing_intensity',
                        y='pct_dark_lt_30',
                        hue='species',
                        size='pct_diff_gt_50',
                        sizes=(20, 200),
                        alpha=0.7)

        plt.title('Relationship Between Wing Intensity and Very Dark Pixels', fontsize=16)
        plt.xlabel('Mean Wing Intensity', fontsize=14)
        plt.ylabel('Percentage of Very Dark Pixels (<30)', fontsize=14)
        plt.legend(title='Species', fontsize=12)

        # Add annotation for key differences
        for species in ['Slaty_Backed_Gull', 'Glaucous_Winged_Gull']:
            species_data = detailed_data[detailed_data['species'] == species]
            mean_x = species_data['mean_wing_intensity'].mean()
            mean_y = species_data['pct_dark_lt_30'].mean()
            plt.annotate(species.replace('_', '-'),
                         xy=(mean_x, mean_y),
                         xytext=(mean_x + (10 if species == 'Slaty_Backed_Gull' else -10),
                                 mean_y + (5 if species == 'Slaty_Backed_Gull' else -5)),
                         arrowprops=dict(arrowstyle='->'),
                         fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'wing_vs_darkness_scatter.png'), dpi=300)
        plt.close()


def analyze_diff_distribution():
    """Analyze the distribution of wing-wingtip intensity differences"""
    if 'intensity_0_10' not in detailed_data.columns:
        return

    # Create a figure for the distribution analysis
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)

    # 1. Distribution of wing-wingtip intensity differences by threshold
    ax1 = fig.add_subplot(gs[0, 0])

    # Create data for violin plot
    diff_columns = [col for col in detailed_data.columns if col.startswith('pct_diff_gt_')]
    diff_values = []

    for species in detailed_data['species'].unique():
        species_data = detailed_data[detailed_data['species'] == species]
        for col in diff_columns:
            threshold = int(col.split('_')[-1])
            for _, row in species_data.iterrows():
                diff_values.append({
                    'species': species.replace('_', '-'),
                    'threshold': threshold,
                    'percentage': row[col]
                })

    diff_df = pd.DataFrame(diff_values)

    # Group by threshold for boxplot
    threshold_data = []
    for species in diff_df['species'].unique():
        for threshold in diff_df['threshold'].unique():
            species_threshold_data = diff_df[(diff_df['species'] == species) &
                                             (diff_df['threshold'] == threshold)]
            if not species_threshold_data.empty:
                threshold_data.append({
                    'species': species,
                    'threshold': threshold,
                    'mean_percentage': species_threshold_data['percentage'].mean()
                })

    threshold_df = pd.DataFrame(threshold_data)

    # Create line plot showing the distribution by threshold
    sns.lineplot(data=threshold_df, x='threshold', y='mean_percentage',
                 hue='species', marker='o', ax=ax1)

    ax1.set_title('Distribution of Wing-Wingtip Intensity Differences', fontsize=16)
    ax1.set_xlabel('Intensity Difference Threshold', fontsize=14)
    ax1.set_ylabel('Percentage of Pixels (%)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 2. Comparison of distributions between species
    ax2 = fig.add_subplot(gs[0, 1])

    # Calculate the difference between species at each threshold
    diff_between_species = []
    thresholds = sorted(threshold_df['threshold'].unique())

    for threshold in thresholds:
        slaty_data = threshold_df[(threshold_df['species'] == 'Slaty-Backed-Gull') &
                                  (threshold_df['threshold'] == threshold)]
        glaucous_data = threshold_df[(threshold_df['species'] == 'Glaucous-Winged-Gull') &
                                     (threshold_df['threshold'] == threshold)]

        if not slaty_data.empty and not glaucous_data.empty:
            slaty_pct = slaty_data['mean_percentage'].values[0]
            glaucous_pct = glaucous_data['mean_percentage'].values[0]
            diff_pct = slaty_pct - glaucous_pct

            diff_between_species.append({
                'threshold': threshold,
                'difference': diff_pct
            })

    diff_between_df = pd.DataFrame(diff_between_species)

    # Create bar chart showing the difference
    bars = sns.barplot(data=diff_between_df, x='threshold', y='difference', ax=ax2)

    # Color positive bars blue and negative bars red
    for i, bar in enumerate(bars.patches):
        if bar.get_height() >= 0:
            bar.set_color('blue')
        else:
            bar.set_color('red')

    ax2.set_title('Difference in Darkness Between Species by Threshold', fontsize=16)
    ax2.set_xlabel('Intensity Difference Threshold', fontsize=14)
    ax2.set_ylabel('Difference in Percentage (Slaty - Glaucous)', fontsize=14)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add value labels
    for i, p in enumerate(ax2.patches):
        ax2.annotate(f'{p.get_height():.1f}%',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center' if p.get_height() < 0 else 'bottom',
                     xytext=(0, 9 if p.get_height() >= 0 else -9),
                     textcoords='offset points')

    # 3. Darkness difference heatmap by individual images
    ax3 = fig.add_subplot(gs[1, :])

    # Create data for the heatmap
    key_thresholds = [10, 30, 50, 70]
    heatmap_data = []

    for species in detailed_data['species'].unique():
        species_data = detailed_data[detailed_data['species'] == species]
        for _, row in species_data.iterrows():
            for threshold in key_thresholds:
                col = f'pct_diff_gt_{threshold}'
                if col in detailed_data.columns:
                    heatmap_data.append({
                        'species': species.replace('_', '-'),
                        'image': row['image_name'],
                        'threshold': f'>{threshold}',
                        'percentage': row[col]
                    })

    heatmap_df = pd.DataFrame(heatmap_data)

    # Pivot the data for the heatmap
    heatmap_pivot = heatmap_df.pivot_table(index='image',
                                           columns=['species', 'threshold'],
                                           values='percentage')

    # Sort by species and threshold
    # Limit to 10 images per species for readability
    slaty_images = detailed_data[detailed_data['species'] == 'Slaty_Backed_Gull']['image_name'].unique()[:10]
    glaucous_images = detailed_data[detailed_data['species'] == 'Glaucous_Winged_Gull']['image_name'].unique()[:10]
    selected_images = np.concatenate([slaty_images, glaucous_images])

    if len(selected_images) > 0:
        # Filter heatmap data to selected images
        filtered_heatmap = heatmap_pivot.loc[heatmap_pivot.index.isin(selected_images)]

        # Plot heatmap
        sns.heatmap(filtered_heatmap, cmap='YlOrRd', annot=False,
                    cbar_kws={'label': 'Percentage (%)'})

        ax3.set_title('Distribution of Wing-Wingtip Intensity Differences by Image', fontsize=16)
        ax3.set_xlabel('Species and Threshold', fontsize=14)
        ax3.set_ylabel('Image', fontsize=14)
    else:
        ax3.text(0.5, 0.5, "Insufficient image data for heatmap",
                 ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wingtip_diff_distribution.png'), dpi=300)
    plt.close()


# Run the visualization functions
visualize_intensity_differences()
analyze_diff_distribution()

print(f"Analysis complete! Results saved to {output_dir}/")
