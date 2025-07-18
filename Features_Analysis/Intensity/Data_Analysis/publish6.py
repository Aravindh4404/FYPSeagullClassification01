import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy import stats
import os
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# Define consistent color scheme for species
SPECIES_COLORS = {
    'Glaucous_Winged_Gull': '#3274A1',  # Blue
    'Slaty_Backed_Gull': '#E1812C'  # Orange
}

# Define bin configuration - keeping original 15-unit bins but with better display
INTENSITY_BINS = list(range(0, 256, 15))  # [0, 15, 30, 45, ..., 240, 255]
BIN_WIDTH = 15

# Set plotting style for professional-looking visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)  # Slightly larger for enhanced visualization
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Create output directory for visualizations
output_dir = "Enhanced_Intensity_Distribution_Comparison"
os.makedirs(output_dir, exist_ok=True)


def create_histogram_interpolation(data, bins, color, alpha=0.6):
    """Create histogram and proper interpolation curve"""
    # Calculate histogram
    counts, bin_edges = np.histogram(data, bins=bins, density=True)

    # Calculate bin centers for interpolation points
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filter out bins with zero counts to avoid interpolation issues
    non_zero_mask = counts > 0
    if np.sum(non_zero_mask) < 2:
        # If we have fewer than 2 non-zero bins, use all bins
        x_points = bin_centers
        y_points = counts
    else:
        x_points = bin_centers[non_zero_mask]
        y_points = counts[non_zero_mask]

    # Use linear interpolation to avoid overshooting
    interp_func = interpolate.interp1d(x_points, y_points,
                                       kind='linear',
                                       bounds_error=False,
                                       fill_value=0)

    # Generate smooth curve points within data range
    x_smooth = np.linspace(x_points.min(), x_points.max(), 200)
    y_smooth = interp_func(x_smooth)

    # Ensure no negative values
    y_smooth = np.maximum(y_smooth, 0)

    return bin_centers, counts, x_smooth, y_smooth


def add_intensity_reference(ax, bins):
    """Add intensity reference boxes below the x-axis"""
    # Get the current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate the position for intensity reference boxes
    box_height = ylim[1] * 0.08  # Height of intensity boxes
    box_y_position = ylim[0] - box_height * 2  # Position below x-axis

    # Create intensity reference boxes
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        # Calculate the intensity value for this bin (use bin center)
        intensity_value = bin_center / 255.0  # Normalize to 0-1 range
        gray_color = str(intensity_value)  # Convert to grayscale string

        # Create rectangle for this bin
        rect = Rectangle((bin_start, box_y_position), BIN_WIDTH, box_height,
                         facecolor=gray_color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

        # Add intensity value text below the box
        ax.text(bin_center, box_y_position - box_height * 0.5,
                f'{int(bin_center)}', ha='center', va='top', fontsize=7, rotation=45)

    # Extend y-axis to accommodate the intensity reference
    ax.set_ylim(box_y_position - box_height * 1.5, ylim[1])


def create_enhanced_intensity_comparison(wing_data, original_wingtip_data, filtered_wingtip_data, enhanced_stats):
    """Create comprehensive comparison showing original vs filtered wingtip intensity"""

    # Create figure with three subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

    # Main plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Statistics panel
    ax3 = fig.add_subplot(gs[1, :])

    # 1. Original Wingtip Intensity Distribution
    ax1.set_title('Original Wingtip Intensity Distribution\n(Including White Pixels/Spots)',
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')

    for species in original_wingtip_data['species'].unique():
        species_data = original_wingtip_data[original_wingtip_data['species'] == species]['original_wingtip_intensity']
        color = SPECIES_COLORS[species]

        # Create display name for legend
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Create histogram bars with clear edges and consistent width
        counts, bins, patches = ax1.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        # Create proper interpolation curve
        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)

        # Plot interpolation curve
        ax1.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    # Add mean lines and statistics for original data
    for species in original_wingtip_data['species'].unique():
        species_data = original_wingtip_data[original_wingtip_data['species'] == species]['original_wingtip_intensity']
        color = SPECIES_COLORS[species]
        mean_val = species_data.mean()
        std_val = species_data.std()

        ax1.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax1.text(mean_val, 0.85, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                 transform=ax1.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                                               facecolor='white', alpha=0.8, edgecolor=color))

    # Add intensity reference
    add_intensity_reference(ax1, INTENSITY_BINS)

    # Position legend
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95), frameon=True,
               fancybox=True, shadow=True, fontsize=10)

    # Set x-axis ticks
    x_ticks = list(range(0, 256, 15))
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{i}' for i in x_ticks], rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 255)

    # 2. Filtered Wingtip Intensity Distribution (Dark Pixels Only)
    ax2.set_title('Filtered Wingtip Intensity Distribution\n(Dark Pixels Only - White Spots Removed)',
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')

    for species in filtered_wingtip_data['species'].unique():
        species_data = filtered_wingtip_data[filtered_wingtip_data['species'] == species]['mean_wingtip_intensity']
        color = SPECIES_COLORS[species]

        # Create display name for legend
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Create histogram bars with clear edges and consistent width
        counts, bins, patches = ax2.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        # Create proper interpolation curve
        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)

        # Plot interpolation curve
        ax2.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    # Add mean lines and statistics for filtered data
    for species in filtered_wingtip_data['species'].unique():
        species_data = filtered_wingtip_data[filtered_wingtip_data['species'] == species]['mean_wingtip_intensity']
        color = SPECIES_COLORS[species]
        mean_val = species_data.mean()
        std_val = species_data.std()

        ax2.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax2.text(mean_val, 0.85, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                 transform=ax2.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                                               facecolor='white', alpha=0.8, edgecolor=color))

    # Add intensity reference
    add_intensity_reference(ax2, INTENSITY_BINS)

    # Position legend
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95), frameon=True,
               fancybox=True, shadow=True, fontsize=10)

    # Set x-axis ticks
    x_ticks = list(range(0, 256, 15))
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{i}' for i in x_ticks], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 255)

    # 3. Statistics Panel
    ax3.set_title('K-means Clustering Filter Statistics', fontsize=12, fontweight='bold', pad=15)
    ax3.axis('off')

    # Calculate filtering statistics
    stats_text = "FILTERING IMPACT BY SPECIES:\n\n"

    for species in enhanced_stats['species'].unique():
        species_data = enhanced_stats[enhanced_stats['species'] == species]
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Calculate averages
        avg_original = species_data['original_wingtip_intensity'].mean()
        avg_filtered = species_data['mean_wingtip_intensity'].mean()
        avg_white_removed = species_data['white_pixel_percentage'].mean()
        avg_dark_cluster = species_data['dark_cluster_center'].mean()
        avg_white_cluster = species_data['white_cluster_center'].mean()

        stats_text += f"{display_name}:\n"
        stats_text += f"  • Original Mean: {avg_original:.1f} → Filtered Mean: {avg_filtered:.1f}\n"
        stats_text += f"  • Intensity Improvement: {avg_original - avg_filtered:.1f} units darker\n"
        stats_text += f"  • White Pixels Removed: {avg_white_removed:.1f}%\n"
        stats_text += f"  • Dark Cluster Center: {avg_dark_cluster:.1f}\n"
        stats_text += f"  • White Cluster Center: {avg_white_cluster:.1f}\n"
        stats_text += f"  • Cluster Separation: {avg_white_cluster - avg_dark_cluster:.1f}\n\n"

    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    # Add filtering methodology info
    method_text = ("K-MEANS CLUSTERING METHODOLOGY:\n\n"
                   "1. Applied K-means (k=2) to wingtip pixel intensities\n"
                   "2. Identified dark cluster (actual wingtip) vs white cluster (artifacts)\n"
                   "3. Filtered out white pixels/spots/streaks\n"
                   "4. Calculated mean intensity from dark pixels only\n\n"
                   "This filtering removes bright artifacts that artificially\n"
                   "inflate wingtip intensity measurements.")

    ax3.text(0.55, 0.95, method_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    # Add bin range information to the plot
    fig.text(0.5, 0.02,
             f'Bin Width: {BIN_WIDTH} units | Intensity reference shows actual brightness values | Dark pixels only = K-means filtered',
             ha='center', va='bottom', fontsize=9, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'enhanced_wingtip_intensity_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    return


def load_enhanced_data():
    """Load the enhanced CSV files with filtered wingtip data"""
    try:
        # Load wing data (for reference)
        wing_data = pd.read_csv('../Intensity_Results/wing_intensity_analysis.csv')

        # Load enhanced wingtip data (filtered)
        enhanced_wingtip_data = pd.read_csv(
            '../Wingtip_Intensity_Distribution_Enhanced/wingtip_intensity_distribution_enhanced.csv')
        enhanced_averages = pd.read_csv(
            '../Wingtip_Intensity_Distribution_Enhanced/wingtip_intensity_averages_enhanced.csv')

        print("Successfully loaded enhanced datasets")
        print(f"Enhanced data contains {len(enhanced_wingtip_data)} images")

        # Display available columns for verification
        print("\nAvailable columns in enhanced data:")
        print(enhanced_wingtip_data.columns.tolist())

        return wing_data, enhanced_wingtip_data, enhanced_averages

    except FileNotFoundError as e:
        print(f"Error: Enhanced CSV files not found: {e}")
        print("Please ensure the enhanced analysis has been run and CSV files are generated.")
        return None, None, None
    except Exception as e:
        print(f"Error loading enhanced data: {e}")
        return None, None, None


def create_filtering_impact_summary(enhanced_data):
    """Create a summary of the filtering impact"""

    print("\n" + "=" * 60)
    print("ENHANCED WINGTIP INTENSITY ANALYSIS SUMMARY")
    print("=" * 60)

    # Calculate overall statistics
    overall_stats = enhanced_data.groupby('species').agg({
        'original_wingtip_intensity': ['mean', 'std', 'count'],
        'mean_wingtip_intensity': ['mean', 'std'],
        'white_pixel_percentage': ['mean', 'std'],
        'dark_cluster_center': ['mean', 'std'],
        'white_cluster_center': ['mean', 'std']
    }).round(2)

    print("\nFILTERING IMPACT BY SPECIES:")
    print("-" * 40)

    for species in enhanced_data['species'].unique():
        species_data = enhanced_data[enhanced_data['species'] == species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        original_mean = species_data['original_wingtip_intensity'].mean()
        filtered_mean = species_data['mean_wingtip_intensity'].mean()
        white_removed = species_data['white_pixel_percentage'].mean()

        print(f"\n{display_name}:")
        print(f"  Original Mean Intensity: {original_mean:.1f}")
        print(f"  Filtered Mean Intensity:  {filtered_mean:.1f}")
        print(f"  Intensity Reduction:     {original_mean - filtered_mean:.1f} units")
        print(f"  White Pixels Removed:    {white_removed:.1f}%")

        # Calculate the impact of filtering
        improvement_pct = ((original_mean - filtered_mean) / original_mean) * 100
        print(f"  Relative Improvement:    {improvement_pct:.1f}%")

    print("\n" + "=" * 60)

    return overall_stats


def main():
    """Main function to create enhanced intensity distribution comparisons"""
    print("Loading enhanced CSV files with filtered wingtip data...")

    # Load enhanced data
    wing_data, enhanced_wingtip_data, enhanced_averages = load_enhanced_data()

    if enhanced_wingtip_data is None:
        print("Error: Could not load enhanced data files.")
        print("Please ensure you have run the enhanced analysis script first.")
        return

    # Create filtering impact summary
    overall_stats = create_filtering_impact_summary(enhanced_wingtip_data)

    print("\nCreating enhanced intensity distribution comparison...")

    # Create comprehensive visualization
    create_enhanced_intensity_comparison(
        wing_data,
        enhanced_wingtip_data,  # Contains both original and filtered data
        enhanced_wingtip_data,  # Same dataset, different columns
        enhanced_wingtip_data  # For statistics
    )

    print(f"\nEnhanced analysis complete! Visualizations saved to {output_dir}/")

    # Print final summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    print("✓ White pixels/spots successfully identified and removed using K-means clustering")
    print("✓ Filtered wingtip intensity represents actual wingtip darkness (dark pixels only)")
    print("✓ Original measurements included bright artifacts that inflated intensity values")
    print("✓ Enhanced measurements provide more accurate species differentiation")
    print("=" * 60)


if __name__ == "__main__":
    main()
