import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy import stats
from scipy.interpolate import UnivariateSpline
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
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Create output directory for visualizations
output_dir = "Enhanced_Intensity_Distribution_Comparison"
os.makedirs(output_dir, exist_ok=True)


def create_histogram_interpolation(data, bins, color, alpha=0.6):
    """
    Create histogram and proper interpolation curve with correct density handling.

    Key improvements:
    1. Use counts instead of density for interpolation
    2. Properly normalize the interpolated curve
    3. Handle edge cases more robustly
    """
    # Calculate histogram with counts (not density)
    counts, bin_edges = np.histogram(data, bins=bins)

    # Calculate bin centers for interpolation points
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Convert counts to density manually for consistency
    # Density = counts / (total_samples * bin_width)
    bin_widths = np.diff(bin_edges)
    density = counts / (np.sum(counts) * bin_widths)

    # Filter out bins with zero counts to avoid interpolation issues
    non_zero_mask = counts > 0

    if np.sum(non_zero_mask) < 2:
        # If we have fewer than 2 non-zero bins, return minimal interpolation
        x_smooth = bin_centers
        y_smooth = density
    else:
        x_points = bin_centers[non_zero_mask]
        y_points = density[non_zero_mask]

        # Use cubic spline interpolation with smoothing for better visual results
        # Alternative: Use UnivariateSpline with smoothing parameter
        try:
            # Cubic spline with smoothing (s parameter controls smoothing amount)
            spline = UnivariateSpline(x_points, y_points, k=3, s=0.001)

            # Generate smooth curve points
            x_smooth = np.linspace(x_points.min(), x_points.max(), 200)
            y_smooth = spline(x_smooth)

            # Ensure no negative values (can happen with spline interpolation)
            y_smooth = np.maximum(y_smooth, 0)

        except Exception as e:
            # Fallback to linear interpolation if spline fails
            interp_func = interpolate.interp1d(x_points, y_points,
                                               kind='linear',
                                               bounds_error=False,
                                               fill_value=0)
            x_smooth = np.linspace(x_points.min(), x_points.max(), 200)
            y_smooth = interp_func(x_smooth)
            y_smooth = np.maximum(y_smooth, 0)

    return bin_centers, density, x_smooth, y_smooth


def create_histogram_kde(data, bins, color, alpha=0.6):
    """
    Alternative: Use Kernel Density Estimation (KDE) for smooth curves.
    This is often more statistically appropriate than interpolation.
    """
    # Calculate histogram for bars (using density=True)
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Use KDE for smooth curve
    try:
        kde = stats.gaussian_kde(data, bw_method='scott')  # or 'silverman'
        x_smooth = np.linspace(data.min(), data.max(), 200)
        y_smooth = kde(x_smooth)
    except Exception as e:
        # Fallback if KDE fails
        x_smooth = bin_centers
        y_smooth = counts

    return bin_centers, counts, x_smooth, y_smooth


def add_intensity_reference(ax, bins):
    """Add intensity reference boxes below the x-axis"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    box_height = ylim[1] * 0.08
    box_y_position = ylim[0] - box_height * 2

    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        intensity_value = bin_center / 255.0
        gray_color = str(intensity_value)

        rect = Rectangle((bin_start, box_y_position), BIN_WIDTH, box_height,
                         facecolor=gray_color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

        ax.text(bin_center, box_y_position - box_height * 0.5,
                f'{int(bin_center)}', ha='center', va='top', fontsize=7, rotation=45)

    ax.set_ylim(box_y_position - box_height * 1.5, ylim[1])


def create_enhanced_intensity_comparison(wing_data, original_wingtip_data, filtered_wingtip_data, enhanced_stats,
                                         use_kde=False):
    """
    Create comprehensive comparison showing original vs filtered wingtip intensity.

    Parameters:
    -----------
    use_kde : bool
        If True, use KDE for smooth curves. If False, use spline interpolation.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # Choose interpolation method
    interpolation_func = create_histogram_kde if use_kde else create_histogram_interpolation
    method_name = "KDE" if use_kde else "Spline Interpolation"

    # 1. Original Wingtip Intensity Distribution
    ax1.set_title('Original Wingtip Intensity Distribution\n(Including White Pixels/Spots)',
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Probability Density')

    for species in original_wingtip_data['species'].unique():
        species_data = original_wingtip_data[original_wingtip_data['species'] == species]['original_wingtip_intensity']
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Create histogram bars
        counts, bins, patches = ax1.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        # Create smooth curve
        bin_centers, hist_counts, x_smooth, y_smooth = interpolation_func(
            species_data, INTENSITY_BINS, color)

        ax1.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    # Add mean lines and statistics
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

    add_intensity_reference(ax1, INTENSITY_BINS)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95), frameon=True,
               fancybox=True, shadow=True, fontsize=10)

    x_ticks = list(range(0, 256, 15))
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{i}' for i in x_ticks], rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 255)

    # 2. Filtered Wingtip Intensity Distribution
    ax2.set_title('Filtered Wingtip Intensity Distribution\n(Dark Pixels Only - White Spots Removed)',
                  fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Probability Density')

    for species in filtered_wingtip_data['species'].unique():
        species_data = filtered_wingtip_data[filtered_wingtip_data['species'] == species]['mean_wingtip_intensity']
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        counts, bins, patches = ax2.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        bin_centers, hist_counts, x_smooth, y_smooth = interpolation_func(
            species_data, INTENSITY_BINS, color)

        ax2.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

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

    add_intensity_reference(ax2, INTENSITY_BINS)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95), frameon=True,
               fancybox=True, shadow=True, fontsize=10)

    x_ticks = list(range(0, 256, 15))
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{i}' for i in x_ticks], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 255)

    # 3. Statistics Panel
    ax3.set_title('K-means Clustering Filter Statistics', fontsize=12, fontweight='bold', pad=15)
    ax3.axis('off')

    stats_text = "FILTERING IMPACT BY SPECIES:\n\n"

    for species in enhanced_stats['species'].unique():
        species_data = enhanced_stats[enhanced_stats['species'] == species]
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

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

    method_text = (f"SMOOTH CURVE METHOD: {method_name}\n\n"
                   "K-MEANS CLUSTERING METHODOLOGY:\n"
                   "1. Applied K-means (k=2) to wingtip pixel intensities\n"
                   "2. Identified dark cluster (actual wingtip) vs white cluster (artifacts)\n"
                   "3. Filtered out white pixels/spots/streaks\n"
                   "4. Calculated mean intensity from dark pixels only\n\n"
                   "Histogram bars show probability density (area = 1)\n"
                   "Smooth curves show the estimated probability distribution")

    ax3.text(0.55, 0.95, method_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    fig.text(0.5, 0.02,
             f'Bin Width: {BIN_WIDTH} units | Y-axis: Probability Density | Method: {method_name}',
             ha='center', va='bottom', fontsize=9, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    filename = f'enhanced_wingtip_intensity_comparison_{method_name.lower().replace(" ", "_")}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()


def load_enhanced_data():
    """Load the enhanced CSV files with filtered wingtip data"""
    try:
        wing_data = pd.read_csv('D:/FYPSeagullClassification01/Wing_Intensity_Results_New/wing_intensity_analysis.csv')
        enhanced_wingtip_data = pd.read_csv(
            '../Wingtip_Intensity_Distribution_Enhanced/wingtip_intensity_distribution_enhanced.csv')
        enhanced_averages = pd.read_csv(
            '../Wingtip_Intensity_Distribution_Enhanced/wingtip_intensity_averages_enhanced.csv')

        print("Successfully loaded enhanced datasets")
        print(f"Enhanced data contains {len(enhanced_wingtip_data)} images")
        print("\nAvailable columns in enhanced data:")
        print(enhanced_wingtip_data.columns.tolist())

        return wing_data, enhanced_wingtip_data, enhanced_averages

    except FileNotFoundError as e:
        print(f"Error: Enhanced CSV files not found: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading enhanced data: {e}")
        return None, None, None


def create_filtering_impact_summary(enhanced_data):
    """Create a summary of the filtering impact"""
    print("\n" + "=" * 60)
    print("ENHANCED WINGTIP INTENSITY ANALYSIS SUMMARY")
    print("=" * 60)

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

        improvement_pct = ((original_mean - filtered_mean) / original_mean) * 100
        print(f"  Relative Improvement:    {improvement_pct:.1f}%")

    print("\n" + "=" * 60)


def main():
    """Main function - creates visualizations with both methods"""
    print("Loading enhanced CSV files with filtered wingtip data...")

    wing_data, enhanced_wingtip_data, enhanced_averages = load_enhanced_data()

    if enhanced_wingtip_data is None:
        print("Error: Could not load enhanced data files.")
        return

    create_filtering_impact_summary(enhanced_wingtip_data)

    print("\nCreating enhanced intensity distribution comparisons...")

    # Create visualization with spline interpolation
    print("\n1. Creating visualization with spline interpolation...")
    create_enhanced_intensity_comparison(
        wing_data, enhanced_wingtip_data, enhanced_wingtip_data,
        enhanced_wingtip_data, use_kde=False
    )

    # Create visualization with KDE
    print("\n2. Creating visualization with KDE...")
    create_enhanced_intensity_comparison(
        wing_data, enhanced_wingtip_data, enhanced_wingtip_data,
        enhanced_wingtip_data, use_kde=True
    )

    print(f"\nEnhanced analysis complete! Visualizations saved to {output_dir}/")
    print("\n" + "=" * 60)
    print("KEY IMPROVEMENTS:")
    print("=" * 60)
    print("✓ Density calculation corrected for proper normalization")
    print("✓ Two smoothing methods provided: Spline and KDE")
    print("✓ KDE is statistically more appropriate for distribution estimation")
    print("✓ Y-axis correctly labeled as 'Probability Density'")
    print("=" * 60)


if __name__ == "__main__":
    main()