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
plt.rcParams['figure.figsize'] = (15, 10)  # Larger figure for additional plots
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Create output directory for visualizations
output_dir = "../Intensity_Distribution_Comparison_Filtered"
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


def create_comprehensive_intensity_comparison(wing_data, wingtip_distribution_filtered):
    """Create comprehensive comparison including original vs filtered wingtip data"""

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Wing Intensity Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Wing Intensity Distribution', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')

    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Create histogram and interpolation
        counts, bins, patches = ax1.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)
        ax1.plot(x_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.9)

        # Add mean line
        mean_val = species_data.mean()
        std_val = species_data.std()
        ax1.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax1.text(mean_val, 0.85, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                 transform=ax1.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                                              facecolor='white', alpha=0.8, edgecolor=color))

    add_intensity_reference(ax1, INTENSITY_BINS)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 255)
    ax1.grid(True, alpha=0.3)

    # 2. Original Wingtip Intensity Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Original Wingtip Intensity Distribution\n(Including White Spots)', fontsize=12, fontweight='bold',
                  pad=15)
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')

    for species in wingtip_distribution_filtered['species'].unique():
        species_data = wingtip_distribution_filtered[wingtip_distribution_filtered['species'] == species][
            'original_wingtip_intensity']
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        counts, bins, patches = ax2.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)
        ax2.plot(x_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.9)

        mean_val = species_data.mean()
        std_val = species_data.std()
        ax2.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax2.text(mean_val, 0.85, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                 transform=ax2.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                                              facecolor='white', alpha=0.8, edgecolor=color))

    add_intensity_reference(ax2, INTENSITY_BINS)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, 255)
    ax2.grid(True, alpha=0.3)

    # 3. Filtered Wingtip Intensity Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('Filtered Wingtip Intensity Distribution\n(White Spots Removed)', fontsize=12, fontweight='bold',
                  pad=15)
    ax3.set_xlabel('Mean Intensity (0-255)')
    ax3.set_ylabel('Density')

    for species in wingtip_distribution_filtered['species'].unique():
        species_data = wingtip_distribution_filtered[wingtip_distribution_filtered['species'] == species][
            'mean_wingtip_intensity']
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        counts, bins, patches = ax3.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)
        ax3.plot(x_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.9)

        mean_val = species_data.mean()
        std_val = species_data.std()
        ax3.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax3.text(mean_val, 0.85, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                 transform=ax3.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                                              facecolor='white', alpha=0.8, edgecolor=color))

    add_intensity_reference(ax3, INTENSITY_BINS)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(0, 255)
    ax3.grid(True, alpha=0.3)

    # 4. White Spot Removal Impact
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Impact of White Spot Removal', fontsize=12, fontweight='bold', pad=15)

    species_names = []
    original_means = []
    filtered_means = []
    white_percentages = []

    for species in wingtip_distribution_filtered['species'].unique():
        species_data = wingtip_distribution_filtered[wingtip_distribution_filtered['species'] == species]
        species_names.append(species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed'))
        original_means.append(species_data['original_wingtip_intensity'].mean())
        filtered_means.append(species_data['mean_wingtip_intensity'].mean())
        white_percentages.append(species_data['white_pixel_percentage'].mean())

    x = np.arange(len(species_names))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, original_means, width, label='Original Mean', alpha=0.7, color='lightcoral')
    bars2 = ax4.bar(x + width / 2, filtered_means, width, label='Filtered Mean', alpha=0.7, color='darkblue')

    ax4.set_ylabel('Mean Intensity')
    ax4.set_xlabel('Species')
    ax4.set_title('Original vs Filtered Wingtip Intensity')
    ax4.set_xticks(x)
    ax4.set_xticklabels(species_names, rotation=45, ha='right')
    ax4.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # 5. White Pixel Percentage by Species
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title('White Pixels Removed by Species', fontsize=12, fontweight='bold', pad=15)

    colors = [SPECIES_COLORS[species] for species in wingtip_distribution_filtered['species'].unique()]
    bars = ax5.bar(species_names, white_percentages, color=colors, alpha=0.7, edgecolor='black')

    ax5.set_ylabel('White Pixels Removed (%)')
    ax5.set_xlabel('Species')
    ax5.set_xticklabels(species_names, rotation=45, ha='right')

    # Add value labels on bars
    for bar, percentage in zip(bars, white_percentages):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax5.grid(True, alpha=0.3)

    # 6. Clustering Statistics Summary
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    # Calculate comprehensive statistics
    stats_text = "Clustering Analysis Summary:\n\n"

    for species in wingtip_distribution_filtered['species'].unique():
        species_data = wingtip_distribution_filtered[wingtip_distribution_filtered['species'] == species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        orig_mean = species_data['original_wingtip_intensity'].mean()
        filt_mean = species_data['mean_wingtip_intensity'].mean()
        white_pct = species_data['white_pixel_percentage'].mean()
        dark_center = species_data['dark_cluster_center'].mean()
        white_center = species_data['white_cluster_center'].mean()

        stats_text += f"{display_name}:\n"
        stats_text += f"  Original Mean: {orig_mean:.1f}\n"
        stats_text += f"  Filtered Mean: {filt_mean:.1f}\n"
        stats_text += f"  Improvement: {abs(orig_mean - filt_mean):.1f}\n"
        stats_text += f"  White Pixels: {white_pct:.1f}%\n"
        stats_text += f"  Dark Cluster Center: {dark_center:.1f}\n"
        stats_text += f"  White Cluster Center: {white_center:.1f}\n\n"

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_intensity_comparison_filtered.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def create_before_after_comparison(wingtip_distribution_filtered):
    """Create a focused before/after comparison"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Before: Original wingtip intensity
    ax1.set_title('BEFORE: Original Wingtip Intensity\n(Including White Spots)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')

    for species in wingtip_distribution_filtered['species'].unique():
        species_data = wingtip_distribution_filtered[wingtip_distribution_filtered['species'] == species][
            'original_wingtip_intensity']
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        counts, bins, patches = ax1.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)
        ax1.plot(x_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.9)

        mean_val = species_data.mean()
        ax1.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax1.text(mean_val, 0.85, f'Mean: {mean_val:.1f}',
                 transform=ax1.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                                               facecolor='white', alpha=0.8, edgecolor=color))

    add_intensity_reference(ax1, INTENSITY_BINS)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(0, 255)
    ax1.grid(True, alpha=0.3)

    # After: Filtered wingtip intensity
    ax2.set_title('AFTER: Filtered Wingtip Intensity\n(White Spots Removed with K-means)', fontsize=12,
                  fontweight='bold')
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')

    for species in wingtip_distribution_filtered['species'].unique():
        species_data = wingtip_distribution_filtered[wingtip_distribution_filtered['species'] == species][
            'mean_wingtip_intensity']
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        counts, bins, patches = ax2.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)
        ax2.plot(x_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.9)

        mean_val = species_data.mean()
        ax2.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax2.text(mean_val, 0.85, f'Mean: {mean_val:.1f}',
                 transform=ax2.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                                               facecolor='white', alpha=0.8, edgecolor=color))

    add_intensity_reference(ax2, INTENSITY_BINS)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(0, 255)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'before_after_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def load_data():
    """Load and prepare all necessary datasets"""
    try:
        # Try to load the original wing data
        wing_data = pd.read_csv('../Intensity_Results/wing_intensity_analysis.csv')
        print("Loaded wing intensity data")

        # Load the new filtered wingtip data
        wingtip_distribution_filtered = pd.read_csv(
            '../Wingtip_Intensity_Distribution_Filtered220/wingtip_intensity_distribution_filtered.csv')
        print("Loaded filtered wingtip intensity data")

        # Load the new filtered averages
        wingtip_avg_filtered = pd.read_csv(
            '../Wingtip_Intensity_Distribution_Filtered220/wingtip_intensity_averages_filtered.csv')
        print("Loaded filtered wingtip averages")

        return wing_data, wingtip_distribution_filtered, wingtip_avg_filtered
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def calculate_improvement_statistics(wingtip_distribution_filtered):
    """Calculate and print improvement statistics"""
    print("\n" + "=" * 60)
    print("WHITE SPOT REMOVAL IMPROVEMENT ANALYSIS")
    print("=" * 60)

    for species in wingtip_distribution_filtered['species'].unique():
        species_data = wingtip_distribution_filtered[wingtip_distribution_filtered['species'] == species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        orig_mean = species_data['original_wingtip_intensity'].mean()
        orig_std = species_data['original_wingtip_intensity'].std()
        filt_mean = species_data['mean_wingtip_intensity'].mean()
        filt_std = species_data['mean_wingtip_intensity'].std()

        white_pct = species_data['white_pixel_percentage'].mean()
        dark_center = species_data['dark_cluster_center'].mean()
        white_center = species_data['white_cluster_center'].mean()

        improvement = abs(orig_mean - filt_mean)

        print(f"\n{display_name}:")
        print(f"  Original Mean ± Std: {orig_mean:.1f} ± {orig_std:.1f}")
        print(f"  Filtered Mean ± Std: {filt_mean:.1f} ± {filt_std:.1f}")
        print(f"  Absolute Improvement: {improvement:.1f}")
        print(f"  White Pixels Removed: {white_pct:.1f}%")
        print(f"  Dark Cluster Center: {dark_center:.1f}")
        print(f"  White Cluster Center: {white_center:.1f}")
        print(f"  Cluster Separation: {abs(white_center - dark_center):.1f}")


def main():
    """Main function to create filtered intensity distribution comparisons"""
    # Load data
    wing_data, wingtip_distribution_filtered, wingtip_avg_filtered = load_data()

    if wing_data is None or wingtip_distribution_filtered is None:
        print("Error: Could not load required data files.")
        print("Make sure you have run the updated intensity analysis script first.")
        return

    print("Creating comprehensive intensity distribution comparison with filtered data...")
    create_comprehensive_intensity_comparison(wing_data, wingtip_distribution_filtered)

    print("Creating before/after comparison...")
    create_before_after_comparison(wingtip_distribution_filtered)

    # Calculate and display improvement statistics
    calculate_improvement_statistics(wingtip_distribution_filtered)

    print(f"\nAnalysis complete! All charts saved to {output_dir}/")


if __name__ == "__main__":
    main()