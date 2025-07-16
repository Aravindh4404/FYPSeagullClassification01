import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy import stats
import os
from matplotlib.gridspec import GridSpec

# Define consistent color scheme for species
SPECIES_COLORS = {
    'Glaucous_Winged_Gull': '#3274A1',
    'Slaty_Backed_Gull': '#E1812C'
}

# Define bin configuration - keeping original 15-unit bins but with better display
INTENSITY_BINS = list(range(0, 256, 15))  # [0, 15, 30, 45, ..., 240, 255]
BIN_WIDTH = 15

# Bin ranges for clarity:
# Bin 1: 0-14,   Bin 2: 15-29,  Bin 3: 30-44,  Bin 4: 45-59,  Bin 5: 60-74,
# Bin 6: 75-89,  Bin 7: 90-104, Bin 8: 105-119, Bin 9: 120-134, Bin 10: 135-149,
# Bin 11: 150-164, Bin 12: 165-179, Bin 13: 180-194, Bin 14: 195-209,
# Bin 15: 210-224, Bin 16: 225-239, Bin 17: 240-255

# Set plotting style for professional-looking visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)  # Reduced height: was (12, 7), now (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Create output directory for visualizations
output_dir = "Intensity_Distribution_Comparison"
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


def create_intensity_distribution_comparison_improved(wing_data, wingtip_distribution):
    """Create side-by-side comparison with improved bin visualization"""

    # Create figure with two subplots - good height for spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 1. Wing Intensity Distribution
    ax1.set_title('Wing Intensity Distribution', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')

    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS[species]

        # Create histogram bars with clear edges and consistent width
        counts, bins, patches = ax1.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=f'{species.replace("_", "-")}')

        # Create proper interpolation curve
        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)

        # Plot interpolation curve
        ax1.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    # Add mean lines and statistics - positioned much higher
    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS[species]
        mean_val = species_data.mean()
        std_val = species_data.std()

        ax1.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        # Move text boxes to good position - using 0.85 instead of 0.9
        ax1.text(mean_val, 0.85, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                 transform=ax1.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                                               facecolor='white', alpha=0.8, edgecolor=color))

    # Position legend much higher - moved even higher
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95), frameon=True,
               fancybox=True, shadow=True, fontsize=10)

    # Set x-axis ticks to show bin boundaries clearly - from 0 to 255
    x_ticks = list(range(0, 256, 15))  # [0, 15, 30, 45, ..., 240, 255]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f'{i}' for i in x_ticks], rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 255)  # Ensure full range is shown

    # 2. Wingtip Intensity Distribution
    ax2.set_title('Wingtip Intensity Distribution', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')

    for species in wingtip_distribution['species'].unique():
        species_data = wingtip_distribution[wingtip_distribution['species'] == species]['mean_wingtip_intensity']
        color = SPECIES_COLORS[species]

        # Create histogram bars with clear edges and consistent width
        counts, bins, patches = ax2.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=f'{species.replace("_", "-")}')

        # Create proper interpolation curve
        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)

        # Plot interpolation curve
        ax2.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    # Add mean lines and statistics - positioned much higher
    for species in wingtip_distribution['species'].unique():
        species_data = wingtip_distribution[wingtip_distribution['species'] == species]['mean_wingtip_intensity']
        color = SPECIES_COLORS[species]
        mean_val = species_data.mean()
        std_val = species_data.std()

        ax2.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        # Move text boxes to good position - using 0.85 instead of 0.9
        ax2.text(mean_val, 0.8, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                 transform=ax2.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                                               facecolor='white', alpha=0.8, edgecolor=color))

    # Position legend much higher - moved even higher
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 0.95), frameon=True,
               fancybox=True, shadow=True, fontsize=10)

    # Set x-axis ticks to show bin boundaries clearly - from 0 to 255
    x_ticks = list(range(0, 256, 15))  # [0, 15, 30, 45, ..., 240, 255]
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{i}' for i in x_ticks], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 255)  # Ensure full range is shown

    # Ensure both plots show the full 0-255 range
    ax1.set_xlim(0, 255)
    ax2.set_xlim(0, 255)

    # Make y-axis ranges similar for better comparison, with extra space at top
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    # Add more space at the top to ensure legend and mean boxes don't overlap with histograms
    ax1.set_ylim(0, y_max * 1.35)  # Increased from 1.15 to 1.35
    ax2.set_ylim(0, y_max * 1.35)  # Increased from 1.15 to 1.35

    # Add bin range information to the plot
    fig.text(0.5, 0.02, f'Bin Width: {BIN_WIDTH} units | Range: 0-255 | Bins: 0-14, 15-29, 30-44, ..., 240-255',
             ha='center', va='bottom', fontsize=8, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Calculate and display summary statistics
    wing_stats = wing_data.groupby('species')['mean_intensity'].agg(['mean', 'std', 'count'])
    wingtip_stats = wingtip_distribution.groupby('species')['mean_wingtip_intensity'].agg(['mean', 'std', 'count'])

    plt.savefig(os.path.join(output_dir, 'wing_wingtip_intensity_comparison_improved.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    return wing_stats, wingtip_stats


def create_overlaid_comparison_improved(wing_data, wingtip_distribution):
    """Create an overlaid comparison with improved bins"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))  # Reduced height: was (16, 8), now (16, 7)

    # Species 1: Slaty-backed Gull
    slaty_wing = wing_data[wing_data['species'] == 'Slaty_Backed_Gull']['mean_intensity']
    slaty_wingtip = wingtip_distribution[wingtip_distribution['species'] == 'Slaty_Backed_Gull'][
        'mean_wingtip_intensity']

    # Plot histograms with consistent bins
    ax1.hist(slaty_wing, bins=INTENSITY_BINS, alpha=0.5,
             color=SPECIES_COLORS['Slaty_Backed_Gull'], density=True,
             label='Wing', edgecolor='black', linewidth=0.5)
    ax1.hist(slaty_wingtip, bins=INTENSITY_BINS, alpha=0.5,
             color='red', density=True, label='Wingtip',
             edgecolor='black', linewidth=0.5)

    # Plot interpolation curves
    _, _, x_wing, y_wing = create_histogram_interpolation(slaty_wing, INTENSITY_BINS,
                                                          SPECIES_COLORS['Slaty_Backed_Gull'])
    _, _, x_wingtip, y_wingtip = create_histogram_interpolation(slaty_wingtip, INTENSITY_BINS, 'red')

    ax1.plot(x_wing, y_wing, color=SPECIES_COLORS['Slaty_Backed_Gull'],
             linewidth=2.5, alpha=0.9, label='Wing (trend)')
    ax1.plot(x_wingtip, y_wingtip, color='red', linewidth=2.5,
             linestyle='--', alpha=0.9, label='Wingtip (trend)')

    # Add mean lines
    ax1.axvline(slaty_wing.mean(), color=SPECIES_COLORS['Slaty_Backed_Gull'],
                linestyle=':', alpha=0.8, linewidth=2)
    ax1.axvline(slaty_wingtip.mean(), color='red', linestyle=':', alpha=0.8, linewidth=2)

    ax1.set_title('Slaty-backed Gull: Wing vs Wingtip Intensity', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')

    # Position legend higher using bbox_to_anchor
    ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.95), frameon=True,
               fancybox=True, shadow=True, fontsize=10)

    ax1.set_xticks(np.arange(0, 256, 15))  # Show every bin boundary
    ax1.set_xticklabels([f'{i}' for i in np.arange(0, 256, 15)], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Species 2: Glaucous-winged Gull
    glaucous_wing = wing_data[wing_data['species'] == 'Glaucous_Winged_Gull']['mean_intensity']
    glaucous_wingtip = wingtip_distribution[wingtip_distribution['species'] == 'Glaucous_Winged_Gull'][
        'mean_wingtip_intensity']

    # Plot histograms with consistent bins
    ax2.hist(glaucous_wing, bins=INTENSITY_BINS, alpha=0.5,
             color=SPECIES_COLORS['Glaucous_Winged_Gull'], density=True,
             label='Wing', edgecolor='black', linewidth=0.5)
    ax2.hist(glaucous_wingtip, bins=INTENSITY_BINS, alpha=0.5,
             color='red', density=True, label='Wingtip',
             edgecolor='black', linewidth=0.5)

    # Plot interpolation curves
    _, _, x_wing, y_wing = create_histogram_interpolation(glaucous_wing, INTENSITY_BINS,
                                                          SPECIES_COLORS['Glaucous_Winged_Gull'])
    _, _, x_wingtip, y_wingtip = create_histogram_interpolation(glaucous_wingtip, INTENSITY_BINS, 'red')

    ax2.plot(x_wing, y_wing, color=SPECIES_COLORS['Glaucous_Winged_Gull'],
             linewidth=2.5, alpha=0.9, label='Wing (trend)')
    ax2.plot(x_wingtip, y_wingtip, color='red', linewidth=2.5,
             linestyle='--', alpha=0.9, label='Wingtip (trend)')

    # Add mean lines
    ax2.axvline(glaucous_wing.mean(), color=SPECIES_COLORS['Glaucous_Winged_Gull'],
                linestyle=':', alpha=0.8, linewidth=2)
    ax2.axvline(glaucous_wingtip.mean(), color='red', linestyle=':', alpha=0.8, linewidth=2)

    ax2.set_title('Glaucous-winged Gull: Wing vs Wingtip Intensity', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')

    # Position legend higher using bbox_to_anchor
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.95), frameon=True,
               fancybox=True, shadow=True, fontsize=10)

    ax2.set_xticks(np.arange(0, 256, 15))  # Show every bin boundary
    ax2.set_xticklabels([f'{i}' for i in np.arange(0, 256, 15)], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Ensure consistent ranges
    all_values = np.concatenate([slaty_wing, slaty_wingtip, glaucous_wing, glaucous_wingtip])
    x_min, x_max = all_values.min() - 5, all_values.max() + 5
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    # Make y-axis ranges consistent with more space at top
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, y_max * 1.2)  # Added extra space at top
    ax2.set_ylim(0, y_max * 1.2)  # Added extra space at top

    # Add bin width information
    fig.text(0.5, 0.02, f'Bin Width: {BIN_WIDTH} intensity units',
             ha='center', va='bottom', fontsize=10, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.suptitle('Within-Species Comparison: Wing vs Wingtip Intensity (Improved Bins)',
                 fontsize=16, y=0.98)

    plt.savefig(os.path.join(output_dir, 'within_species_wing_wingtip_comparison_improved.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def print_bin_analysis(wing_data, wingtip_distribution):
    """Print analysis of data distribution across bins"""
    print("=== BIN ANALYSIS ===")
    print(f"Bin configuration: {len(INTENSITY_BINS) - 1} bins with width {BIN_WIDTH}")
    print(f"Bin edges: {INTENSITY_BINS[:5]}...{INTENSITY_BINS[-5:]}")
    print(f"Full range: {INTENSITY_BINS[0]} to {INTENSITY_BINS[-1]}")

    # Analyze wing data
    print("\n--- Wing Data Distribution ---")
    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        print(f"{species}:")
        print(f"  Range: {species_data.min():.1f} - {species_data.max():.1f}")
        print(f"  Mean: {species_data.mean():.1f} ± {species_data.std():.1f}")
        print(f"  Sample size: {len(species_data)}")

        # Count non-empty bins
        counts, _ = np.histogram(species_data, bins=INTENSITY_BINS)
        non_empty_bins = np.sum(counts > 0)
        print(f"  Non-empty bins: {non_empty_bins}/{len(INTENSITY_BINS) - 1}")

    # Analyze wingtip data
    print("\n--- Wingtip Data Distribution ---")
    for species in wingtip_distribution['species'].unique():
        species_data = wingtip_distribution[wingtip_distribution['species'] == species]['mean_wingtip_intensity']
        print(f"{species}:")
        print(f"  Range: {species_data.min():.1f} - {species_data.max():.1f}")
        print(f"  Mean: {species_data.mean():.1f} ± {species_data.std():.1f}")
        print(f"  Sample size: {len(species_data)}")

        # Count non-empty bins
        counts, _ = np.histogram(species_data, bins=INTENSITY_BINS)
        non_empty_bins = np.sum(counts > 0)
        print(f"  Non-empty bins: {non_empty_bins}/{len(INTENSITY_BINS) - 1}")


def load_data():
    """Load and prepare all necessary datasets"""
    try:
        wing_data = pd.read_csv('../Intensity_Results/wing_intensity_analysis.csv')
        wingtip_avg_data = pd.read_csv('../Wingtip_Intensity_Distribution/wingtip_intensity_averages.csv')
        wingtip_distribution = pd.read_csv('../Wingtip_Intensity_Distribution/wingtip_intensity_distribution.csv')
        print("Successfully loaded all datasets")
        return wing_data, wingtip_avg_data, wingtip_distribution
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def main():
    """Main function to create intensity distribution comparisons with improved bins"""
    # Load data
    wing_data, wingtip_avg_data, wingtip_distribution = load_data()

    if wing_data is None or wingtip_distribution is None:
        print("Error: Could not load required data files.")
        return

    # Print bin analysis first
    print_bin_analysis(wing_data, wingtip_distribution)

    print("\nCreating wing vs wingtip intensity distribution comparison with improved bins...")
    wing_stats, wingtip_stats = create_intensity_distribution_comparison_improved(wing_data, wingtip_distribution)

    print("\nCreating overlaid comparison with improved bins...")
    create_overlaid_comparison_improved(wing_data, wingtip_distribution)

    print(f"\nAnalysis complete! Charts saved to {output_dir}/")

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("\nWing Intensity Statistics:")
    print(wing_stats)
    print("\nWingtip Intensity Statistics:")
    print(wingtip_stats)


if __name__ == "__main__":
    main()