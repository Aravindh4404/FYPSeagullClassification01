import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy import stats
import os
from matplotlib.gridspec import GridSpec

# Define gray color scheme matching the actual gull appearance
SPECIES_COLORS = {
    'Glaucous_Winged_Gull': '#A8A8A8',  # Light gray for lighter-plumaged gull
    'Slaty_Backed_Gull': '#4A4A4A'  # Dark gray for darker-plumaged gull
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


def create_intensity_distribution_comparison_gray(wing_data, wingtip_distribution):
    """Create side-by-side comparison with gray colors matching gull appearance"""

    # Create figure with two subplots - good height for spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 1. Wing Intensity Distribution
    ax1.set_title('Wing Intensity Distribution', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')

    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS[species]

        # Create display name for legend
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Create histogram bars with clear edges and consistent width
        counts, bins, patches = ax1.hist(species_data, bins=INTENSITY_BINS, alpha=0.7,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        # Create proper interpolation curve
        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)

        # Plot interpolation curve with slightly darker version of the same color
        darker_color = '#2A2A2A' if species == 'Slaty_Backed_Gull' else '#606060'
        ax1.plot(x_smooth, y_smooth, color=darker_color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    # Add mean lines and statistics - positioned much higher
    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS[species]
        darker_color = '#2A2A2A' if species == 'Slaty_Backed_Gull' else '#606060'
        mean_val = species_data.mean()
        std_val = species_data.std()

        ax1.axvline(mean_val, color=darker_color, linestyle='--', alpha=0.8, linewidth=2)
        # Move text boxes to good position - using 0.85 instead of 0.9
        ax1.text(mean_val, 0.85, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                 transform=ax1.get_xaxis_transform(), color=darker_color, fontweight='bold',
                 ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                                               facecolor='white', alpha=0.8, edgecolor=darker_color))

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

        # Create display name for legend
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Create histogram bars with clear edges and consistent width
        counts, bins, patches = ax2.hist(species_data, bins=INTENSITY_BINS, alpha=0.7,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        # Create proper interpolation curve
        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)

        # Plot interpolation curve with slightly darker version of the same color
        darker_color = '#2A2A2A' if species == 'Slaty_Backed_Gull' else '#606060'
        ax2.plot(x_smooth, y_smooth, color=darker_color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    # Add mean lines and statistics - positioned much higher
    for species in wingtip_distribution['species'].unique():
        species_data = wingtip_distribution[wingtip_distribution['species'] == species]['mean_wingtip_intensity']
        color = SPECIES_COLORS[species]
        darker_color = '#2A2A2A' if species == 'Slaty_Backed_Gull' else '#606060'
        mean_val = species_data.mean()
        std_val = species_data.std()

        ax2.axvline(mean_val, color=darker_color, linestyle='--', alpha=0.8, linewidth=2)
        # Move text boxes to good position - using 0.85 instead of 0.9
        ax2.text(mean_val, 0.8, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                 transform=ax2.get_xaxis_transform(), color=darker_color, fontweight='bold',
                 ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                                               facecolor='white', alpha=0.8, edgecolor=darker_color))

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

    plt.savefig(os.path.join(output_dir, 'wing_wingtip_intensity_comparison_gray.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    return wing_stats, wingtip_stats


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
    """Main function to create intensity distribution comparisons with gray colors"""
    # Load data
    wing_data, wingtip_avg_data, wingtip_distribution = load_data()

    if wing_data is None or wingtip_distribution is None:
        print("Error: Could not load required data files.")
        return

    print("Creating wing vs wingtip intensity distribution comparison with gray colors...")
    wing_stats, wingtip_stats = create_intensity_distribution_comparison_gray(wing_data, wingtip_distribution)

    print(f"\nAnalysis complete! Chart saved to {output_dir}/")

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("\nWing Intensity Statistics:")
    print(wing_stats)
    print("\nWingtip Intensity Statistics:")
    print(wingtip_stats)


if __name__ == "__main__":
    main()publish