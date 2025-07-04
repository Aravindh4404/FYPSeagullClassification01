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

# Define consistent bin configuration
INTENSITY_BINS = list(range(0, 255, 15))  # [0, 15, 30, ..., 240]

# Set plotting style for professional-looking visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

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


def create_intensity_distribution_comparison_fixed(wing_data, wingtip_distribution):
    """Create side-by-side comparison with proper interpolation curves"""

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Wing Intensity Distribution
    ax1.set_title('Wing Intensity Distribution', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')

    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS[species]

        # Create histogram bars with black edges
        ax1.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                 color=color, density=True, edgecolor='black', linewidth=0.8,
                 label=f'{species.replace("_", "-")}')

        # Create proper interpolation curve
        bin_centers, counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)

        # Plot interpolation curve
        ax1.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    # Add mean lines after plotting histograms to get proper y-limits
    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS[species]
        mean_val = species_data.mean()
        ax1.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        # Position mean labels with larger gap below title, horizontally aligned with the mean line
        ax1.text(mean_val, 0.98, f'Mean: {mean_val:.1f}',
                 transform=ax1.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=11)

    # Position legend further to the right
    ax1.legend(loc='upper right', bbox_to_anchor=(1.02, 0.98))

    # 2. Wingtip Intensity Distribution
    ax2.set_title('Wingtip Intensity Distribution', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')

    for species in wingtip_distribution['species'].unique():
        species_data = wingtip_distribution[wingtip_distribution['species'] == species]['mean_wingtip_intensity']
        color = SPECIES_COLORS[species]

        # Create histogram bars with black edges
        ax2.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                 color=color, density=True, edgecolor='black', linewidth=0.8,
                 label=f'{species.replace("_", "-")}')

        # Create proper interpolation curve
        bin_centers, counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)

        # Plot interpolation curve
        ax2.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    # Add mean lines after plotting histograms to get proper y-limits
    for species in wingtip_distribution['species'].unique():
        species_data = wingtip_distribution[wingtip_distribution['species'] == species]['mean_wingtip_intensity']
        color = SPECIES_COLORS[species]
        mean_val = species_data.mean()
        ax2.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        # Position mean labels with larger gap below title, horizontally aligned with the mean line
        ax2.text(mean_val, 0.98, f'Mean: {mean_val:.1f}',
                 transform=ax2.get_xaxis_transform(), color=color, fontweight='bold',
                 ha='center', va='top', fontsize=11)

    # Position legend further to the right
    ax2.legend(loc='upper right', bbox_to_anchor=(1.02, 0.98))

    # Ensure both plots have the same x-axis range for better comparison
    x_min = min(wing_data['mean_intensity'].min(), wingtip_distribution['mean_wingtip_intensity'].min())
    x_max = max(wing_data['mean_intensity'].max(), wingtip_distribution['mean_wingtip_intensity'].max())

    # Add some padding
    x_range = x_max - x_min
    x_min -= x_range * 0.05
    x_max += x_range * 0.05

    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    # Make y-axis ranges similar for better comparison
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, y_max)
    ax2.set_ylim(0, y_max)

    plt.tight_layout()

    # Calculate and display summary statistics
    wing_stats = wing_data.groupby('species')['mean_intensity'].agg(['mean', 'std', 'count'])
    wingtip_stats = wingtip_distribution.groupby('species')['mean_wingtip_intensity'].agg(['mean', 'std', 'count'])

    plt.savefig(os.path.join(output_dir, 'wing_wingtip_intensity_comparison_fixed.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return wing_stats, wingtip_stats


def create_overlaid_comparison_fixed(wing_data, wingtip_distribution):
    """Create an overlaid comparison with proper interpolation curves"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Species 1: Slaty-backed Gull
    slaty_wing = wing_data[wing_data['species'] == 'Slaty_Backed_Gull']['mean_intensity']
    slaty_wingtip = wingtip_distribution[wingtip_distribution['species'] == 'Slaty_Backed_Gull'][
        'mean_wingtip_intensity']

    # Plot histograms
    ax1.hist(slaty_wing, bins=INTENSITY_BINS, alpha=0.4,
             color=SPECIES_COLORS['Slaty_Backed_Gull'], density=True, label='Wing (histogram)')
    ax1.hist(slaty_wingtip, bins=INTENSITY_BINS, alpha=0.4,
             color='red', density=True, label='Wingtip (histogram)')

    # Plot interpolation curves
    _, _, x_wing, y_wing = create_histogram_interpolation(slaty_wing, INTENSITY_BINS,
                                                          SPECIES_COLORS['Slaty_Backed_Gull'])
    _, _, x_wingtip, y_wingtip = create_histogram_interpolation(slaty_wingtip, INTENSITY_BINS, 'red')

    ax1.plot(x_wing, y_wing, color=SPECIES_COLORS['Slaty_Backed_Gull'],
             linewidth=2.5, alpha=0.9, label='Wing (interpolation)')
    ax1.plot(x_wingtip, y_wingtip, color='red', linewidth=2.5,
             linestyle='--', alpha=0.9, label='Wingtip (interpolation)')

    ax1.set_title('Slaty-backed Gull: Wing vs Wingtip Intensity', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')
    # Move legend to upper right
    ax1.legend(loc='upper right')

    # Species 2: Glaucous-winged Gull
    glaucous_wing = wing_data[wing_data['species'] == 'Glaucous_Winged_Gull']['mean_intensity']
    glaucous_wingtip = wingtip_distribution[wingtip_distribution['species'] == 'Glaucous_Winged_Gull'][
        'mean_wingtip_intensity']

    # Plot histograms
    ax2.hist(glaucous_wing, bins=INTENSITY_BINS, alpha=0.4,
             color=SPECIES_COLORS['Glaucous_Winged_Gull'], density=True, label='Wing (histogram)')
    ax2.hist(glaucous_wingtip, bins=INTENSITY_BINS, alpha=0.4,
             color='red', density=True, label='Wingtip (histogram)')

    # Plot interpolation curves
    _, _, x_wing, y_wing = create_histogram_interpolation(glaucous_wing, INTENSITY_BINS,
                                                          SPECIES_COLORS['Glaucous_Winged_Gull'])
    _, _, x_wingtip, y_wingtip = create_histogram_interpolation(glaucous_wingtip, INTENSITY_BINS, 'red')

    ax2.plot(x_wing, y_wing, color=SPECIES_COLORS['Glaucous_Winged_Gull'],
             linewidth=2.5, alpha=0.9, label='Wing (interpolation)')
    ax2.plot(x_wingtip, y_wingtip, color='red', linewidth=2.5,
             linestyle='--', alpha=0.9, label='Wingtip (interpolation)')

    ax2.set_title('Glaucous-winged Gull: Wing vs Wingtip Intensity', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')
    # Move legend to upper right
    ax2.legend(loc='upper right')

    # Ensure consistent x-axis ranges
    all_values = np.concatenate([slaty_wing, slaty_wingtip, glaucous_wing, glaucous_wingtip])
    x_min, x_max = all_values.min(), all_values.max()
    x_range = x_max - x_min
    x_min -= x_range * 0.05
    x_max += x_range * 0.05

    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    # Make y-axis ranges consistent
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, y_max)
    ax2.set_ylim(0, y_max)

    plt.tight_layout()
    plt.suptitle('Within-Species Comparison: Wing vs Wingtip Intensity (Fixed Interpolation)', fontsize=16, y=0.98)
    plt.savefig(os.path.join(output_dir, 'within_species_wing_wingtip_comparison_fixed.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Alternative: If you want KDE but with better control
def create_controlled_kde_comparison(wing_data, wingtip_distribution):
    """Create comparison using controlled KDE that aligns better with data range"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Wing Intensity Distribution
    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS[species]

        # Plot histogram
        ax1.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                 color=color, density=True, label=f'{species.replace("_", "-")}')

        # Create controlled KDE
        kde = stats.gaussian_kde(species_data)
        x_range = np.linspace(species_data.min(), species_data.max(), 200)
        kde_values = kde(x_range)

        ax1.plot(x_range, kde_values, color=color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    ax1.set_title('Wing Intensity Distribution (Controlled KDE)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')
    # Move legend to upper right
    ax1.legend(loc='upper right')

    # 2. Wingtip Intensity Distribution
    for species in wingtip_distribution['species'].unique():
        species_data = wingtip_distribution[wingtip_distribution['species'] == species]['mean_wingtip_intensity']
        color = SPECIES_COLORS[species]

        # Plot histogram
        ax2.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                 color=color, density=True, label=f'{species.replace("_", "-")}')

        # Create controlled KDE
        kde = stats.gaussian_kde(species_data)
        x_range = np.linspace(species_data.min(), species_data.max(), 200)
        kde_values = kde(x_range)

        ax2.plot(x_range, kde_values, color=color, linewidth=2.5,
                 alpha=0.9, linestyle='-')

    ax2.set_title('Wingtip Intensity Distribution (Controlled KDE)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')
    # Move legend to upper right
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'controlled_kde_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


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
    """Main function to create intensity distribution comparisons with fixed interpolation"""
    # Load data
    wing_data, wingtip_avg_data, wingtip_distribution = load_data()

    if wing_data is None or wingtip_distribution is None:
        print("Error: Could not load required data files.")
        return

    print("Creating wing vs wingtip intensity distribution comparison with proper interpolation...")
    wing_stats, wingtip_stats = create_intensity_distribution_comparison_fixed(wing_data, wingtip_distribution)

    print(f"Analysis complete! Charts saved to {output_dir}/")

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("\nWing Intensity Statistics:")
    print(wing_stats)
    print("\nWingtip Intensity Statistics:")
    print(wingtip_stats)


if __name__ == "__main__":
    main()