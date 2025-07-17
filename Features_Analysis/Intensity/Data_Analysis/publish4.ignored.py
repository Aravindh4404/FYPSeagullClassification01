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
plt.style.use('default')  # Changed from seaborn-v0_8-whitegrid to default
plt.rcParams['figure.figsize'] = (12, 7)
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
    if len(x_points) >= 2:
        interp_func = interpolate.interp1d(x_points, y_points,
                                           kind='linear',
                                           bounds_error=False,
                                           fill_value=0)

        # Generate smooth curve points within data range
        x_smooth = np.linspace(x_points.min(), x_points.max(), 200)
        y_smooth = interp_func(x_smooth)

        # Ensure no negative values
        y_smooth = np.maximum(y_smooth, 0)
    else:
        # If we have insufficient data points, return original data
        x_smooth = x_points
        y_smooth = y_points

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


def create_intensity_distribution_comparison_with_reference(wing_data, wingtip_distribution):
    """Create side-by-side comparison with intensity reference bars"""

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    # 1. Wing Intensity Distribution
    ax1.set_title('Wing Intensity Distribution', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')

    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS.get(species, '#808080')  # Default gray if species not found

        # Create display name for legend
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Create histogram bars with clear edges and consistent width
        counts, bins, patches = ax1.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        # Create proper interpolation curve
        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)

        # Plot interpolation curve only if we have enough data
        if len(x_smooth) > 1:
            ax1.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                     alpha=0.9, linestyle='-')

    # Add mean lines and statistics
    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS.get(species, '#808080')
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

    # 2. Wingtip Intensity Distribution
    ax2.set_title('Wingtip Intensity Distribution', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')

    # Check which column to use for wingtip intensity
    if 'filtered_mean_wingtip_intensity' in wingtip_distribution.columns:
        intensity_column = 'filtered_mean_wingtip_intensity'
    else:
        intensity_column = 'mean_wingtip_intensity'

    for species in wingtip_distribution['species'].unique():
        species_data = wingtip_distribution[wingtip_distribution['species'] == species][intensity_column]
        color = SPECIES_COLORS.get(species, '#808080')

        # Create display name for legend
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Create histogram bars with clear edges and consistent width
        counts, bins, patches = ax2.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                         color=color, density=True, edgecolor='black',
                                         linewidth=0.8, label=display_name)

        # Create proper interpolation curve
        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)

        # Plot interpolation curve only if we have enough data
        if len(x_smooth) > 1:
            ax2.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                     alpha=0.9, linestyle='-')

    # Add mean lines and statistics
    for species in wingtip_distribution['species'].unique():
        species_data = wingtip_distribution[wingtip_distribution['species'] == species][intensity_column]
        color = SPECIES_COLORS.get(species, '#808080')
        mean_val = species_data.mean()
        std_val = species_data.std()

        ax2.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax2.text(mean_val, 0.8, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
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

    # Ensure both plots show the full 0-255 range
    ax1.set_xlim(0, 255)
    ax2.set_xlim(0, 255)

    # Add bin range information to the plot
    fig.text(0.5, 0.02, f'Bin Width: {BIN_WIDTH} units | Intensity reference shows actual brightness values',
             ha='center', va='bottom', fontsize=8, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Calculate and display summary statistics
    wing_stats = wing_data.groupby('species')['mean_intensity'].agg(['mean', 'std', 'count'])
    wingtip_stats = wingtip_distribution.groupby('species')[intensity_column].agg(['mean', 'std', 'count'])

    plt.savefig(os.path.join(output_dir, 'wing_wingtip_intensity_comparison_with_reference.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    return wing_stats, wingtip_stats


def load_data():
    """Load and prepare all necessary datasets"""
    try:
        # First, try to load wing data
        wing_data = pd.read_csv('../Intensity_Results/wing_intensity_analysis.csv')
        print("Successfully loaded wing data")

        # Try to load filtered wingtip data first, then fall back to original
        try:
            wingtip_distribution = pd.read_csv(
                '../Wingtip_Intensity_Distribution_Filtered220/wingtip_intensity_distribution_filtered.csv')
            print("Successfully loaded filtered wingtip distribution data")
        except FileNotFoundError:
            try:
                wingtip_distribution = pd.read_csv(
                    '../Wingtip_Intensity_Distribution_Filtered/wingtip_intensity_distribution_filtered.csv')
                print("Successfully loaded filtered wingtip distribution data (alternative path)")
            except FileNotFoundError:
                wingtip_distribution = pd.read_csv(
                    '../Wingtip_Intensity_Distribution/wingtip_intensity_distribution.csv')
                print("Successfully loaded original wingtip distribution data")

        # Try to load wingtip averages (optional)
        try:
            wingtip_avg_data = pd.read_csv(
                '../Wingtip_Intensity_Distribution_Filtered220/wingtip_intensity_averages_filtered.csv')
            print("Successfully loaded filtered wingtip averages data")
        except FileNotFoundError:
            try:
                wingtip_avg_data = pd.read_csv(
                    '../Wingtip_Intensity_Distribution_Filtered/wingtip_intensity_averages_filtered.csv')
                print("Successfully loaded filtered wingtip averages data (alternative path)")
            except FileNotFoundError:
                try:
                    wingtip_avg_data = pd.read_csv('../Wingtip_Intensity_Distribution/wingtip_intensity_averages.csv')
                    print("Successfully loaded original wingtip averages data")
                except FileNotFoundError:
                    wingtip_avg_data = None
                    print("Could not load wingtip averages data, continuing without it")

        print("Data loading completed")
        return wing_data, wingtip_avg_data, wingtip_distribution

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check that the following files exist:")
        print("  - ../Intensity_Results/wing_intensity_analysis.csv")
        print("  - ../Wingtip_Intensity_Distribution_Filtered220/wingtip_intensity_distribution_filtered.csv")
        print("  - OR ../Wingtip_Intensity_Distribution/wingtip_intensity_distribution.csv")
        return None, None, None


def main():
    """Main function to create intensity distribution comparisons with intensity reference"""
    # Load data
    wing_data, wingtip_avg_data, wingtip_distribution = load_data()

    if wing_data is None or wingtip_distribution is None:
        print("Error: Could not load required data files.")
        return

    # Print data info for debugging
    print("\n=== DATA INFO ===")
    print(f"Wing data shape: {wing_data.shape}")
    print(f"Wing data columns: {wing_data.columns.tolist()}")
    print(f"Wingtip distribution shape: {wingtip_distribution.shape}")
    print(f"Wingtip distribution columns: {wingtip_distribution.columns.tolist()}")
    print(f"Species in wing data: {wing_data['species'].unique()}")
    print(f"Species in wingtip data: {wingtip_distribution['species'].unique()}")

    print("\nCreating wing vs wingtip intensity distribution comparison with intensity reference...")

    try:
        wing_stats, wingtip_stats = create_intensity_distribution_comparison_with_reference(wing_data,
                                                                                            wingtip_distribution)

        print(f"\nAnalysis complete! Chart saved to {output_dir}/")

        # Print summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print("\nWing Intensity Statistics:")
        print(wing_stats)
        print("\nWingtip Intensity Statistics:")
        print(wingtip_stats)

    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()