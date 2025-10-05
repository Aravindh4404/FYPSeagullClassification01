import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
import os
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
plt.rcParams['figure.figsize'] = (12, 8)
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


def create_wing_intensity_distribution(wing_data):
    """Create only the wing intensity distribution graph"""

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_title('Wing Intensity Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Mean Intensity (0-255)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)

    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        color = SPECIES_COLORS[species]
        display_name = species.replace('_', ' ').replace('Winged', '-winged').replace('Backed', '-backed')

        # Create histogram and interpolation
        counts, bins, patches = ax.hist(species_data, bins=INTENSITY_BINS, alpha=0.6,
                                        color=color, density=True, edgecolor='black',
                                        linewidth=0.8, label=display_name)

        bin_centers, hist_counts, x_smooth, y_smooth = create_histogram_interpolation(
            species_data, INTENSITY_BINS, color)
        ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.9)

        # Add mean line
        mean_val = species_data.mean()
        std_val = species_data.std()
        ax.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax.text(mean_val, 0.85, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                transform=ax.get_xaxis_transform(), color=color, fontweight='bold',
                ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                                              facecolor='white', alpha=0.8, edgecolor=color))

    add_intensity_reference(ax, INTENSITY_BINS)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 255)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wing_intensity_distribution_only.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def load_data():
    """Load and prepare wing intensity data"""
    try:
        # Load the original wing data
        wing_data = pd.read_csv('../Intensity_Results/wing_intensity_analysis.csv')
        print("Loaded wing intensity data")
        return wing_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def main():
    """Main function to create wing intensity distribution only"""
    # Load data
    wing_data = load_data()

    if wing_data is None:
        print("Error: Could not load wing intensity data file.")
        return

    print("Creating wing intensity distribution graph...")
    create_wing_intensity_distribution(wing_data)

    print(f"\nAnalysis complete! Graph saved to {output_dir}/wing_intensity_distribution_only.png")


if __name__ == "__main__":
    main()