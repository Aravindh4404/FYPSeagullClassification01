import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from matplotlib.gridspec import GridSpec

# Define consistent color scheme for species
SPECIES_COLORS = {
    'Glaucous_Winged_Gull': '#3274A1',
    'Slaty_Backed_Gull': '#E1812C'
}

# Define consistent bin configuration
# INTENSITY_BINS = list(range(0, 256, 10))  # [0, 10, 20, ..., 250]

INTENSITY_BINS = list(range(0, 255, 15))  # [0, 10, 20, ..., 250, 260]

# Set plotting style for professional-looking visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Create output directory for visualizations
output_dir = "Intensity_Distribution_Comparison"
os.makedirs(output_dir, exist_ok=True)


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


def create_intensity_distribution_comparison(wing_data, wingtip_distribution):
    """Create side-by-side comparison of wing and wingtip intensity distributions with interpolation curves"""

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Common histogram parameters for consistency
    hist_params = {
        'bins': INTENSITY_BINS,
        'alpha': 0.6,
        'kde': True,
        'stat': 'density'  # Use density for better comparison
    }

    # 1. Wing Intensity Distribution
    sns.histplot(
        data=wing_data,
        x='mean_intensity',
        hue='species',
        ax=ax1,
        palette=SPECIES_COLORS,
        **hist_params
    )


    # Customize KDE line appearance
    for line in ax1.lines:
        line.set_linewidth(2.5)
        line.set_alpha(0.9)

    ax1.set_title('Wing Intensity Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')
    ax1.legend(title='Species', labels=['Glaucous-winged Gull', 'Slaty-backed Gull'])

    # Add mean lines for wing intensity
    for species in wing_data['species'].unique():
        species_data = wing_data[wing_data['species'] == species]['mean_intensity']
        mean_val = species_data.mean()
        color = SPECIES_COLORS[species]
        ax1.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax1.text(mean_val + 5, ax1.get_ylim()[1] * 0.9, f'Mean: {mean_val:.1f}',
                 rotation=90, color=color, fontweight='bold', ha='left', va='top')

    # 2. Wingtip Intensity Distribution
    sns.histplot(
        data=wingtip_distribution,
        x='mean_wingtip_intensity',
        hue='species',
        ax=ax2,
        palette=SPECIES_COLORS,
        **hist_params
    )

    # Customize KDE line appearance
    for line in ax2.lines:
        line.set_linewidth(2.5)
        line.set_alpha(0.9)

    ax2.set_title('Wingtip Intensity Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')
    ax2.legend(title='Species', labels=['Glaucous-winged Gull', 'Slaty-backed Gull'])

    # Add mean lines for wingtip intensity
    for species in wingtip_distribution['species'].unique():
        species_data = wingtip_distribution[wingtip_distribution['species'] == species]['mean_wingtip_intensity']
        mean_val = species_data.mean()
        color = SPECIES_COLORS[species]
        ax2.axvline(mean_val, color=color, linestyle='--', alpha=0.8, linewidth=2)
        ax2.text(mean_val + 5, ax2.get_ylim()[1] * 0.9, f'Mean: {mean_val:.1f}',
                 rotation=90, color=color, fontweight='bold', ha='left', va='top')

    # Ensure both plots have the same x-axis range for better comparison
    x_min = min(wing_data['mean_intensity'].min(), wingtip_distribution['mean_wingtip_intensity'].min())
    x_max = max(wing_data['mean_intensity'].max(), wingtip_distribution['mean_wingtip_intensity'].max())

    # Add some padding
    # Increase padding significantly to accommodate KDE tails
    x_range = x_max - x_min
    x_min -= x_range * 0.15  # Increased from 0.05 to 0.15
    x_max += x_range * 0.15  # Increased from 0.05 to 0.15

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

    # Add summary text below the plots
    summary_text = "Summary Statistics:\n"
    summary_text += "Wing Intensity - "
    summary_text += f"Slaty-backed: {wing_stats.loc['Slaty_Backed_Gull', 'mean']:.1f}±{wing_stats.loc['Slaty_Backed_Gull', 'std']:.1f} | "
    summary_text += f"Glaucous-winged: {wing_stats.loc['Glaucous_Winged_Gull', 'mean']:.1f}±{wing_stats.loc['Glaucous_Winged_Gull', 'std']:.1f}\n"
    summary_text += "Wingtip Intensity - "
    summary_text += f"Slaty-backed: {wingtip_stats.loc['Slaty_Backed_Gull', 'mean']:.1f}±{wingtip_stats.loc['Slaty_Backed_Gull', 'std']:.1f} | "
    summary_text += f"Glaucous-winged: {wingtip_stats.loc['Glaucous_Winged_Gull', 'mean']:.1f}±{wingtip_stats.loc['Glaucous_Winged_Gull', 'std']:.1f}"


    plt.savefig(os.path.join(output_dir, 'wing_wingtip_intensity_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return wing_stats, wingtip_stats


def create_overlaid_comparison(wing_data, wingtip_distribution):
    """Create an overlaid comparison showing both wing and wingtip intensities for each species"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Common histogram parameters
    hist_params = {
        'bins': INTENSITY_BINS,
        'alpha': 0.4,
        'kde': True,
        'stat': 'density'
    }

    # Species 1: Slaty-backed Gull
    slaty_wing = wing_data[wing_data['species'] == 'Slaty_Backed_Gull']['mean_intensity']
    slaty_wingtip = wingtip_distribution[wingtip_distribution['species'] == 'Slaty_Backed_Gull'][
        'mean_wingtip_intensity']

    # Plot wing intensity
    sns.histplot(x=slaty_wing, color=SPECIES_COLORS['Slaty_Backed_Gull'],
                 label='Wing', ax=ax1, **hist_params)

    # Plot wingtip intensity with different alpha and style
    sns.histplot(x=slaty_wingtip, color='red', alpha=0.4,
                 label='Wingtip', ax=ax1, bins=INTENSITY_BINS, kde=True,
                 stat='density')

    # Customize KDE lines for ax1
    kde_lines = [line for line in ax1.lines if line.get_linestyle() == '-']
    if len(kde_lines) >= 2:
        kde_lines[-1].set_linestyle('--')  # Make wingtip line dashed
        kde_lines[-1].set_linewidth(2.5)

    ax1.set_title('Slaty-backed Gull: Wing vs Wingtip Intensity', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mean Intensity (0-255)')
    ax1.set_ylabel('Density')
    ax1.legend()

    # Species 2: Glaucous-winged Gull
    glaucous_wing = wing_data[wing_data['species'] == 'Glaucous_Winged_Gull']['mean_intensity']
    glaucous_wingtip = wingtip_distribution[wingtip_distribution['species'] == 'Glaucous_Winged_Gull'][
        'mean_wingtip_intensity']

    # Plot wing intensity
    sns.histplot(x=glaucous_wing, color=SPECIES_COLORS['Glaucous_Winged_Gull'],
                 label='Wing', ax=ax2, **hist_params)

    # Plot wingtip intensity with different alpha and style
    sns.histplot(x=glaucous_wingtip, color='red', alpha=0.4,
                 label='Wingtip', ax=ax2, bins=INTENSITY_BINS, kde=True,
                 stat='density')

    # Customize KDE lines for ax2
    kde_lines = [line for line in ax2.lines if line.get_linestyle() == '-']
    if len(kde_lines) >= 2:
        kde_lines[-1].set_linestyle('--')  # Make wingtip line dashed
        kde_lines[-1].set_linewidth(2.5)

    ax2.set_title('Glaucous-winged Gull: Wing vs Wingtip Intensity', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mean Intensity (0-255)')
    ax2.set_ylabel('Density')
    ax2.legend()

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
    plt.suptitle('Within-Species Comparison: Wing vs Wingtip Intensity', fontsize=16, y=0.98)
    plt.savefig(os.path.join(output_dir, 'within_species_wing_wingtip_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    """Main function to create intensity distribution comparisons"""
    # Load data
    wing_data, wingtip_avg_data, wingtip_distribution = load_data()

    if wing_data is None or wingtip_distribution is None:
        print("Error: Could not load required data files.")
        return

    print("Creating wing vs wingtip intensity distribution comparison...")
    wing_stats, wingtip_stats = create_intensity_distribution_comparison(wing_data, wingtip_distribution)

    print("Creating within-species comparison...")
    create_overlaid_comparison(wing_data, wingtip_distribution)

    print(f"Analysis complete! Charts saved to {output_dir}/")

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print("\nWing Intensity Statistics:")
    print(wing_stats)
    print("\nWingtip Intensity Statistics:")
    print(wingtip_stats)


if __name__ == "__main__":
    main()