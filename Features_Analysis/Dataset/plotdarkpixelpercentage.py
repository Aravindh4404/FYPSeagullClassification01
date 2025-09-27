import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
# Use only dark blue and orange colors
DARK_BLUE = '#1f77b4'  # Default matplotlib blue
DARK_ORANGE = '#ff7f0e'  # Default matplotlib orange


def get_blue_orange_colors(n):
    """Generate alternating dark blue and orange colors"""
    colors = []
    for i in range(n):
        if i % 2 == 0:
            colors.append(DARK_BLUE)
        else:
            colors.append(DARK_ORANGE)
    return colors


def load_and_prepare_data(file_path):
    """Load the Excel file and prepare data for plotting"""
    try:
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Check if required columns exist
        required_cols = ['species', 'method3_percentage']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            return None

        # Clean species names for better display
        df['species_clean'] = df['species'].str.replace('_', ' ').str.title()

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_box_violin_plots(df, output_dir="plots"):
    """Create box plot and violin plot visualization"""

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # DEBUG: Check data before plotting
    print("\n🔍 DEBUGGING DATA BEFORE PLOTTING:")
    print(f"Data shape: {df.shape}")
    print(f"method3_percentage column type: {df['method3_percentage'].dtype}")
    print(f"Min value in method3_percentage: {df['method3_percentage'].min()}")
    print(f"Max value in method3_percentage: {df['method3_percentage'].max()}")
    print(f"Any NaN values: {df['method3_percentage'].isna().sum()}")
    print(f"Unique species: {df['species_clean'].unique()}")
    print("\nFirst few rows:")
    print(df[['species_clean', 'method3_percentage']].head(10))

    # Clean data: remove any NaN values and ensure numeric type
    df_clean = df.dropna(subset=['method3_percentage', 'species_clean']).copy()
    df_clean['method3_percentage'] = pd.to_numeric(df_clean['method3_percentage'], errors='coerce')
    df_clean = df_clean.dropna(subset=['method3_percentage'])

    # Ensure percentage data is within valid bounds (0-100%)
    df_clean['method3_percentage'] = df_clean['method3_percentage'].clip(0, 100)

    print(f"\nAfter cleaning: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Main title
    fig.suptitle('Wingtip Dark Pixel Analysis: Method 1 (< Mean Wing Intensity) by Species',
                 fontsize=14, fontweight='bold', y=0.95)

    # Get colors for species
    n_species = len(df_clean['species_clean'].unique())
    colors = get_blue_orange_colors(n_species)

    # Set meaningful y-axis limits for percentages (0-100% range)
    y_min = max(0, df_clean['method3_percentage'].min() - 2)
    y_max = min(100, df_clean['method3_percentage'].max() + 2)

    # If data is very close to 100%, allow a small extension for readability
    if df_clean['method3_percentage'].max() > 98:
        y_max = 101

    # If data is very close to 0%, allow a small extension for readability
    if df_clean['method3_percentage'].min() < 2:
        y_min = -1

    # Plot 1: Box plot
    sns.boxplot(data=df_clean, x='species_clean', y='method3_percentage', ax=ax1, palette=colors)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Dark Pixel Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Box Plot: Distribution Summary\n(Shows median, quartiles, and outliers)',
                  fontsize=11, fontweight='bold', pad=20)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Violin plot with proper constraints
    # Use cut=0 to prevent density estimation beyond data range
    sns.violinplot(data=df_clean, x='species_clean', y='method3_percentage', ax=ax2,
                   palette=colors, cut=0, inner='box')

    # Force y-axis limits to stay within percentage bounds
    ax2.set_ylim(0, 100)  # Hard limit for percentage data

    ax2.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dark Pixel Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Violin Plot: Full Distribution Shape\n(Shows probability density at each value)',
                  fontsize=11, fontweight='bold', pad=20)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add horizontal reference lines for context
    for ax in [ax1, ax2]:
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(0.02, 50.5, '50%', transform=ax.get_yaxis_transform(),
                fontsize=9, alpha=0.7, color='gray')

    plt.tight_layout()

    # Save the plot
    output_path = Path(output_dir) / "wingtip_box_violin_plots_fixed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Fixed plot saved as: {output_path}")

    plt.show()

    # Calculate and return summary statistics using cleaned data
    summary_stats = df_clean.groupby('species_clean')['method3_percentage'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)

    return summary_stats


def print_interpretation_guide(df, summary_stats):
    """Print interpretation guide for the plots"""

    print("\n" + "=" * 80)
    print("PLOT INTERPRETATION GUIDE")
    print("=" * 80)

    print("\n📊 BOX PLOT INTERPRETATION:")
    print("-" * 50)
    print("• Box: Shows the middle 50% of data (25th to 75th percentile)")
    print("• Line inside box: Median (50th percentile)")
    print("• Whiskers: Extend to show data range (typically 1.5 × IQR from box)")
    print("• Dots beyond whiskers: Outliers (unusually high or low values)")
    print("• Box width: All boxes same width (doesn't indicate sample size)")

    print("\n🎻 VIOLIN PLOT INTERPRETATION:")
    print("-" * 50)
    print("• Width at each height: Shows how many data points exist at that value")
    print("• Wider sections: More common values (higher density)")
    print("• Narrow sections: Less common values (lower density)")
    print("• Shape reveals distribution: symmetric, skewed, bimodal, etc.")
    print("• Inner box: Shows quartiles and median (like a mini box plot)")
    print("• FIXED: Now properly constrained to 0-100% range for percentage data")

    print("\n📈 WHAT TO LOOK FOR:")
    print("-" * 50)
    print("• Central tendency: Which species have higher/lower median dark pixels?")
    print("• Variability: Which species show more consistent vs. variable results?")
    print("• Distribution shape: Are values normally distributed or skewed?")
    print("• Outliers: Are there unusual specimens that deviate from the pattern?")
    print("• Sample size effects: Remember that smaller samples may look different")

    print(f"\n📊 SUMMARY STATISTICS:")
    print("-" * 50)
    print(summary_stats)

    print(f"\n🏆 SPECIES RANKING (by median dark pixel percentage):")
    print("-" * 50)
    ranked_species = summary_stats.sort_values('median', ascending=False)
    for i, (species, stats) in enumerate(ranked_species.iterrows(), 1):
        print(f"{i:2d}. {species:20s}: {stats['median']:6.2f}% median ({int(stats['count']):3d} samples)")

    print(f"\n⚠️  INTERPRETATION NOTES:")
    print("-" * 50)
    print("• Method 1 counts pixels darker than the mean wing intensity")
    print("• Higher percentages = more dark pixels relative to wing brightness")
    print("• Species differences may reflect:")
    print("  - Natural wing coloration patterns")
    print("  - Sexual dimorphism (if mixed sexes)")
    print("  - Age-related changes in wing patterns")
    print("  - Lighting/photography conditions")
    print("  - Wing wear or damage")

    # Identify interesting patterns
    print(f"\n🔍 NOTABLE PATTERNS:")
    print("-" * 50)

    # Find species with highest variability
    cv_data = []
    for species, stats in summary_stats.iterrows():
        cv = (stats['std'] / stats['mean']) * 100 if stats['mean'] > 0 else 0
        cv_data.append((species, cv, stats['std']))

    cv_data.sort(key=lambda x: x[1], reverse=True)
    print(f"• Most variable species: {cv_data[0][0]} (CV = {cv_data[0][1]:.1f}%)")
    print(f"• Most consistent species: {cv_data[-1][0]} (CV = {cv_data[-1][1]:.1f}%)")

    # Find species with extreme values
    highest_median = ranked_species.iloc[0]
    lowest_median = ranked_species.iloc[-1]
    print(f"• Highest median dark pixels: {highest_median.name} ({highest_median['median']:.2f}%)")
    print(f"• Lowest median dark pixels: {lowest_median.name} ({lowest_median['median']:.2f}%)")


def main():
    """Main function to run the simplified analysis"""

    # File path - update this to match your file location
    file_path = "cleaned_for_plotting.xlsx"  # Update this path as needed

    print("Loading data...")
    df = load_and_prepare_data(file_path)

    if df is None:
        print("Failed to load data. Please check the file path and format.")
        return

    print(f"Successfully loaded data with {len(df)} records for {df['species'].nunique()} species")

    # Create box and violin plots
    print("\nCreating fixed box plot and violin plot...")
    summary_stats = create_box_violin_plots(df)

    # Print interpretation guide
    print_interpretation_guide(df, summary_stats)

    print("\n✅ Analysis complete! Check the 'plots' directory for output files.")


if __name__ == "__main__":
    main()