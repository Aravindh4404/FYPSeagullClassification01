import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob

# Directory where the texture property files are stored
analysis_folder = "R16_P128_Muniform"


DATA_DIR = os.path.join("Outputs/LBP_Analysis/", analysis_folder, "region_analysis")
RESULTS_DIR = os.path.join("Outputs/LBP_Analysis/", analysis_folder, "comparison_results") # Directory to save the visualization results

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_texture_data(file_path):
    """Load texture data from CSV files"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        # Handle text files by parsing them
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Skip the first line (title) and parse the table
        return pd.read_csv(file_path, skiprows=1, delim_whitespace=True)


def find_texture_files(directory):
    """Find all texture property files in the directory"""
    csv_files = glob.glob(os.path.join(directory, "*_texture_properties.csv"))
    txt_files = glob.glob(os.path.join(directory, "*_texture_properties.txt"))
    return csv_files + txt_files


def extract_region_name(file_path):
    """Extract region name from file path"""
    base_name = os.path.basename(file_path)
    region = base_name.split('_texture_properties')[0]
    return region


def calculate_significance(data, property_name):
    """Calculate statistical significance for a property between species
    Returns p-value (if sample sizes were available, otherwise returns None)"""
    # In real data analysis you would use actual measurements for statistical tests
    # Here we're just demonstrating the concept
    return None  # Would return a p-value if we had sample data


def create_bar_chart(data, region_name, output_dir):
    """Create bar chart for texture properties with pseudo error bars"""
    # Get properties and species
    properties = data['Property'].values
    species = data.columns[1:]

    # Set up the figure
    fig, axs = plt.subplots(len(properties), 1, figsize=(10, 3 * len(properties)))

    # If there's only one property, wrap axs in a list
    if len(properties) == 1:
        axs = [axs]

    # Colors for the species
    colors = ['blue', 'orange']

    # For each property, create a bar chart
    for i, prop in enumerate(properties):
        # Get values for this property
        values = data.loc[data['Property'] == prop, species].values[0]

        # Create bars
        x = np.arange(len(species))
        bars = axs[i].bar(x, values, width=0.6, color=colors[:len(species)])

        # Add pseudo error bars (10% of value as an example)
        # In a real analysis, you would calculate actual standard errors
        error = [v * 0.1 for v in values]  # 10% of each value as example error
        axs[i].errorbar(x, values, yerr=error, fmt='none', ecolor='black', capsize=5)

        # Add value labels on top of bars
        for j, v in enumerate(values):
            axs[i].text(j, v + error[j], f"{v:.4f}", ha='center', va='bottom')

        # Calculate p-value for statistical significance (if we had sample data)
        p_value = calculate_significance(data, prop)
        sig_text = f"p = {p_value:.4f}" if p_value is not None else ""

        # Add titles and labels
        axs[i].set_title(f"{prop.capitalize()} Comparison for {region_name} {sig_text}")
        axs[i].set_ylabel(prop.capitalize())
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(species)

        # Highlight differences
        if len(values) >= 2:
            diff_pct = abs(values[1] - values[0]) / max(values) * 100
            diff_text = f"Diff: {diff_pct:.1f}%"
            axs[i].text(0.5, 0.05, diff_text, transform=axs[i].transAxes, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{region_name}_texture_comparison.png"), dpi=300)
    plt.close()

    return fig


def create_property_comparison(all_data, output_dir):
    """Create comparative visualization across all regions for each property"""
    # Get all unique properties and regions
    all_properties = set()
    all_regions = []
    for region, data in all_data.items():
        all_regions.append(region)
        all_properties.update(data['Property'].values)

    all_properties = sorted(list(all_properties))
    all_regions = sorted(all_regions)
    species = all_data[all_regions[0]].columns[1:]  # Get species names

    # For each property, create a comparison across regions
    for prop in all_properties:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Width of each bar
        bar_width = 0.35

        # For each region, get the property values
        region_values = []
        for region in all_regions:
            if prop in all_data[region]['Property'].values:
                values = all_data[region].loc[all_data[region]['Property'] == prop, species].values[0]
                region_values.append(values)
            else:
                region_values.append([0] * len(species))

        # Create positions for bars
        x = np.arange(len(all_regions))

        # Create grouped bars for each species
        for i, s in enumerate(species):
            values = [rv[i] for rv in region_values]
            ax.bar(x + i * bar_width - bar_width / 2, values, bar_width, label=s, alpha=0.7)

        # Add labels and title
        ax.set_xlabel('Region')
        ax.set_ylabel(prop.capitalize())
        ax.set_title(f'{prop.capitalize()} Comparison Across Regions')
        ax.set_xticks(x)
        ax.set_xticklabels(all_regions)
        ax.legend()

        # Add value labels
        for i, s in enumerate(species):
            for j, region in enumerate(all_regions):
                value = region_values[j][i]
                ax.text(j + i * bar_width - bar_width / 2, value, f"{value:.2f}",
                        ha='center', va='bottom', fontsize=8, rotation=90)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prop}_across_regions.png"), dpi=300)
        plt.close()


def create_discriminative_power_chart(all_data, output_dir):
    """Create a chart showing which properties best discriminate between species across regions"""
    all_regions = sorted(list(all_data.keys()))
    all_properties = sorted(list(set(prop for region in all_data.values() for prop in region['Property'].values)))
    species = all_data[all_regions[0]].columns[1:]

    # Calculate difference percentage for each property in each region
    diff_data = {}
    for region in all_regions:
        diff_data[region] = {}
        for prop in all_properties:
            if prop in all_data[region]['Property'].values:
                values = all_data[region].loc[all_data[region]['Property'] == prop, species].values[0]
                if len(values) >= 2:
                    # Calculate normalized difference
                    diff_pct = abs(values[1] - values[0]) / max(values) * 100
                    diff_data[region][prop] = diff_pct
                else:
                    diff_data[region][prop] = 0
            else:
                diff_data[region][prop] = 0

    # Create heatmap of differences
    diff_df = pd.DataFrame(columns=all_properties, index=all_regions)
    for region in all_regions:
        for prop in all_properties:
            diff_df.loc[region, prop] = diff_data[region].get(prop, 0)

    # Convert diff_df to float to ensure numeric data
    diff_df = diff_df.astype(float)

    plt.figure(figsize=(12, 8))
    plt.imshow(diff_df.values, cmap='viridis', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Difference (%)')

    # Add labels
    plt.xticks(np.arange(len(all_properties)), all_properties, rotation=45, ha='right')
    plt.yticks(np.arange(len(all_regions)), all_regions)

    # Add values in cells
    for i in range(len(all_regions)):
        for j in range(len(all_properties)):
            value = diff_df.iloc[i, j]
            plt.text(j, i, f"{value:.1f}%", ha='center', va='center',
                     color='white' if value > 30 else 'black')

    plt.title('Discriminative Power (% Difference Between Species)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "discriminative_power_heatmap.png"), dpi=300)
    plt.close()

    # Create a bar chart of the most discriminative features
    flat_diffs = []
    for region in all_regions:
        for prop in all_properties:
            if diff_data[region].get(prop, 0) > 0:
                flat_diffs.append((region, prop, diff_data[region][prop]))

    # Sort by difference
    flat_diffs.sort(key=lambda x: x[2], reverse=True)

    # Plot top 10 most discriminative features
    top_n = min(10, len(flat_diffs))

    plt.figure(figsize=(12, 6))
    labels = [f"{x[0]}-{x[1]}" for x in flat_diffs[:top_n]]
    values = [x[2] for x in flat_diffs[:top_n]]

    bars = plt.bar(range(len(values)), values)

    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{values[i]:.1f}%", ha='center', va='bottom')

    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Difference (%)')
    plt.title('Top Discriminative Features (Region-Property Pairs)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_discriminative_features.png"), dpi=300)
    plt.close()


def create_summary_report(all_data, output_dir):
    """Create a summary text report of the analysis"""
    with open(os.path.join(output_dir, "texture_analysis_summary.txt"), 'w') as f:
        f.write("BIRD TEXTURE ANALYSIS SUMMARY\n")
        f.write("============================\n\n")

        # Summary of regions and properties analyzed
        all_regions = sorted(list(all_data.keys()))
        all_properties = sorted(list(set(prop for region in all_data.values() for prop in region['Property'].values)))
        species = all_data[all_regions[0]].columns[1:]

        f.write(f"Species analyzed: {', '.join(species)}\n")
        f.write(f"Regions analyzed: {', '.join(all_regions)}\n")
        f.write(f"Properties analyzed: {', '.join(all_properties)}\n\n")

        # Calculate most discriminative features
        most_discriminative = {}
        for region in all_regions:
            most_discriminative[region] = {"property": "", "diff_pct": 0}
            for prop in all_data[region]['Property'].values:
                values = all_data[region].loc[all_data[region]['Property'] == prop, species].values[0]
                if len(values) >= 2:
                    diff_pct = abs(values[1] - values[0]) / max(values) * 100
                    if diff_pct > most_discriminative[region]["diff_pct"]:
                        most_discriminative[region]["property"] = prop
                        most_discriminative[region]["diff_pct"] = diff_pct

        # Write most discriminative features for each region
        f.write("MOST DISCRIMINATIVE FEATURES BY REGION\n")
        f.write("------------------------------------\n")
        for region in all_regions:
            prop = most_discriminative[region]["property"]
            diff = most_discriminative[region]["diff_pct"]
            if diff > 0:
                f.write(f"{region}: {prop} (difference: {diff:.2f}%)\n")

                # Get the actual values
                values = all_data[region].loc[all_data[region]['Property'] == prop, species].values[0]
                for i, s in enumerate(species):
                    f.write(f"  {s}: {values[i]:.4f}\n")
                f.write("\n")

        # Write overall comparison
        f.write("\nOVERALL SPECIES COMPARISON\n")
        f.write("-------------------------\n")

        # Analyze distinctive characteristics of each species
        for s_idx, s in enumerate(species):
            f.write(f"{s} distinctive characteristics:\n")

            for region in all_regions:
                region_props = []
                for prop in all_data[region]['Property'].values:
                    values = all_data[region].loc[all_data[region]['Property'] == prop, species].values[0]
                    if len(values) >= 2:
                        # If this species has a notably higher or lower value
                        other_idx = 1 - s_idx  # Index of the other species (assuming 2 species)
                        diff = values[s_idx] - values[other_idx]
                        if abs(diff) / max(values) > 0.1:  # If difference is >10%
                            comparison = "higher" if diff > 0 else "lower"
                            region_props.append(f"{prop} ({comparison} by {abs(diff / values[other_idx] * 100):.1f}%)")

                if region_props:
                    f.write(f"  {region}: {', '.join(region_props)}\n")

            f.write("\n")

        # Write conclusion
        f.write("\nCONCLUSION\n")
        f.write("----------\n")

        # Find overall most discriminative region
        best_region = max(most_discriminative.items(), key=lambda x: x[1]["diff_pct"])
        f.write(f"The most discriminative region for species identification is the {best_region[0]}, ")
        f.write(f"with a {best_region[1]['property']} difference of {best_region[1]['diff_pct']:.2f}%.\n\n")

        # Suggest texture properties for identification
        f.write("For species identification, the following texture properties show the most promise:\n")
        flat_diffs = []
        for region in all_regions:
            for prop in all_properties:
                if prop in all_data[region]['Property'].values:
                    values = all_data[region].loc[all_data[region]['Property'] == prop, species].values[0]
                    if len(values) >= 2:
                        diff_pct = abs(values[1] - values[0]) / max(values) * 100
                        flat_diffs.append((region, prop, diff_pct))

        flat_diffs.sort(key=lambda x: x[2], reverse=True)
        for i, (region, prop, diff) in enumerate(flat_diffs[:5]):
            f.write(f"{i + 1}. {region}'s {prop} ({diff:.2f}% difference)\n")


def main():
    """Main function to process texture files and create visualizations"""
    # Find all texture property files
    texture_files = find_texture_files(DATA_DIR)

    if not texture_files:
        print(f"No texture property files found in {DATA_DIR}. Please check the directory.")
        return

    print(f"Found {len(texture_files)} texture property files.")

    # Process each file
    all_data = {}
    for file_path in texture_files:
        region_name = extract_region_name(file_path)
        data = load_texture_data(file_path)
        all_data[region_name] = data

        print(f"Processing {region_name} texture properties...")
        # Create individual bar charts for each region
        create_bar_chart(data, region_name, RESULTS_DIR)

    # Create property comparisons across regions
    print("Creating cross-region property comparisons...")
    create_property_comparison(all_data, RESULTS_DIR)

    # Create discriminative power analysis
    print("Analyzing discriminative power of texture properties...")
    create_discriminative_power_chart(all_data, RESULTS_DIR)

    # Create summary report
    print("Creating summary report...")
    create_summary_report(all_data, RESULTS_DIR)

    print("\nAnalysis complete! Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()