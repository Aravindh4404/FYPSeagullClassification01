import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from pathlib import Path
# Use these paths in your code
import pandas as pd
import os

# Get project root directory using parent relationships
def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    # Go up until we find the project root (where your main code/git repository is)
    project_root = current_file.parent
    while not (project_root / '.git').exists():  # Using .git as marker for project root
        project_root = project_root.parent
    return project_root

# Set up paths
PROJECT_ROOT = get_project_root()
INPUT_DIR = PROJECT_ROOT / "Features_Analysis" / "Intensity" / "Wingtip_Intensity_Distribution"
output_dir = PROJECT_ROOT / "Features_Analysis" / "Intensity" / "Wingtip_Intensity_Analysis_Results"

# Input/output files
input_csv = INPUT_DIR / "wingtip_intensity_distribution.csv"


os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(input_csv)

# Load the data
try:
    df = pd.read_csv(input_csv)
    print("Data loaded successfully. First few rows:")
    print(df.head())
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Check if required columns exist
required_columns = ['species', 'mean_wingtip_intensity']
if not all(col in df.columns for col in required_columns):
    print("Error: CSV file doesn't contain required columns.")
    print(f"Expected columns: {required_columns}")
    print(f"Found columns: {df.columns.tolist()}")
    exit()

# Clean species names (correcting the typo in sample data)
df['species'] = df['species'].str.replace('Staty_Backed_Gu1', 'Slaty_Backed_Gull')

# Calculate basic statistics
stats_df = df.groupby('species')['mean_wingtip_intensity'].agg(['mean', 'std', 'count', 'min', 'max'])
print("\nBasic statistics per species:")
print(stats_df)

# Perform t-test
slaty = df[df['species'] == 'Slaty_Backed_Gull']['mean_wingtip_intensity']
glaucous = df[df['species'] == 'Glaucous_Winged_Gull']['mean_wingtip_intensity']

t_stat, p_value = stats.ttest_ind(slaty, glaucous, equal_var=False)

# Create results dictionary
results = {
    't_statistic': t_stat,
    'p_value': p_value,
    'interpretation': "Significantly different" if p_value < 0.05 else "Not significantly different"
}

# Combine all results into one DataFrame
full_results = pd.concat([
    stats_df,
    pd.DataFrame(results, index=['Statistical Test'])
])

# Save all results to CSV
results_csv_path = os.path.join(output_dir, 'wingtip_intensity_full_results.csv')
full_results.to_csv(results_csv_path)
print(f"\nSaved full results to: {results_csv_path}")

# Plot 1: Distribution of Wingtip Mean Intensity by Species
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='species', y='mean_wingtip_intensity',
            palette=['#1f77b4', '#ff7f0e'])
plt.title(f'Distribution of Wingtip Mean Intensity by Species\n(p-value = {p_value:.4f})', fontsize=14)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Mean Intensity (0-255 scale)', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
dist_plot_path = os.path.join(output_dir, 'wingtip_intensity_distribution.png')
plt.savefig(dist_plot_path, dpi=300)
print(f"\nSaved distribution plot to: {dist_plot_path}")

# Plot 2: Mean Wingtip Intensity with Error Bars (Standard Deviation)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='species', y='mean_wingtip_intensity',
            palette=['#1f77b4', '#ff7f0e'], ci='sd', capsize=0.1)
plt.title('Mean Wingtip Intensity by Species with Standard Deviation', fontsize=14)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Mean Intensity (0-255 scale)', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
mean_std_plot_path = os.path.join(output_dir, 'mean_wingtip_intensity_with_std.png')
plt.savefig(mean_std_plot_path, dpi=300)
print(f"Saved mean with std plot to: {mean_std_plot_path}")

# Plot 3: Violin plot showing distribution and statistics
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='species', y='mean_wingtip_intensity',
              palette=['#1f77b4', '#ff7f0e'], inner='quartile')
plt.title('Detailed Distribution of Wingtip Intensity by Species', fontsize=14)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Mean Intensity (0-255 scale)', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
violin_plot_path = os.path.join(output_dir, 'wingtip_intensity_violin_plot.png')
plt.savefig(violin_plot_path, dpi=300)
print(f"Saved violin plot to: {violin_plot_path}")

print("\nStatistical Test Results:")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Interpretation: {results['interpretation']}")

print("\nAnalysis complete. All outputs saved to:", output_dir)