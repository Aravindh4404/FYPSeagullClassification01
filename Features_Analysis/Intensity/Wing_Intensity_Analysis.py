import pandas as pd
import numpy as np
from scipy import stats
import os

# Create a directory for saving results
output_dir = "Wing_Analysis_Results"
os.makedirs(output_dir, exist_ok=True)

# Load the data
try:
    file_path = 'wing_intensity_analysis.csv'
    data = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")
except Exception as e:
    print(f"Error loading data: {e}")
    # Exit if file cannot be loaded
    exit(1)

# Calculate summary statistics grouped by species
summary = data.groupby('species').agg({
    'mean_intensity': ['mean', 'std', 'min', 'max', 'median', 'count'],
    'std_intensity': ['mean', 'std', 'min', 'max'],
    'skewness': ['mean', 'std'],
    'kurtosis': ['mean', 'std'],
    'pixel_count': ['sum', 'mean', 'count']
})

# Determine which species is darker (lower mean intensity)
darker_species = summary[('mean_intensity', 'mean')].idxmin()
brighter_species = summary[('mean_intensity', 'mean')].idxmax()

# Calculate percentage difference
darker_value = summary[('mean_intensity', 'mean')][darker_species]
brighter_value = summary[('mean_intensity', 'mean')][brighter_species]
percentage_difference = ((brighter_value - darker_value) / darker_value) * 100

# Statistical test (t-test)
slaty = data[data['species'] == 'Slaty_Backed_Gull']['mean_intensity']
glaucous = data[data['species'] == 'Glaucous_Winged_Gull']['mean_intensity']
t_stat, p_value = stats.ttest_ind(slaty, glaucous, equal_var=False)

# Create a more readable summary DataFrame
summary_data = {
    'Species': ['Slaty_Backed_Gull', 'Glaucous_Winged_Gull'],
    'Mean_Intensity': [summary[('mean_intensity', 'mean')]['Slaty_Backed_Gull'],
                      summary[('mean_intensity', 'mean')]['Glaucous_Winged_Gull']],
    'Std_Deviation': [summary[('mean_intensity', 'std')]['Slaty_Backed_Gull'],
                     summary[('mean_intensity', 'std')]['Glaucous_Winged_Gull']],
    'Min_Intensity': [summary[('mean_intensity', 'min')]['Slaty_Backed_Gull'],
                     summary[('mean_intensity', 'min')]['Glaucous_Winged_Gull']],
    'Max_Intensity': [summary[('mean_intensity', 'max')]['Slaty_Backed_Gull'],
                     summary[('mean_intensity', 'max')]['Glaucous_Winged_Gull']],
    'Median_Intensity': [summary[('mean_intensity', 'median')]['Slaty_Backed_Gull'],
                        summary[('mean_intensity', 'median')]['Glaucous_Winged_Gull']],
    'Sample_Count': [summary[('mean_intensity', 'count')]['Slaty_Backed_Gull'],
                    summary[('mean_intensity', 'count')]['Glaucous_Winged_Gull']],
    'Mean_Std_Intensity': [summary[('std_intensity', 'mean')]['Slaty_Backed_Gull'],
                          summary[('std_intensity', 'mean')]['Glaucous_Winged_Gull']],
    'Mean_Skewness': [summary[('skewness', 'mean')]['Slaty_Backed_Gull'],
                     summary[('skewness', 'mean')]['Glaucous_Winged_Gull']],
    'Mean_Kurtosis': [summary[('kurtosis', 'mean')]['Slaty_Backed_Gull'],
                     summary[('kurtosis', 'mean')]['Glaucous_Winged_Gull']],
    'Total_Pixels': [summary[('pixel_count', 'sum')]['Slaty_Backed_Gull'],
                    summary[('pixel_count', 'sum')]['Glaucous_Winged_Gull']]
}

# Create statistics DataFrame
results_df = pd.DataFrame(summary_data)

# Add t-test results
ttest_data = {
    'Test': ['T-Test: Slaty vs Glaucous'],
    'T_Statistic': [t_stat],
    'P_Value': [p_value],
    'Significant': [p_value < 0.05],
    'Percentage_Difference': [percentage_difference],
    'Darker_Species': [darker_species],
    'Brighter_Species': [brighter_species]
}

ttest_df = pd.DataFrame(ttest_data)

# Save results to CSV files
results_df.to_csv(os.path.join(output_dir, 'wing_intensity_summary.csv'), index=False)
ttest_df.to_csv(os.path.join(output_dir, 'wing_intensity_ttest.csv'), index=False)

print(f"Analysis complete! Results saved to {output_dir}")
print(f"- Summary statistics: wing_intensity_summary.csv")
print(f"- T-test results: wing_intensity_ttest.csv")

# Print key findings
print("\nKey findings:")
print(f"- Slaty-backed Gull wing intensity: {summary[('mean_intensity', 'mean')]['Slaty_Backed_Gull']:.2f} ± {summary[('mean_intensity', 'std')]['Slaty_Backed_Gull']:.2f}")
print(f"- Glaucous-winged Gull wing intensity: {summary[('mean_intensity', 'mean')]['Glaucous_Winged_Gull']:.2f} ± {summary[('mean_intensity', 'std')]['Glaucous_Winged_Gull']:.2f}")
print(f"- Percentage difference: {percentage_difference:.1f}%")
print(f"- Statistical significance: p = {p_value:.8f} ({'Significant' if p_value < 0.05 else 'Not Significant'})")
