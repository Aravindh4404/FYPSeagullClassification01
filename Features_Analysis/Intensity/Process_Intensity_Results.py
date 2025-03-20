import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Load the CSV file
file_path = 'D:\FYPSeagullClassification01\Features_Analysis\Intensity\Intensity_Results\Intensity_Results\wing_intensity_analysis.csv'
data = pd.read_csv(file_path)

# Calculate summary statistics grouped by species
summary = data.groupby('species').agg({
    'mean_intensity': ['mean', 'std', 'min', 'max', 'median'],
    'std_intensity': 'mean',
    'skewness': 'mean',
    'kurtosis': 'mean',
    'pixel_count': 'sum'
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

# Create visualizations
plt.figure(figsize=(12, 10))

# Histogram of mean intensity
plt.subplot(2, 2, 1)
sns.histplot(data=data, x='mean_intensity', hue='species', bins=20, kde=True)
plt.title('Distribution of Wing Mean Intensity by Species')

# Box plot of mean intensity
plt.subplot(2, 2, 2)
sns.boxplot(data=data, x='species', y='mean_intensity')
plt.title('Box Plot of Wing Mean Intensity by Species')

# Bar plot of mean intensities
plt.subplot(2, 2, 3)
species_means = data.groupby('species')['mean_intensity'].mean().reset_index()
sns.barplot(data=species_means, x='species', y='mean_intensity')
plt.title('Mean Wing Intensity by Species')
plt.ylabel('Mean Intensity (0-255 scale)')

# Standard deviation comparison
plt.subplot(2, 2, 4)
species_std = data.groupby('species')['std_intensity'].mean().reset_index()
sns.barplot(data=species_std, x='species', y='std_intensity')
plt.title('Mean Standard Deviation of Wing Intensity')
plt.ylabel('Standard Deviation')

# Save plot
plt.tight_layout()
plt.savefig('wing_intensity_comparison.png')

# Create markdown report
md_content = f"""# Wing Intensity Comparison: Slaty-backed Gull vs Glaucous-winged Gull

## Summary of Results

### Mean Wing Intensity
- **Slaty-backed Gull**: {summary[('mean_intensity', 'mean')]['Slaty_Backed_Gull']:.2f}
- **Glaucous-winged Gull**: {summary[('mean_intensity', 'mean')]['Glaucous_Winged_Gull']:.2f}

**Result**: The {darker_species.replace('_', '-')} has darker wings (lower intensity values).

The {brighter_species.replace('_', '-')} wings are approximately {percentage_difference:.1f}% brighter than the {darker_species.replace('_', '-')}.

### Statistical Significance
- **t-statistic**: {t_stat:.4f}
- **p-value**: {p_value:.8f}
- **Significant difference**: {"Yes" if p_value < 0.05 else "No"}

## Detailed Statistics

| Statistic | Slaty-backed Gull | Glaucous-winged Gull |
|-----------|-------------------|----------------------|
| Mean intensity | {summary[('mean_intensity', 'mean')]['Slaty_Backed_Gull']:.2f} | {summary[('mean_intensity', 'mean')]['Glaucous_Winged_Gull']:.2f} |
| Standard deviation | {summary[('mean_intensity', 'std')]['Slaty_Backed_Gull']:.2f} | {summary[('mean_intensity', 'std')]['Glaucous_Winged_Gull']:.2f} |
| Minimum | {summary[('mean_intensity', 'min')]['Slaty_Backed_Gull']:.2f} | {summary[('mean_intensity', 'min')]['Glaucous_Winged_Gull']:.2f} |
| Maximum | {summary[('mean_intensity', 'max')]['Slaty_Backed_Gull']:.2f} | {summary[('mean_intensity', 'max')]['Glaucous_Winged_Gull']:.2f} |
| Median | {summary[('mean_intensity', 'median')]['Slaty_Backed_Gull']:.2f} | {summary[('mean_intensity', 'median')]['Glaucous_Winged_Gull']:.2f} |
| Mean pixel variation (std) | {summary[('std_intensity', 'mean')]['Slaty_Backed_Gull']:.2f} | {summary[('std_intensity', 'mean')]['Glaucous_Winged_Gull']:.2f} |
| Mean skewness | {summary[('skewness', 'mean')]['Slaty_Backed_Gull']:.2f} | {summary[('skewness', 'mean')]['Glaucous_Winged_Gull']:.2f} |
| Mean kurtosis | {summary[('kurtosis', 'mean')]['Slaty_Backed_Gull']:.2f} | {summary[('kurtosis', 'mean')]['Glaucous_Winged_Gull']:.2f} |
| Total pixels analyzed | {int(summary[('pixel_count', 'sum')]['Slaty_Backed_Gull']):,} | {int(summary[('pixel_count', 'sum')]['Glaucous_Winged_Gull']):,} |

## Visualization

![Wing Intensity Comparison](wing_intensity_comparison.png)

## Conclusion

The analysis shows that the {darker_species.replace('_', '-')} has significantly darker wings compared to the {brighter_species.replace('_', '-')}, with a mean intensity value of {summary[('mean_intensity', 'mean')][darker_species]:.2f} compared to {summary[('mean_intensity', 'mean')][brighter_species]:.2f}.

The statistical test confirms that this difference is {"statistically significant" if p_value < 0.05 else "not statistically significant"} (p = {p_value:.8f}).

## Methods

The analysis was performed by measuring the grayscale intensity of segmented wing regions from images of both species. Intensity values range from 0 (black) to 255 (white), with lower values indicating darker coloration.
"""

# Save markdown file
with open('wing_intensity_comparison.md', 'w') as f:
    f.write(md_content)

# Try to convert to HTML if markdown package is available
try:
    import markdown
    html_content = markdown.markdown(md_content)
    with open('wing_intensity_comparison.html', 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Wing Intensity Comparison</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """)
    print("HTML file created successfully!")
except ImportError:
    print("Markdown package not found. Only MD file created.")

print(f"Comparison complete. Results saved to wing_intensity_comparison.md and wing_intensity_comparison.png")
