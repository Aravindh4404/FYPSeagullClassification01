import os
import pandas as pd
import numpy as np
from scipy.stats import norm


def analyze_wingtip_darkness():
    # Load both datasets
    wing_df = pd.read_csv('Wing_Greyscale_Intensity_Results/wing_intensity_analysis.csv')
    wingtip_df = pd.read_csv('Wingtip_Greyscale_Intensity_Results/wingtip_intensity_analysis.csv')

    # Merge data on image_name and species
    merged_df = pd.merge(wing_df, wingtip_df, on=['image_name', 'species'], suffixes=('_wing', '_wingtip'))

    # Calculate darkness percentage for each image
    results = []
    for _, row in merged_df.iterrows():
        wing_mean = row['mean_intensity_wing']
        wingtip_mean = row['mean_intensity_wingtip']
        wingtip_std = row['std_intensity_wingtip']
        wingtip_pixel_count = row['pixel_count_wingtip']

        # Estimate percentage of wingtip pixels darker than wing mean
        # Using normal distribution approximation since we only have summary statistics
        z_score = (wing_mean - wingtip_mean) / wingtip_std if wingtip_std > 0 else 0
        percentage_darker = norm.cdf(z_score) * 100

        results.append({
            'image_name': row['image_name'],
            'species': row['species'],
            'wing_mean_intensity': wing_mean,
            'wingtip_mean_intensity': wingtip_mean,
            'percentage_wingtip_darker': percentage_darker,
            'pixel_count_wingtip': wingtip_pixel_count
        })

    # Create results dataframe
    result_df = pd.DataFrame(results)

    # Calculate species-level averages
    species_avg = result_df.groupby('species')['percentage_wingtip_darker'].mean().reset_index()
    species_avg.columns = ['species', 'mean_percentage_darker']

    # Save individual results
    output_dir = 'Darkness_Analysis_Results'
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'per_image_darkness_comparison.csv')
    result_df.to_csv(csv_path, index=False)

    # Save species averages
    species_csv_path = os.path.join(output_dir, 'species_darkness_averages.csv')
    species_avg.to_csv(species_csv_path, index=False)

    print(f"Individual results saved to {csv_path}")
    print(f"Species averages saved to {species_csv_path}")

    return result_df, species_avg


if __name__ == "__main__":
    analyze_wingtip_darkness()
