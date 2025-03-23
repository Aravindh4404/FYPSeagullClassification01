import os

import pandas as pd
from scipy.stats import norm
import numpy as np


def debug_wingtip_calculation(image_name):
    # Load test data for one image
    wing_data = pd.read_csv('Wing_Greyscale_Intensity_Results/wing_intensity_analysis.csv')
    wingtip_data = pd.read_csv('Wingtip_Greyscale_Intensity_Results/wingtip_intensity_analysis.csv')

    # Get data for specific image
    wing_row = wing_data[wing_data['image_name'] == image_name].iloc[0]
    wingtip_row = wingtip_data[wingtip_data['image_name'] == image_name].iloc[0]

    print("=" * 50)
    print(f"Debugging analysis for: {image_name}")
    print("=" * 50 + "\n")

    # Show raw data
    print("Wing Data:")
    print(f"  Mean intensity: {wing_row['mean_intensity']:.2f}")
    print(f"  Pixel count: {wing_row['pixel_count']}\n")

    print("Wingtip Data:")
    print(f"  Mean intensity: {wingtip_row['mean_intensity']:.2f}")
    print(f"  Std intensity: {wingtip_row['std_intensity']:.2f}")
    print(f"  Pixel count: {wingtip_row['pixel_count']}\n")

    # Perform calculations
    wing_mean = wing_row['mean_intensity']
    wingtip_mean = wingtip_row['mean_intensity']
    wingtip_std = wingtip_row['std_intensity']

    # Calculate Z-score
    if wingtip_std > 0:
        z_score = (wing_mean - wingtip_mean) / wingtip_std
    else:
        z_score = 0

    print("Calculation Steps:")
    print(
        f"1. Intensity difference: {wing_mean:.2f} (wing) - {wingtip_mean:.2f} (wingtip) = {wing_mean - wingtip_mean:.2f}")
    print(f"2. Z-score: ({wing_mean - wingtip_mean:.2f}) / {wingtip_std:.2f} = {z_score:.2f}")

    # Calculate percentage
    percentage = norm.cdf(z_score) * 100
    print(f"3. Percentage calculation: norm.cdf({z_score:.2f}) * 100 = {percentage:.2f}%\n")

    # Validation check
    print("Validation:")
    print(
        f"- If wingtip is darker than wing, wing_mean - wingtip_mean should be positive: {wing_mean - wingtip_mean:.2f}")
    print(f"- Z-score interpretation: Every 1 unit Z-score = 1 standard deviation difference")
    print(f"- CDF value represents probability of values below wing_mean in wingtip distribution")

    # Create synthetic test case
    print("\nSynthetic Test Case:")
    test_wing_mean = 120
    test_wingtip_mean = 100
    test_wingtip_std = 20
    test_z = (test_wing_mean - test_wingtip_mean) / test_wingtip_std
    test_perc = norm.cdf(test_z) * 100
    print(f"Test values: wing_mean=120, wingtip_mean=100, std=20")
    print(f"Expected percentage: ~84.13% | Calculated: {test_perc:.2f}%")

    return percentage
import pandas as pd

def analyze_wingtip_darkness():
    # Load datasets
    wing_df = pd.read_csv('Wing_Greyscale_Intensity_Results/wing_intensity_analysis.csv')
    wingtip_df = pd.read_csv('Wingtip_Greyscale_Intensity_Results/wingtip_intensity_analysis.csv')

    # Merge data
    merged_df = pd.merge(wing_df, wingtip_df, on=['image_name', 'species'], suffixes=('_wing', '_wingtip'))

    # Calculate percentage
    merged_df['percentage_darker'] = (merged_df['pixels_below_wing_mean'] / merged_df['pixel_count_wingtip']) * 100

    # Save results
    output_dir = 'Darkness_Analysis_Results'
    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(f'{output_dir}/per_image_darkness.csv', index=False)

    # Species averages
    species_avg = merged_df.groupby('species')['percentage_darker'].mean().reset_index()
    species_avg.to_csv(f'{output_dir}/species_averages.csv', index=False)
    return merged_df, species_avg


# Example usage
if __name__ == "__main__":
    # Replace with actual image name from your dataset
    TEST_IMAGE = "0H5A6657.png"

    result = debug_wingtip_calculation(TEST_IMAGE)
    print(f"\nFinal result for {TEST_IMAGE}: {result:.2f}%")
