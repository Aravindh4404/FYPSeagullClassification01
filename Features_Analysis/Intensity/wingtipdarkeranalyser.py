import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import sys

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))
from Features_Analysis.config import *


def analyze_wingtip_darkness_ratio(image_path, seg_path, species, file_name, mean_wing_intensity):
    """
    Analyzes the portion of wingtip that is darker than the wing for a single image.

    Args:
        image_path: Path to the original image
        seg_path: Path to the segmentation image
        species: Species name
        file_name: Image file name
        mean_wing_intensity: Pre-calculated mean wing intensity for this image

    Returns:
        Dictionary with analysis results
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Convert to grayscale and normalize (same as in your original code)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)

    # Extract wingtip region from the normalized grayscale image
    wingtip_region, wingtip_mask = extract_region(gray_img, segmentation_img, "wingtip")

    # Get wingtip pixels
    wingtip_pixels = wingtip_region[wingtip_mask > 0]

    if len(wingtip_pixels) == 0:
        print(f"No wingtip region found in {file_name}")
        return None

    # Count pixels darker than mean wing intensity
    darker_pixels = wingtip_pixels < mean_wing_intensity
    darker_pixel_count = np.sum(darker_pixels)
    total_wingtip_pixels = len(wingtip_pixels)

    # Calculate the ratio/percentage
    darkness_ratio = darker_pixel_count / total_wingtip_pixels
    darkness_percentage = darkness_ratio * 100

    # Additional statistics for analysis
    mean_wingtip_intensity = np.mean(wingtip_pixels)
    intensity_difference = mean_wing_intensity - mean_wingtip_intensity

    # Calculate how much darker the dark pixels are on average
    if darker_pixel_count > 0:
        dark_wingtip_pixels = wingtip_pixels[darker_pixels]
        mean_dark_intensity = np.mean(dark_wingtip_pixels)
        avg_darkness_diff = mean_wing_intensity - mean_dark_intensity
    else:
        mean_dark_intensity = None
        avg_darkness_diff = 0

    return {
        'image_name': file_name,
        'species': species,
        'mean_wing_intensity': mean_wing_intensity,
        'mean_wingtip_intensity': mean_wingtip_intensity,
        'total_wingtip_pixels': total_wingtip_pixels,
        'darker_pixel_count': darker_pixel_count,
        'darkness_ratio': darkness_ratio,
        'darkness_percentage': darkness_percentage,
        'wing_wingtip_diff': intensity_difference,
        'mean_dark_pixel_intensity': mean_dark_intensity,
        'avg_darkness_difference': avg_darkness_diff
    }


def load_wing_intensity_data():
    """
    Load the previously saved wing intensity data.
    Modify this path to match where your wing intensity CSV is saved.
    """
    wing_data_path = "Intensity_Results/wing_intensity_analysis.csv"

    if not os.path.exists(wing_data_path):
        print(f"Wing intensity data not found at {wing_data_path}")
        print("Please make sure you've run the wing intensity analysis first.")
        return None

    wing_df = pd.read_csv(wing_data_path)

    # Create a lookup dictionary for quick access
    wing_intensity_lookup = {}
    for _, row in wing_df.iterrows():
        wing_intensity_lookup[row['image_name']] = row['mean_intensity']

    return wing_intensity_lookup


def main():
    """
    Main function to process all images and analyze wingtip darkness ratios.
    """
    # Load wing intensity data
    print("Loading wing intensity data...")
    wing_intensity_lookup = load_wing_intensity_data()

    if wing_intensity_lookup is None:
        return

    print(f"Loaded wing intensity data for {len(wing_intensity_lookup)} images")

    # Process all images
    results = []

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} wingtip darkness...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f"  Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")

            # Get the corresponding wing intensity
            if file_name not in wing_intensity_lookup:
                print(f"  Warning: No wing intensity data found for {file_name}")
                continue

            mean_wing_intensity = wing_intensity_lookup[file_name]

            # Analyze darkness ratio
            darkness_stats = analyze_wingtip_darkness_ratio(
                img_path, seg_path, species_name, file_name, mean_wing_intensity
            )

            if darkness_stats:
                results.append(darkness_stats)

    # Save results
    if results:
        df = pd.DataFrame(results)

        # Create results directory
        results_dir = "Wingtip_Darkness_Analysis"
        os.makedirs(results_dir, exist_ok=True)

        # Save detailed results
        csv_path = os.path.join(results_dir, "wingtip_darkness_analysis.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")

        # Calculate and save species averages
        species_avg = df.groupby('species').agg({
            'darkness_percentage': ['mean', 'std', 'min', 'max'],
            'wing_wingtip_diff': ['mean', 'std'],
            'avg_darkness_difference': ['mean', 'std'],
            'total_wingtip_pixels': 'mean'
        }).round(2)

        # Flatten column names
        species_avg.columns = ['_'.join(col).strip() for col in species_avg.columns.values]
        species_avg = species_avg.reset_index()

        avg_csv_path = os.path.join(results_dir, "species_darkness_averages.csv")
        species_avg.to_csv(avg_csv_path, index=False)
        print(f"Species averages saved to: {avg_csv_path}")

        # Print summary statistics
        print("\n" + "=" * 50)
        print("WINGTIP DARKNESS ANALYSIS SUMMARY")
        print("=" * 50)

        for species in df['species'].unique():
            species_data = df[df['species'] == species]
            print(f"\n{species.upper()}:")
            print(
                f"  Average darkness percentage: {species_data['darkness_percentage'].mean():.1f}% (±{species_data['darkness_percentage'].std():.1f}%)")
            print(
                f"  Range: {species_data['darkness_percentage'].min():.1f}% - {species_data['darkness_percentage'].max():.1f}%")
            print(f"  Images analyzed: {len(species_data)}")
            print(f"  Average wing-wingtip difference: {species_data['wing_wingtip_diff'].mean():.1f} intensity units")

        print(f"\nAnalysis complete! Results saved in {results_dir} directory.")
        print("Use 'wingtip_darkness_plotter.py' to create visualizations from this data.")

    else:
        print("No results generated. Check if images and wing intensity data are available.")


if __name__ == "__main__":
    main()