import pandas as pd
from Features_Analysis.config import *  # Import configuration file


def analyze_wingtip_intensity_distribution(image_path, seg_path, species, file_name):
    """
    Analyzes the intensity distribution of wingtip pixels and their difference
    from the wing intensity for a single image.
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Extract wing and wingtip regions
    wing_region, wing_mask = extract_region(original_img, segmentation_img, "wing")
    wingtip_region, wingtip_mask = extract_region(original_img, segmentation_img, "wingtip")

    # Convert to grayscale
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_wing = cv2.cvtColor(wing_region, cv2.COLOR_BGR2GRAY)

    # Get wing pixels
    wing_pixels = gray_wing[wing_mask > 0]

    if len(wing_pixels) == 0:
        print(f"No wing region found in {file_name}")
        return None

    # Calculate mean wing intensity
    mean_wing_intensity = np.mean(wing_pixels)

    # Get wingtip pixels from the original grayscale image
    wingtip_pixels = gray_img[wingtip_mask > 0]

    if len(wingtip_pixels) == 0:
        print(f"No wingtip region found in {file_name}")
        return None

    # Define intensity ranges (bins)
    intensity_ranges = [
        (0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
        (50, 60), (60, 70), (70, 80), (80, 90), (90, 100),
        (100, 110), (110, 120), (120, 130), (130, 140), (140, 150),
        (150, 160), (160, 170), (170, 180), (180, 190), (190, 200),
        (200, 210), (210, 220), (220, 230), (230, 240), (240, 255)
    ]

    # Count pixels in each intensity range
    range_counts = {}
    for start, end in intensity_ranges:
        # Count raw number of pixels in this range
        pixel_count = np.sum((wingtip_pixels >= start) & (wingtip_pixels < end))
        range_counts[f"intensity_{start}_{end}"] = pixel_count
        # Calculate percentage of wingtip pixels in this range
        range_counts[f"pct_{start}_{end}"] = (pixel_count / len(wingtip_pixels)) * 100

    # Calculate wing-wingtip differences
    # For each wingtip pixel, calculate how much darker it is than the mean wing
    intensity_diffs = mean_wing_intensity - wingtip_pixels

    # Only keep positive differences (darker pixels)
    positive_diffs = intensity_diffs[intensity_diffs > 0]

    # Define difference thresholds
    diff_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Count pixels with differences above thresholds
    diff_counts = {}
    for threshold in diff_thresholds:
        pixel_count = np.sum(intensity_diffs > threshold)
        diff_counts[f"diff_gt_{threshold}"] = pixel_count
        diff_counts[f"pct_diff_gt_{threshold}"] = (pixel_count / len(wingtip_pixels)) * 100

    # Calculate statistics about very dark pixels
    very_dark_counts = {}
    dark_thresholds = [30, 40, 50, 60]
    for threshold in dark_thresholds:
        pixel_count = np.sum(wingtip_pixels < threshold)
        very_dark_counts[f"dark_lt_{threshold}"] = pixel_count
        very_dark_counts[f"pct_dark_lt_{threshold}"] = (pixel_count / len(wingtip_pixels)) * 100

    # Prepare results
    results = {
        "image_name": file_name,
        "species": species,
        "mean_wing_intensity": mean_wing_intensity,
        "mean_wingtip_intensity": np.mean(wingtip_pixels),
        "wing_pixel_count": len(wing_pixels),
        "wingtip_pixel_count": len(wingtip_pixels),
        "darker_pixel_count": len(positive_diffs),
        "pct_darker_pixels": (len(positive_diffs) / len(wingtip_pixels)) * 100,
        **range_counts,
        **diff_counts,
        **very_dark_counts
    }

    return results


def main():
    """
    Process all images for both species and save intensity distribution results.
    """
    results = []

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} images...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f" Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")

            stats = analyze_wingtip_intensity_distribution(img_path, seg_path, species_name, file_name)

            if stats:
                results.append(stats)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)

        results_dir = "Wingtip_Intensity_Distribution"
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "wingtip_intensity_distribution.csv")
        df.to_csv(csv_path, index=False)

        print(f"\nDetailed results saved to: {csv_path}")

        # Calculate averages by species
        # Select columns that are percentages for easier comparison
        pct_columns = [col for col in df.columns if col.startswith("pct_")]
        dark_columns = [col for col in df.columns if col.startswith("dark_")]
        diff_columns = [col for col in df.columns if col.startswith("diff_")]

        # Calculate species averages
        species_avg = df.groupby('species')[['mean_wing_intensity', 'mean_wingtip_intensity'] +
                                            pct_columns + dark_columns + diff_columns].mean().reset_index()

        avg_csv_path = os.path.join(results_dir, "wingtip_intensity_averages.csv")
        species_avg.to_csv(avg_csv_path, index=False)

        print(f"\nSpecies averages saved to: {avg_csv_path}")


if __name__ == "__main__":
    main()
