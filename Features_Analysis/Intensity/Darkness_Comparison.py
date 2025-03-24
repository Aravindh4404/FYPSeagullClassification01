import pandas as pd
from Features_Analysis.config import *  # Import everything from the config file


def analyze_wingtip_darkness(image_path, seg_path, species, file_name):
    """
    Analyzes what percentage of wingtip pixels are darker than the mean wing intensity
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
    gray_wing = cv2.cvtColor(wing_region, cv2.COLOR_BGR2GRAY)

    # Get wing pixels (non-zero)
    wing_pixels = gray_wing[wing_mask > 0]

    if len(wing_pixels) == 0:
        print(f"No wing region found in {file_name}")
        return None

    # Calculate mean intensity of the wing
    mean_wing_intensity = np.mean(wing_pixels)

    # Get grayscale of original image for wingtip analysis
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Get wingtip pixels
    wingtip_pixels = gray_img[wingtip_mask > 0]

    if len(wingtip_pixels) == 0:
        print(f"No wingtip region found in {file_name}")
        return None

    # Find wingtip pixels darker than the mean wing intensity
    darker_pixels = wingtip_pixels[wingtip_pixels < mean_wing_intensity]

    # Calculate percentage of darker pixels
    percentage_darker = (len(darker_pixels) / len(wingtip_pixels)) * 100

    return {
        "image_name": file_name,
        "species": species,
        "mean_wing_intensity": mean_wing_intensity,
        "wing_pixel_count": len(wing_pixels),
        "wingtip_pixel_count": len(wingtip_pixels),
        "darker_wingtip_pixels": len(darker_pixels),
        "percentage_darker": percentage_darker
    }


def main():
    """
    Process all images and save results
    """
    results = []

    for species_name, paths in SPECIES.items():
        print(f"\nAnalyzing {species_name} images...")

        image_paths = get_image_paths(species_name)

        for i, (img_path, seg_path) in enumerate(image_paths[:S]):
            file_name = os.path.basename(img_path)
            print(f" Processing image {i + 1}/{min(S, len(image_paths))}: {file_name}")

            stats = analyze_wingtip_darkness(img_path, seg_path, species_name, file_name)

            if stats:
                results.append(stats)

    # Save results
    if results:
        df = pd.DataFrame(results)

        # Create results directory
        results_dir = "Darkness_Analysis_Results"
        os.makedirs(results_dir, exist_ok=True)

        # Save detailed results to CSV
        csv_path = os.path.join(results_dir, "wingtip_darkness_analysis.csv")
        df.to_csv(csv_path, index=False)

        # Calculate averages by species
        species_avg = df.groupby('species')['percentage_darker'].mean().reset_index()
        species_avg = species_avg.rename(columns={'percentage_darker': 'average_percentage_darker'})

        # Save species averages to CSV
        avg_csv_path = os.path.join(results_dir, "wingtip_darkness_averages.csv")
        species_avg.to_csv(avg_csv_path, index=False)

        # Save a combined CSV with all results and the averages appended
        combined_csv_path = os.path.join(results_dir, "wingtip_darkness_combined.csv")
        with open(combined_csv_path, 'w') as f:
            df.to_csv(f, index=False)
            f.write("\n\nSpecies Averages:\n")
            species_avg.to_csv(f, index=False)

        print(f"\nResults saved to:\n- {csv_path}\n- {avg_csv_path}\n- {combined_csv_path}")

        # Print summary
        print("\nSpecies averages:")
        for _, row in species_avg.iterrows():
            print(f"  {row['species']}: {row['average_percentage_darker']:.2f}% of wingtip darker than wing")
    else:
        print("No results generated. Check if wing and wingtip regions were detected.")


if __name__ == "__main__":
    main()
