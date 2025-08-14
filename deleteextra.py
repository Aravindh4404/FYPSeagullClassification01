import os
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION - Fill in your paths here
# =============================================================================
DATASET_PATH = "D:/FYPSeagullClassification01/Features_Analysis/Dataset/Original_Images"  # Fill in your image folder path here (e.g., "/path/to/your/images")
CSV_PATH = "D:/FYPSeagullClassification01/Features_Analysis/Intensity/Intensity_Results/wing_intensity_analysis.csv"  # Fill in your CSV file path here (e.g., "/path/to/your/file.csv")


# =============================================================================
# SCRIPT
# =============================================================================

def find_extra_glaucous_winged_images(dataset_path, csv_path):
    """
    Find Glaucous Winged Gull images that are in the Glaucous_Winged_Gull folder
    but not present in the CSV file.

    Args:
        dataset_path (str): Path to the main dataset folder containing species subfolders
        csv_path (str): Path to the CSV file

    Returns:
        list: List of extra Glaucous Winged Gull image filenames
    """

    # Validate paths
    if not dataset_path or not csv_path:
        print("ERROR: Please fill in both DATASET_PATH and CSV_PATH variables!")
        return []

    glaucous_folder = Path(dataset_path) / "Glaucous_Winged_Gull"

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return []

    if not glaucous_folder.exists():
        print(f"ERROR: Glaucous_Winged_Gull folder does not exist: {glaucous_folder}")
        return []

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file does not exist: {csv_path}")
        return []

    try:
        # Read the CSV file
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        # Check if required columns exist
        if 'image_name' not in df.columns or 'species' not in df.columns:
            print("ERROR: CSV file must contain 'image_name' and 'species' columns!")
            return []

        # Filter for Glaucous Winged Gull entries only
        glaucous_df = df[df['species'] == 'Glaucous_Winged_Gull']
        print(f"Found {len(glaucous_df)} Glaucous Winged Gull entries in CSV")

        # Get Glaucous Winged Gull image names from CSV
        csv_glaucous_images = set(glaucous_df['image_name'].astype(str))

        # Get all image files from Glaucous_Winged_Gull folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'}

        folder_glaucous_images = set()
        for file_path in glaucous_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                folder_glaucous_images.add(file_path.name)

        print(f"Found {len(folder_glaucous_images)} Glaucous Winged Gull image files in folder")

        # Find images that are in Glaucous_Winged_Gull folder but NOT in CSV
        extra_images = folder_glaucous_images - csv_glaucous_images

        return sorted(list(extra_images))

    except Exception as e:
        print(f"ERROR: An error occurred while processing: {str(e)}")
        return []


def main():
    """Main function to run the missing images checker."""

    print("=" * 60)
    print("EXTRA GLAUCOUS WINGED GULL IMAGES CHECKER")
    print("=" * 60)

    # Find extra Glaucous Winged Gull images
    extra_images = find_extra_glaucous_winged_images(DATASET_PATH, CSV_PATH)

    if extra_images:
        print(f"\n🔍 FOUND {len(extra_images)} EXTRA GLAUCOUS WINGED GULL IMAGES:")
        print("-" * 50)
        for i, image in enumerate(extra_images, 1):
            print(f"{i:3d}. {image}")

        # Optional: Save to file
        save_to_file = input(f"\nDo you want to save the list to a text file? (y/n): ").strip().lower()
        if save_to_file in ['y', 'yes']:
            output_file = "extra_glaucous_winged_gull_images.txt"
            try:
                with open(output_file, 'w') as f:
                    f.write("Extra Glaucous Winged Gull Images (In folder but not in CSV)\n")
                    f.write("=" * 55 + "\n\n")
                    for i, image in enumerate(extra_images, 1):
                        f.write(f"{i:3d}. {image}\n")
                print(f"✅ List saved to: {output_file}")
            except Exception as e:
                print(f"❌ Error saving file: {str(e)}")

    else:
        print("\n✅ NO EXTRA GLAUCOUS WINGED GULL IMAGES FOUND!")
        print("All Glaucous Winged Gull images in the folder are present in the CSV file.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()