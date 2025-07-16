import pandas as pd
import os
import shutil
from pathlib import Path


def copy_misclassified_images():
    """
    Copy misclassified seagull images from source to destination folders
    based on species information in the CSV file.
    """

    # File paths
    csv_path = r"D:\FYPSeagullClassification01\Clustering\clustering_results\gmm_misclassified_points.csv"
    source_base = r"D:\FYPSeagullClassification01\Features_Analysis\Dataset\Original_Images"
    destination_base = r"C:\Users\aravi\Desktop\Clustering_Misclassified"

    try:
        # Load the CSV file
        print("Loading CSV file...")
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} records from CSV file")

        # Extract image names and species
        image_names = df['image_name'].tolist()
        species_list = df['species'].tolist()

        # Get unique species for folder creation
        unique_species = df['species'].unique()
        print(f"Found species: {list(unique_species)}")

        # Create destination folders if they don't exist
        print("Creating destination folders...")
        for species in unique_species:
            dest_folder = os.path.join(destination_base, species)
            Path(dest_folder).mkdir(parents=True, exist_ok=True)
            print(f"Created/verified folder: {dest_folder}")

        # Copy images
        print("Starting image copying process...")
        copied_count = 0
        not_found_count = 0

        for image_name, species in zip(image_names, species_list):
            # Construct source path
            source_path = os.path.join(source_base, species, image_name)

            # Construct destination path
            dest_path = os.path.join(destination_base, species, image_name)

            # Check if source file exists and copy
            if os.path.exists(source_path):
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                    print(f"Copied: {image_name} -> {species}")
                except Exception as e:
                    print(f"Error copying {image_name}: {str(e)}")
            else:
                not_found_count += 1
                print(f"Image not found: {source_path}")

        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Total images processed: {len(image_names)}")
        print(f"Successfully copied: {copied_count}")
        print(f"Not found: {not_found_count}")
        print(f"Destination folder: {destination_base}")

    except FileNotFoundError as e:
        print(f"Error: CSV file not found at {csv_path}")
        print(f"Please check the file path and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    copy_misclassified_images()
