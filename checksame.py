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

def validate_dataset_csv_match(dataset_path, csv_path):
    """
    Comprehensive validation to check if folder contents and CSV entries match exactly.

    Args:
        dataset_path (str): Path to the main dataset folder containing species subfolders
        csv_path (str): Path to the CSV file

    Returns:
        dict: Dictionary containing all validation results
    """

    results = {
        'valid': True,
        'errors': [],
        'glaucous_folder_count': 0,
        'slaty_folder_count': 0,
        'glaucous_csv_count': 0,
        'slaty_csv_count': 0,
        'glaucous_extra_in_folder': [],
        'glaucous_extra_in_csv': [],
        'slaty_extra_in_folder': [],
        'slaty_extra_in_csv': []
    }

    # Validate paths
    if not dataset_path or not csv_path:
        results['errors'].append("ERROR: Please fill in both DATASET_PATH and CSV_PATH variables!")
        results['valid'] = False
        return results

    dataset_path = Path(dataset_path)
    glaucous_folder = dataset_path / "Glaucous_Winged_Gull"
    slaty_folder = dataset_path / "Slaty_Backed_Gull"

    # Check if paths exist
    if not dataset_path.exists():
        results['errors'].append(f"ERROR: Dataset path does not exist: {dataset_path}")
        results['valid'] = False
        return results

    if not glaucous_folder.exists():
        results['errors'].append(f"ERROR: Glaucous_Winged_Gull folder does not exist: {glaucous_folder}")
        results['valid'] = False

    if not slaty_folder.exists():
        results['errors'].append(f"ERROR: Slaty_Backed_Gull folder does not exist: {slaty_folder}")
        results['valid'] = False

    if not os.path.exists(csv_path):
        results['errors'].append(f"ERROR: CSV file does not exist: {csv_path}")
        results['valid'] = False

    if not results['valid']:
        return results

    try:
        # Read the CSV file
        print(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        # Check if required columns exist
        if 'image_name' not in df.columns or 'species' not in df.columns:
            results['errors'].append("ERROR: CSV file must contain 'image_name' and 'species' columns!")
            results['valid'] = False
            return results

        # Get image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'}

        # Get images from folders
        glaucous_folder_images = set()
        slaty_folder_images = set()

        # Scan Glaucous_Winged_Gull folder
        for file_path in glaucous_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                glaucous_folder_images.add(file_path.name)

        # Scan Slaty_Backed_Gull folder
        for file_path in slaty_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                slaty_folder_images.add(file_path.name)

        # Get images from CSV
        glaucous_csv = df[df['species'] == 'Glaucous_Winged_Gull']
        slaty_csv = df[df['species'] == 'Slaty_Backed_Gull']

        glaucous_csv_images = set(glaucous_csv['image_name'].astype(str))
        slaty_csv_images = set(slaty_csv['image_name'].astype(str))

        # Store counts
        results['glaucous_folder_count'] = len(glaucous_folder_images)
        results['slaty_folder_count'] = len(slaty_folder_images)
        results['glaucous_csv_count'] = len(glaucous_csv_images)
        results['slaty_csv_count'] = len(slaty_csv_images)

        # Find discrepancies for Glaucous_Winged_Gull
        results['glaucous_extra_in_folder'] = sorted(list(glaucous_folder_images - glaucous_csv_images))
        results['glaucous_extra_in_csv'] = sorted(list(glaucous_csv_images - glaucous_folder_images))

        # Find discrepancies for Slaty_Backed_Gull
        results['slaty_extra_in_folder'] = sorted(list(slaty_folder_images - slaty_csv_images))
        results['slaty_extra_in_csv'] = sorted(list(slaty_csv_images - slaty_folder_images))

        # Check if everything matches perfectly
        if (results['glaucous_extra_in_folder'] or results['glaucous_extra_in_csv'] or
                results['slaty_extra_in_folder'] or results['slaty_extra_in_csv']):
            results['valid'] = False

        return results

    except Exception as e:
        results['errors'].append(f"ERROR: An error occurred while processing: {str(e)}")
        results['valid'] = False
        return results


def print_validation_results(results):
    """Print detailed validation results."""

    print("=" * 70)
    print("DATASET AND CSV VALIDATION RESULTS")
    print("=" * 70)

    # Print errors first
    if results['errors']:
        print("\n❌ ERRORS FOUND:")
        print("-" * 30)
        for error in results['errors']:
            print(f"   {error}")
        return

    # Print summary
    print(f"\n📊 SUMMARY:")
    print("-" * 30)
    print(f"Glaucous Winged Gull - Folder: {results['glaucous_folder_count']}, CSV: {results['glaucous_csv_count']}")
    print(f"Slaty Backed Gull    - Folder: {results['slaty_folder_count']}, CSV: {results['slaty_csv_count']}")

    # Check overall status
    if results['valid']:
        print(f"\n✅ PERFECT MATCH!")
        print("All images in folders exactly match the CSV entries.")
        print("No extra or missing images found in either folders or CSV.")
        return

    print(f"\n⚠️  MISMATCHES FOUND:")
    print("-" * 40)

    # Glaucous Winged Gull discrepancies
    if results['glaucous_extra_in_folder']:
        print(f"\n🔍 Glaucous Winged Gull - EXTRA IN FOLDER ({len(results['glaucous_extra_in_folder'])} images):")
        for i, img in enumerate(results['glaucous_extra_in_folder'], 1):
            print(f"   {i:3d}. {img}")

    if results['glaucous_extra_in_csv']:
        print(f"\n📄 Glaucous Winged Gull - EXTRA IN CSV ({len(results['glaucous_extra_in_csv'])} images):")
        for i, img in enumerate(results['glaucous_extra_in_csv'], 1):
            print(f"   {i:3d}. {img}")

    # Slaty Backed Gull discrepancies
    if results['slaty_extra_in_folder']:
        print(f"\n🔍 Slaty Backed Gull - EXTRA IN FOLDER ({len(results['slaty_extra_in_folder'])} images):")
        for i, img in enumerate(results['slaty_extra_in_folder'], 1):
            print(f"   {i:3d}. {img}")

    if results['slaty_extra_in_csv']:
        print(f"\n📄 Slaty Backed Gull - EXTRA IN CSV ({len(results['slaty_extra_in_csv'])} images):")
        for i, img in enumerate(results['slaty_extra_in_csv'], 1):
            print(f"   {i:3d}. {img}")


def save_validation_report(results):
    """Save detailed validation report to file."""

    output_file = "dataset_validation_report.txt"

    try:
        with open(output_file, 'w') as f:
            f.write("DATASET AND CSV VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            # Write errors
            if results['errors']:
                f.write("ERRORS FOUND:\n")
                f.write("-" * 30 + "\n")
                for error in results['errors']:
                    f.write(f"   {error}\n")
                f.write("\n")
                return output_file

            # Write summary
            f.write("SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"Glaucous Winged Gull - Folder: {results['glaucous_folder_count']}, CSV: {results['glaucous_csv_count']}\n")
            f.write(
                f"Slaty Backed Gull    - Folder: {results['slaty_folder_count']}, CSV: {results['slaty_csv_count']}\n\n")

            if results['valid']:
                f.write("✅ PERFECT MATCH!\n")
                f.write("All images in folders exactly match the CSV entries.\n")
                f.write("No extra or missing images found in either folders or CSV.\n")
            else:
                f.write("⚠️  MISMATCHES FOUND:\n")
                f.write("-" * 40 + "\n")

                # Write Glaucous discrepancies
                if results['glaucous_extra_in_folder']:
                    f.write(
                        f"\nGlaucous Winged Gull - EXTRA IN FOLDER ({len(results['glaucous_extra_in_folder'])} images):\n")
                    for i, img in enumerate(results['glaucous_extra_in_folder'], 1):
                        f.write(f"   {i:3d}. {img}\n")

                if results['glaucous_extra_in_csv']:
                    f.write(
                        f"\nGlaucous Winged Gull - EXTRA IN CSV ({len(results['glaucous_extra_in_csv'])} images):\n")
                    for i, img in enumerate(results['glaucous_extra_in_csv'], 1):
                        f.write(f"   {i:3d}. {img}\n")

                # Write Slaty discrepancies
                if results['slaty_extra_in_folder']:
                    f.write(
                        f"\nSlaty Backed Gull - EXTRA IN FOLDER ({len(results['slaty_extra_in_folder'])} images):\n")
                    for i, img in enumerate(results['slaty_extra_in_folder'], 1):
                        f.write(f"   {i:3d}. {img}\n")

                if results['slaty_extra_in_csv']:
                    f.write(f"\nSlaty Backed Gull - EXTRA IN CSV ({len(results['slaty_extra_in_csv'])} images):\n")
                    for i, img in enumerate(results['slaty_extra_in_csv'], 1):
                        f.write(f"   {i:3d}. {img}\n")

        return output_file

    except Exception as e:
        print(f"❌ Error saving report: {str(e)}")
        return None


def main():
    """Main function to run the complete validation."""

    print("Starting dataset validation...")

    # Run validation
    results = validate_dataset_csv_match(DATASET_PATH, CSV_PATH)

    # Print results
    print_validation_results(results)

    # Ask to save report
    if not results['errors']:
        save_report = input(f"\nDo you want to save the validation report to a file? (y/n): ").strip().lower()
        if save_report in ['y', 'yes']:
            report_file = save_validation_report(results)
            if report_file:
                print(f"✅ Validation report saved to: {report_file}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()