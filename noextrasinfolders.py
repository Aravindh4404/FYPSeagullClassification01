import os
from pathlib import Path

# =============================================================================
# CONFIGURATION - Fill in your paths here
# =============================================================================
FOLDER1_PATH = "D:/FYPSeagullClassification01/Features_Analysis/Dataset/Colored_Images"  # Fill in your first dataset folder path (e.g., "D:/Dataset/Colored_Images")
FOLDER2_PATH = "D:/FYPSeagullClassification01/Features_Analysis/Dataset/Original_Images"  # Fill in your second dataset folder path (e.g., "D:/Dataset/Original_Images")


# =============================================================================
# SCRIPT
# =============================================================================

def compare_dataset_folders(folder1_path, folder2_path):
    """
    Compare two dataset folders to check if subfolder names and contents match exactly.

    Args:
        folder1_path (str): Path to the first dataset folder
        folder2_path (str): Path to the second dataset folder

    Returns:
        dict: Dictionary containing all comparison results
    """

    results = {
        'valid': True,
        'errors': [],
        'folder1_name': '',
        'folder2_name': '',
        'subfolders_match': True,
        'folder1_subfolders': [],
        'folder2_subfolders': [],
        'missing_in_folder1': [],
        'missing_in_folder2': [],
        'subfolder_comparisons': {}
    }

    # Validate paths
    if not folder1_path or not folder2_path:
        results['errors'].append("ERROR: Please fill in both FOLDER1_PATH and FOLDER2_PATH variables!")
        results['valid'] = False
        return results

    folder1 = Path(folder1_path)
    folder2 = Path(folder2_path)

    results['folder1_name'] = folder1.name
    results['folder2_name'] = folder2.name

    # Check if paths exist
    if not folder1.exists():
        results['errors'].append(f"ERROR: First folder does not exist: {folder1}")
        results['valid'] = False

    if not folder2.exists():
        results['errors'].append(f"ERROR: Second folder does not exist: {folder2}")
        results['valid'] = False

    if not results['valid']:
        return results

    try:
        # Get image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'}

        # Get subfolders from both folders
        folder1_subfolders = set()
        folder2_subfolders = set()

        for item in folder1.iterdir():
            if item.is_dir():
                folder1_subfolders.add(item.name)

        for item in folder2.iterdir():
            if item.is_dir():
                folder2_subfolders.add(item.name)

        results['folder1_subfolders'] = sorted(list(folder1_subfolders))
        results['folder2_subfolders'] = sorted(list(folder2_subfolders))

        # Check if subfolders match
        results['missing_in_folder1'] = sorted(list(folder2_subfolders - folder1_subfolders))
        results['missing_in_folder2'] = sorted(list(folder1_subfolders - folder2_subfolders))

        if results['missing_in_folder1'] or results['missing_in_folder2']:
            results['subfolders_match'] = False
            results['valid'] = False

        # Compare contents of matching subfolders
        common_subfolders = folder1_subfolders.intersection(folder2_subfolders)

        for subfolder in common_subfolders:
            subfolder1_path = folder1 / subfolder
            subfolder2_path = folder2 / subfolder

            # Get images from both subfolders
            images1 = set()
            images2 = set()

            for file_path in subfolder1_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    images1.add(file_path.name)

            for file_path in subfolder2_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    images2.add(file_path.name)

            # Compare contents
            extra_in_folder1 = sorted(list(images1 - images2))
            extra_in_folder2 = sorted(list(images2 - images1))

            results['subfolder_comparisons'][subfolder] = {
                'folder1_count': len(images1),
                'folder2_count': len(images2),
                'extra_in_folder1': extra_in_folder1,
                'extra_in_folder2': extra_in_folder2,
                'perfect_match': len(extra_in_folder1) == 0 and len(extra_in_folder2) == 0
            }

            # If any subfolder has mismatches, overall validation fails
            if extra_in_folder1 or extra_in_folder2:
                results['valid'] = False

        return results

    except Exception as e:
        results['errors'].append(f"ERROR: An error occurred while processing: {str(e)}")
        results['valid'] = False
        return results


def print_comparison_results(results):
    """Print detailed comparison results."""

    print("=" * 80)
    print("DATASET FOLDERS COMPARISON RESULTS")
    print("=" * 80)

    # Print errors first
    if results['errors']:
        print("\n❌ ERRORS FOUND:")
        print("-" * 30)
        for error in results['errors']:
            print(f"   {error}")
        return

    # Print folder names being compared
    print(f"\n📁 COMPARING:")
    print(f"   Folder 1: {results['folder1_name']}")
    print(f"   Folder 2: {results['folder2_name']}")

    # Check subfolder matching
    print(f"\n📋 SUBFOLDER COMPARISON:")
    print("-" * 40)

    if results['subfolders_match']:
        print("✅ Subfolder names match perfectly!")
        print(f"   Common subfolders: {', '.join(results['folder1_subfolders'])}")
    else:
        print("❌ Subfolder names do not match!")
        if results['missing_in_folder1']:
            print(f"   Missing in {results['folder1_name']}: {', '.join(results['missing_in_folder1'])}")
        if results['missing_in_folder2']:
            print(f"   Missing in {results['folder2_name']}: {', '.join(results['missing_in_folder2'])}")

    # Print content comparison for each subfolder
    if results['subfolder_comparisons']:
        print(f"\n📊 CONTENT COMPARISON:")
        print("-" * 40)

        all_perfect = True

        for subfolder, comparison in results['subfolder_comparisons'].items():
            print(f"\n🔍 {subfolder}:")
            print(f"   {results['folder1_name']}: {comparison['folder1_count']} images")
            print(f"   {results['folder2_name']}: {comparison['folder2_count']} images")

            if comparison['perfect_match']:
                print("   ✅ Perfect match!")
            else:
                all_perfect = False
                print("   ❌ Mismatch found!")

                if comparison['extra_in_folder1']:
                    print(f"   Extra in {results['folder1_name']} ({len(comparison['extra_in_folder1'])} files):")
                    for i, img in enumerate(comparison['extra_in_folder1'][:5], 1):  # Show first 5
                        print(f"      {i}. {img}")
                    if len(comparison['extra_in_folder1']) > 5:
                        print(f"      ... and {len(comparison['extra_in_folder1']) - 5} more")

                if comparison['extra_in_folder2']:
                    print(f"   Extra in {results['folder2_name']} ({len(comparison['extra_in_folder2'])} files):")
                    for i, img in enumerate(comparison['extra_in_folder2'][:5], 1):  # Show first 5
                        print(f"      {i}. {img}")
                    if len(comparison['extra_in_folder2']) > 5:
                        print(f"      ... and {len(comparison['extra_in_folder2']) - 5} more")

    # Overall result
    print(f"\n🎯 OVERALL RESULT:")
    print("-" * 30)
    if results['valid']:
        print("✅ PERFECT MATCH!")
        print("Both folders have identical subfolder names and contents.")
    else:
        print("❌ MISMATCHES FOUND!")
        print("The folders do not match perfectly. See details above.")


def save_comparison_report(results):
    """Save detailed comparison report to file."""

    output_file = "folders_comparison_report.txt"

    try:
        with open(output_file, 'w') as f:
            f.write("DATASET FOLDERS COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Write errors
            if results['errors']:
                f.write("ERRORS FOUND:\n")
                f.write("-" * 30 + "\n")
                for error in results['errors']:
                    f.write(f"   {error}\n")
                f.write("\n")
                return output_file

            # Write folder names
            f.write("COMPARING:\n")
            f.write(f"   Folder 1: {results['folder1_name']}\n")
            f.write(f"   Folder 2: {results['folder2_name']}\n\n")

            # Write subfolder comparison
            f.write("SUBFOLDER COMPARISON:\n")
            f.write("-" * 40 + "\n")

            if results['subfolders_match']:
                f.write("✅ Subfolder names match perfectly!\n")
                f.write(f"   Common subfolders: {', '.join(results['folder1_subfolders'])}\n\n")
            else:
                f.write("❌ Subfolder names do not match!\n")
                if results['missing_in_folder1']:
                    f.write(f"   Missing in {results['folder1_name']}: {', '.join(results['missing_in_folder1'])}\n")
                if results['missing_in_folder2']:
                    f.write(f"   Missing in {results['folder2_name']}: {', '.join(results['missing_in_folder2'])}\n")
                f.write("\n")

            # Write content comparison
            if results['subfolder_comparisons']:
                f.write("CONTENT COMPARISON:\n")
                f.write("-" * 40 + "\n")

                for subfolder, comparison in results['subfolder_comparisons'].items():
                    f.write(f"\n{subfolder}:\n")
                    f.write(f"   {results['folder1_name']}: {comparison['folder1_count']} images\n")
                    f.write(f"   {results['folder2_name']}: {comparison['folder2_count']} images\n")

                    if comparison['perfect_match']:
                        f.write("   ✅ Perfect match!\n")
                    else:
                        f.write("   ❌ Mismatch found!\n")

                        if comparison['extra_in_folder1']:
                            f.write(
                                f"   Extra in {results['folder1_name']} ({len(comparison['extra_in_folder1'])} files):\n")
                            for i, img in enumerate(comparison['extra_in_folder1'], 1):
                                f.write(f"      {i:3d}. {img}\n")

                        if comparison['extra_in_folder2']:
                            f.write(
                                f"   Extra in {results['folder2_name']} ({len(comparison['extra_in_folder2'])} files):\n")
                            for i, img in enumerate(comparison['extra_in_folder2'], 1):
                                f.write(f"      {i:3d}. {img}\n")

            # Write overall result
            f.write(f"\nOVERALL RESULT:\n")
            f.write("-" * 30 + "\n")
            if results['valid']:
                f.write("✅ PERFECT MATCH!\n")
                f.write("Both folders have identical subfolder names and contents.\n")
            else:
                f.write("❌ MISMATCHES FOUND!\n")
                f.write("The folders do not match perfectly. See details above.\n")

        return output_file

    except Exception as e:
        print(f"❌ Error saving report: {str(e)}")
        return None


def main():
    """Main function to run the folder comparison."""

    print("Starting dataset folders comparison...")

    # Run comparison
    results = compare_dataset_folders(FOLDER1_PATH, FOLDER2_PATH)

    # Print results
    print_comparison_results(results)

    # Ask to save report
    if not results['errors']:
        save_report = input(f"\nDo you want to save the comparison report to a file? (y/n): ").strip().lower()
        if save_report in ['y', 'yes']:
            report_file = save_comparison_report(results)
            if report_file:
                print(f"✅ Comparison report saved to: {report_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()