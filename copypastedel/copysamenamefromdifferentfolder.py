import os
import shutil


def copy_matching_images(reference_dir, source_dir, destination_dir):
    """
    Copies images from source_dir to destination_dir if they have the same names
    as files in reference_dir.
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Get list of reference filenames
    reference_files = set(os.listdir(reference_dir))

    # Get all files in source directory
    for filename in os.listdir(source_dir):
        src_path = os.path.join(source_dir, filename)

        # Check if it's a file and exists in reference
        if os.path.isfile(src_path) and filename in reference_files:
            dest_path = os.path.join(destination_dir, filename)

            # Copy with metadata preservation
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {filename}")


# Configure your paths here
REFERENCE_DIR = "D:\FYP\GradCAM_OutputFull\chosen sbg grad"
SOURCE_DIR = "D:\ALLIMAGESLATEST\HQ3FULL\Slaty_Backed_Gull"
DESTINATION_DIR = "D:\FYP\GradCAM_OutputFull\chosen sbg ori"

copy_matching_images(REFERENCE_DIR, SOURCE_DIR, DESTINATION_DIR)
print("Operation completed!")
