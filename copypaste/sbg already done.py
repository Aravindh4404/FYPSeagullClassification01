import os
import shutil

# Function to cut and paste files based on a list of names
def move_matching_files_by_name(file_names, source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over the given file names
    for name in file_names:
        # Check for both .jpg and .psd extensions
        for extension in ['.jpg', '.jpeg', '.png', '.psd']:
            file_to_find = name + extension
            source_path = os.path.join(source_folder, file_to_find)
            if os.path.exists(source_path):  # If the file exists in the source folder
                shutil.move(source_path, os.path.join(destination_folder, file_to_find))
                print(f"Moved: {file_to_find}")

# List of base file names (without extensions) to search for
file_names = [
    "007", "011sm", "0H5A0441", "0H5A0763", "0H5A0953", "0H5A2340",
    "0H5A2344", "0H5A2454", "0H5A6659", "0H5A7926 - Version 2", "0H5A8133",
    "0H5A9027", "3O4A0520 2", "3O4A1101 2", "3O4A1105 2", "3O4A2534",
    "3O4A2750", "3O4A3199", "3O4A3219", "3O4A4276", "3O4A4347",
    "3O4A4503", "3O4A4681", "3O4A4935", "3O4A8036", "3O4A8472",
    "480 (15)", "480 (28)", "480 (41)", "480 (45)", "4D2A0024",
    "4D2A0897", "4D2A1442", "4D2A2332", "4D2A2395 2", "4D2A6309",
    "4D2A6676", "4D2A6684", "4D2A6883", "4D2A7325", "4D2A7396",
    "4D2A9213", "4D2A9493", "4D2A9502 2", "FS7E7062", "FS7E8437"
]

# Define the source and destination folder paths
source_folder = r"D:\FYP\GradCAM_OutputFull\aftermask - Copy"
destination_folder = r"D:\FYP\GradCAM_OutputFull\aftermask - Copy\dy sbg"

# Call the function to move matching files
move_matching_files_by_name(file_names, source_folder, destination_folder)
