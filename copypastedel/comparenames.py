import os
import shutil
from hashlib import sha256

# Function to calculate the hash of an image file
def calculate_image_hash(file_path):
    with open(file_path, "rb") as f:
        return sha256(f.read()).hexdigest()

# Function to compare images and move duplicates
def compare_and_move_images(folder1, folder2, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a dictionary to store hashes of images in folder1
    folder1_hashes = {}
    for img1 in os.listdir(folder1):
        if img1.lower().endswith(('jpg', 'jpeg', 'png')):
            img1_path = os.path.join(folder1, img1)
            folder1_hashes[calculate_image_hash(img1_path)] = img1

    # Compare images in folder2 against those in folder1
    for img2 in os.listdir(folder2):
        if img2.lower().endswith(('jpg', 'jpeg', 'png')):
            img2_path = os.path.join(folder2, img2)
            img2_hash = calculate_image_hash(img2_path)
            if img2_hash in folder1_hashes:  # If a duplicate is found
                shutil.move(img2_path, os.path.join(output_folder, img2))
                print(f"Moved: {img2}")

# Define the folder paths
folder1 = r"C:\Users\Aravindh P\Desktop\sbg refined by me"
folder2 = r"C:\Users\Aravindh P\Desktop\chosen sbg ori"
output_folder = r"C:\Users\Aravindh P\Desktop\sbg ori chosen already done"

# Call the function to compare and move images
compare_and_move_images(folder1, folder2, output_folder)
