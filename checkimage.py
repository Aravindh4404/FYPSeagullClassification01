import os
from PIL import Image
import imagehash


def get_image_hashes(folder_path):
    image_hashes = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                img_hash = imagehash.average_hash(img)
                image_hashes[filename] = (img_hash, file_path)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return image_hashes


def find_absent_images(folder_A, folder_B, similarity_threshold=0):
    hashes_A = get_image_hashes(folder_A)
    hashes_B = get_image_hashes(folder_B)

    # Extract just the hash values from B
    hash_values_B = [hb[0] for hb in hashes_B.values()]

    absent_images = []

    for name_A, (hash_A, path_A) in hashes_A.items():
        found_match = False
        for hash_B in hash_values_B:
            if hash_A - hash_B <= similarity_threshold:
                found_match = True
                break
        if not found_match:
            absent_images.append((name_A, path_A))

    print("\nImages in Folder A that are NOT in Folder B:")
    for img_name, img_path in absent_images:
        print(f"❌ {img_name}")
        try:
            Image.open(img_path).show()
        except Exception as e:
            print(f"Could not open {img_name}: {e}")

    return [name for name, _ in absent_images]


# Example usage
folder_A = "D:/Dataset_FYP/Interpretable_Seagull_Classification/Model_Training_Dataset/train/Glaucous_Winged_Gull"
folder_B = "D:/FYPSeagullClassification01/Features_Analysis/Dataset/Original_Images/Glaucous_Winged_Gull"

find_absent_images(folder_A, folder_B, similarity_threshold=5)
