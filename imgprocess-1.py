import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------
# Helper function to compute curvature
# --------------------------
def compute_curvature(contour, window_size=5):
    """
    Compute an approximate curvature for each point along a contour.

    Parameters:
        contour (numpy.ndarray): The contour array (of shape [N,1,2] as returned by cv2.findContours).
        window_size (int): Number of points before and after the current point to use in the calculation.

    Returns:
        curvatures (numpy.ndarray): Array of curvature values computed along the contour.
    """
    contour = contour.reshape(-1, 2)
    curvatures = []
    for i in range(window_size, len(contour) - window_size):
        p_prev = contour[i - window_size]
        p_curr = contour[i]
        p_next = contour[i + window_size]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        chord_length = np.linalg.norm(p_next - p_prev)
        curvature = angle / chord_length if chord_length != 0 else 0
        curvatures.append(curvature)

    return np.array(curvatures)


# --------------------------
# Function to process one image
# --------------------------
def process_image(image_path, output_dir):
    """
    Process one image to compute several descriptors and save intermediate output images.

    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Directory where intermediate images will be saved.

    Returns:
        features (dict): Dictionary of computed descriptors for the image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    intensity_mean = np.mean(gray)
    intensity_std = np.std(gray)
    intensity_min = np.min(gray)
    intensity_max = np.max(gray)

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners_img = image.copy()
    if corners is not None:
        corners = corners.astype(np.int32)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(corners_img, (x, y), 3, (0, 0, 255), -1)
    num_corners = len(corners) if corners is not None else 0

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_features = {}
    curvature_stats = {}
    contour_img = image.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        shape_features = {'area': area, 'perimeter': perimeter, 'aspect_ratio': aspect_ratio}
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 2)

        curvatures = compute_curvature(largest_contour, window_size=5)
        if curvatures.size > 0:
            curvature_mean = np.mean(curvatures)
            curvature_std = np.std(curvatures)
        else:
            curvature_mean = 0
            curvature_std = 0
        curvature_stats = {'curvature_mean': curvature_mean, 'curvature_std': curvature_std}

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, base_name + '_gray.png'), gray)
    cv2.imwrite(os.path.join(output_dir, base_name + '_corners.png'), corners_img)
    cv2.imwrite(os.path.join(output_dir, base_name + '_contour.png'), contour_img)
    cv2.imwrite(os.path.join(output_dir, base_name + '_thresh.png'), thresh)

    features = {
        'image_name': base_name,
        'intensity_mean': intensity_mean,
        'intensity_std': intensity_std,
        'intensity_min': intensity_min,
        'intensity_max': intensity_max,
        'num_corners': num_corners
    }
    features.update(shape_features)
    features.update(curvature_stats)

    return features


# --------------------------
# Function to process all images in a folder
# --------------------------
def process_folder(folder_path, output_folder):
    """
    Process all image files in the given folder.

    Parameters:
        folder_path (str): Path to the folder containing images.
        output_folder (str): Directory to save intermediate output images.

    Returns:
        features_list (list): List of descriptor dictionaries for each processed image.
    """
    os.makedirs(output_folder, exist_ok=True)

    features_list = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(folder_path, file)
            print(f"Processing: {image_path}")
            features = process_image(image_path, output_folder)
            if features is not None:
                features_list.append(features)
    return features_list


# --------------------------
# Main function
# --------------------------
def main():
    # Set up your folder paths correctly
    folder1 = r'C:/Users/Aravindh P/Desktop/HQ3/test/Glaucous_Winged_Gull'
    folder2 = r'C:/Users/Aravindh P/Desktop/HQ3/test/Slaty_Backed_Gull'

    output_folder1 = 'output_species1'
    output_folder2 = 'output_species2'

    print("Processing Species 1 Images...")
    features_species1 = process_folder(folder1, output_folder1)
    print("Processing Species 2 Images...")
    features_species2 = process_folder(folder2, output_folder2)

    df_species1 = pd.DataFrame(features_species1)
    df_species2 = pd.DataFrame(features_species2)
    df_species1.to_csv(os.path.join(output_folder1, 'features_species1.csv'), index=False)
    df_species2.to_csv(os.path.join(output_folder2, 'features_species2.csv'), index=False)
    print("Feature CSV files saved.")

    # Display intermediate results for the first processed image
    if features_species1:
        sample_name = features_species1[0]['image_name']
        sample_image = os.path.join(output_folder1, sample_name + '_contour.png')
        if os.path.exists(sample_image):
            image = cv2.imread(sample_image)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Contour of {sample_name}")
            plt.axis('off')
            plt.show()


if __name__ == "__main__":
    main()
