import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Features_Analysis.config import GLAUCOUS_WINGED_IMG_DIR, SLATY_BACKED_IMG_DIR


##############################################################################
# BASIC FUNCTIONS
##############################################################################
def to_grayscale(image):
    """
    Ensures the image is in grayscale before processing.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


##############################################################################
# NORMALIZATION FUNCTIONS (Per-Image Methods)
##############################################################################
def minmax_normalize(image):
    """
    Scales grayscale pixel intensities to [0, 255] using per-image Min-Max normalization.
    """
    gray = to_grayscale(image)
    min_val = np.min(gray)
    max_val = np.max(gray)
    norm = (gray - min_val) / (max_val - min_val + 1e-8)  # scale to [0,1]
    return (norm * 255).astype(np.uint8)


def zscore_normalize(image):
    """
    Standardizes grayscale pixel intensities (mean=0, std=1) then rescales them to [0,255].
    """
    gray = to_grayscale(image)
    mean, std = np.mean(gray), np.std(gray)
    zscored = (gray - mean) / (std + 1e-8)
    z_min, z_max = np.min(zscored), np.max(zscored)
    z_norm = (zscored - z_min) / (z_max - z_min + 1e-8)
    return (z_norm * 255).astype(np.uint8)


def hist_equalize(image):
    """
    Applies standard histogram equalization to a grayscale image.
    """
    gray = to_grayscale(image)
    return cv2.equalizeHist(gray)


def clahe_normalize(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


##############################################################################
# GLOBAL MIN-MAX NORMALIZATION FUNCTIONS
##############################################################################
def compute_global_minmax(image_list):
    """
    Computes the global minimum and maximum pixel values from a list of images.

    Parameters:
      image_list (list): A list of tuples (filename, image)

    Returns:
      tuple: (global_min, global_max)
    """
    global_min = float('inf')
    global_max = float('-inf')
    for fname, img in image_list:
        gray = to_grayscale(img)
        local_min = np.min(gray)
        local_max = np.max(gray)
        if local_min < global_min:
            global_min = local_min
        if local_max > global_max:
            global_max = local_max
    return global_min, global_max


def global_minmax_normalize(image, global_min, global_max):
    """
    Scales grayscale pixel intensities of an image to [0, 255] using global min and max values.

    Parameters:
      image (numpy.ndarray): Input image.
      global_min (float): Global minimum pixel value.
      global_max (float): Global maximum pixel value.

    Returns:
      numpy.ndarray: Normalized image.
    """
    gray = to_grayscale(image)
    norm = (gray - global_min) / (global_max - global_min + 1e-8)
    return (norm * 255).astype(np.uint8)


##############################################################################
# IMAGE LOADING FUNCTION
##############################################################################
def load_images(folder_path):
    """
    Loads all images from a given folder into a list of (filename, image).

    Returns:
      list: List of tuples (filename, image)
    """
    images = []
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            img = cv2.imread(fpath)
            if img is not None:
                images.append((fname, img))
    return images


##############################################################################
# MAIN SCRIPT
##############################################################################
def main():
    """
    1) Load images from the Slaty-Backed and Glaucous-Winged directories.
    2) Compute global min and max across all images.
    3) Compare different normalization methods (including global min-max) by
       displaying and saving a plot for each method.
    """

    # Load images from both folders
    slaty_images = load_images(SLATY_BACKED_IMG_DIR)  # list of (filename, image)
    glaucous_images = load_images(GLAUCOUS_WINGED_IMG_DIR)  # list of (filename, image)

    # Combine images from both folders
    all_images = slaty_images + glaucous_images
    if not all_images:
        print("No images found. Please check your folder paths.")
        return

    # Compute global min and max across all images
    global_min, global_max = compute_global_minmax(all_images)
    print(f"Global minimum: {global_min}, Global maximum: {global_max}")

    # Define normalization methods including our new Global Min-Max
    normalization_methods = {
        "Greyscale": to_grayscale,
        "MinMax": minmax_normalize,
        "ZScore": zscore_normalize,
        "HistogramEqualization": hist_equalize,
        "CLAHE": clahe_normalize,
        "GlobalMinMax": lambda img: global_minmax_normalize(img, global_min, global_max)
    }

    # Create an output folder to store the comparison plots
    output_folder = "Outputs/Normalization_Comparison_All"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # For each normalization method, create a single figure containing all images
    num_images = len(all_images)
    cols = 5  # Adjust number of columns as needed
    rows = int(np.ceil(num_images / cols))

    for method_name, method_func in normalization_methods.items():
        fig = plt.figure(figsize=(4 * cols, 3 * rows))
        fig.suptitle(f"Normalization Method: {method_name}", fontsize=16)

        for i, (fname, img) in enumerate(all_images, start=1):
            ax = fig.add_subplot(rows, cols, i)
            norm_img = method_func(img)
            ax.imshow(norm_img, cmap='gray')
            ax.set_title(fname, fontsize=9)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_plot = os.path.join(output_folder, f"{method_name}_comparison.png")
        plt.savefig(out_plot, dpi=150)
        plt.close()
        print(f"Saved comparison plot for {method_name}: {out_plot}")

    print("\nAll normalization comparison plots saved in:", output_folder)


if __name__ == "__main__":
    main()
