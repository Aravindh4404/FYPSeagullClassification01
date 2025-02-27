import os
import numpy as np
import cv2
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

from Features_Analysis.config import *


def create_gabor_filters():
    """
    Create a bank of Gabor filters with different orientations and frequencies.
    """
    filters = []
    num_theta = 8  # Number of orientations
    num_freq = 3  # Number of frequencies
    ksize = 31  # Kernel size
    sigma = 4.0  # Gaussian standard deviation

    for theta in range(num_theta):
        theta_val = theta * np.pi / num_theta
        for freq in range(num_freq):
            frequency = 0.1 + freq * 0.1
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta_val,
                frequency,
                0.5,
                0,
                ktype=cv2.CV_32F
            )
            # Normalize kernel so that the sum of values = 1
            kernel /= kernel.sum()
            filter_name = f"gabor_{theta}_{freq}"
            filters.append((filter_name, kernel))

    return filters


def extract_gabor_features(image, mask, filters):
    """
    Extract Gabor filter–based features for a masked region in the image.

    Returns a dictionary of:
      - gabor_{theta}_{freq}_mean   -> average filter response
      - gabor_{theta}_{freq}_std    -> standard deviation of filter response
      - gabor_{theta}_{freq}_energy -> sum of squared filter responses
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # If mask is empty, return None
    if np.sum(mask) == 0:
        return None

    features = {}

    # Apply each Gabor filter and collect stats
    for filter_name, kernel in filters:
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        # Only consider pixels within the mask
        masked_response = filtered[mask > 0]

        features[f"{filter_name}_mean"] = np.mean(masked_response)
        features[f"{filter_name}_std"] = np.std(masked_response)
        features[f"{filter_name}_energy"] = np.sum(masked_response ** 2)

    return features


def plot_gabor_responses(image, mask, filters, title):
    """
    Visualize how each Gabor filter responds to the masked region in the image.
    """
    if np.sum(mask) == 0:
        return

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    n_filters = len(filters)
    n_cols = 4
    # +2 for Original and Masked images
    n_rows = (n_filters + 2 + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))
    plt.suptitle(title, fontsize=16)

    # 1. Original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # 2. Masked image
    plt.subplot(n_rows, n_cols, 2)
    masked_img = gray.copy()
    masked_img[mask == 0] = 0
    plt.imshow(masked_img, cmap='gray')
    plt.title('Masked Region')
    plt.axis('off')

    # 3. Filter responses
    for idx, (filter_name, kernel) in enumerate(filters):
        plt.subplot(n_rows, n_cols, idx + 3)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        filtered[mask == 0] = 0  # Zero out areas outside the mask
        plt.imshow(filtered, cmap='gray')
        plt.title(filter_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_feature_comparison(sb_features, gw_features, feature_name, region_name):
    """
    Create a comparative box plot for a specific Gabor feature across two species.
    """
    sb_values = [f[feature_name] for f in sb_features if f is not None]
    gw_values = [f[feature_name] for f in gw_features if f is not None]

    if not sb_values or not gw_values:
        return  # No data to plot

    plt.figure(figsize=(8, 5))
    plt.boxplot([sb_values, gw_values], labels=['Slaty-backed', 'Glaucous-winged'])
    plt.title(f"{region_name} - {feature_name}")
    plt.grid(True, alpha=0.3)

    # Perform T-test and display p-value in title
    t_stat, p_value = ttest_ind(sb_values, gw_values)
    plt.title(f"{region_name} - {feature_name}\nT-stat={t_stat:.3f}, p={p_value:.3f}")
    plt.show()


def main():
    print("Starting Gabor-based texture analysis...")

    # Number of images to process (change to None or remove [ :S ] to load all)
    S = 5

    # Create a reduced set of Gabor filters
    # (If you want the full set, use create_gabor_filters() instead)
    filters = []
    num_theta = 2  # orientations
    num_freq = 2  # frequencies
    ksize = 31
    sigma = 4.0

    for theta in range(num_theta):
        theta_val = theta * np.pi / num_theta
        for freq in range(num_freq):
            frequency = 0.1 + freq * 0.1
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta_val,
                frequency,
                0.5,
                0,
                ktype=cv2.CV_32F
            )
            kernel /= kernel.sum()
            filter_name = f"gabor_{theta}_{freq}"
            filters.append((filter_name, kernel))

    # --- Load Slaty-backed gull images ---
    # --- Load Slaty-backed gull images ---
    sb_images = []
    sb_segs = []
    sb_filenames = sorted(os.listdir(SLATY_BACKED_IMG_DIR))[:S]
    print(f"Found {len(sb_filenames)} Slaty-backed images")

    for img_name in sb_filenames:
        img_path = os.path.join(SLATY_BACKED_IMG_DIR, img_name)
        seg_path = os.path.join(SLATY_BACKED_SEG_DIR, img_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is not None and seg is not None:
            sb_images.append(img)
            sb_segs.append(seg)
        else:
            print(f"Could not load {img_name} or its segmentation.")

    # --- Load Glaucous-winged gull images ---
    gw_images = []
    gw_segs = []
    gw_filenames = sorted(os.listdir(GLAUCOUS_WINGED_IMG_DIR))[:S]
    print(f"Found {len(gw_filenames)} Glaucous-winged images")

    for img_name in gw_filenames:
        img_path = os.path.join(GLAUCOUS_WINGED_IMG_DIR, img_name)
        seg_path = os.path.join(GLAUCOUS_WINGED_SEG_DIR, img_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is not None and seg is not None:
            gw_images.append(img)
            gw_segs.append(seg)
        else:
            print(f"Could not load {img_name} or its segmentation.")

    # --- Analyze each defined region ---
    for region_name, color in REGION_COLORS.items():
        print(f"\nAnalyzing '{region_name}' region...")

        sb_region_features = []
        gw_region_features = []

        # --- Process Slaty-backed images ---
        for idx, (img, seg) in enumerate(zip(sb_images, sb_segs)):
            tolerance = 10
            lower = np.array([max(c - tolerance, 0) for c in color])
            upper = np.array([min(c + tolerance, 255) for c in color])
            mask = cv2.inRange(seg, lower, upper)

            features = extract_gabor_features(img, mask, filters)
            if features:
                sb_region_features.append(features)
                # Show Gabor response for all images
                plot_gabor_responses(img, mask, filters,
                                     f"Slaty-backed Gull - {region_name} Gabor Responses - Image {idx + 1}")

        # --- Process Glaucous-winged images ---
        for idx, (img, seg) in enumerate(zip(gw_images, gw_segs)):
            tolerance = 10
            lower = np.array([max(c - tolerance, 0) for c in color])
            upper = np.array([min(c + tolerance, 255) for c in color])
            mask = cv2.inRange(seg, lower, upper)

            features = extract_gabor_features(img, mask, filters)
            if features:
                gw_region_features.append(features)
                # Show Gabor response for the first Glaucous-winged image
                plot_gabor_responses(img, mask, filters,
                                         f"Glaucous-winged Gull - {region_name} Gabor Responses - Image {idx + 1}")

        # --- Compare features if both species have data ---
        if sb_region_features and gw_region_features:
            print(f"\nStatistical comparison for {region_name} region:")
            # List out the Gabor feature keys (e.g., 'gabor_0_0_mean', 'gabor_0_0_energy', etc.)
            # We'll just pick a few to visualize or you can loop through them all.
            gabor_keys = [k for k in sb_region_features[0].keys()]

            for gk in gabor_keys:
                plot_feature_comparison(sb_region_features, gw_region_features, gk, region_name)

            print("Comparison done.")


if __name__ == "__main__":
    main()
