import os

import numpy as np
import cv2
from scipy.stats import skew, kurtosis, ttest_ind
from scipy import ndimage
import matplotlib.pyplot as plt

from Features_Analysis.config import *




def create_gabor_filters():
    """Create a bank of Gabor filters with different orientations and frequencies."""
    filters = []
    num_theta = 4  # Number of orientations
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
            kernel /= kernel.sum()
            filters.append(('gabor_{}_{}'.format(theta, freq), kernel))

    return filters


def extract_texture_features(image, mask, filters):
    """Extract texture features for a masked region."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    features = {}
    masked_pixels = gray[mask > 0]

    if len(masked_pixels) == 0:
        return None

    # Basic statistical features
    features['intensity_mean'] = np.mean(masked_pixels)
    features['intensity_std'] = np.std(masked_pixels)
    features['intensity_skew'] = skew(masked_pixels)
    features['intensity_kurt'] = kurtosis(masked_pixels)

    # Gabor filter responses
    for filter_name, kernel in filters:
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        masked_response = filtered[mask > 0]

        features[f'{filter_name}_mean'] = np.mean(masked_response)
        features[f'{filter_name}_std'] = np.std(masked_response)
        features[f'{filter_name}_energy'] = np.sum(masked_response ** 2)

    return features


def plot_feature_comparison(sb_features, gw_features, feature_name, region_name):
    """Create comparative plot for a specific feature."""
    plt.figure(figsize=(10, 6))

    # Prepare data
    sb_values = [features[feature_name] for features in sb_features if features is not None]
    gw_values = [features[feature_name] for features in gw_features if features is not None]

    # Create box plots
    plt.boxplot([sb_values, gw_values], tick_labels=['Slaty-backed', 'Glaucous-winged'])

    # Add individual points
    x_sb = np.random.normal(1, 0.04, size=len(sb_values))
    x_gw = np.random.normal(2, 0.04, size=len(gw_values))
    plt.plot(x_sb, sb_values, 'r.', alpha=0.3)
    plt.plot(x_gw, gw_values, 'b.', alpha=0.3)

    # Add statistics
    t_stat, p_value = ttest_ind(sb_values, gw_values)
    plt.title(f'{region_name} - {feature_name}\np-value: {p_value:.3f}')
    plt.grid(True, alpha=0.3)

    plt.show()

def plot_gabor_responses(image, mask, filters, title):
    """Visualize Gabor filter responses for a region."""
    if np.sum(mask) == 0:
        return

    n_filters = len(filters)
    n_cols = 4
    n_rows = (n_filters + 2 + n_cols - 1) // n_cols  # Add 2 for original and masked images

    plt.figure(figsize=(15, 3 * n_rows))
    plt.suptitle(title, fontsize=16)

    # Original image
    plt.subplot(n_rows, n_cols, 1)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    plt.imshow(gray, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Masked image
    plt.subplot(n_rows, n_cols, 2)
    masked = gray.copy()
    masked[mask == 0] = 0
    plt.imshow(masked, cmap='gray')
    plt.title('Masked Region')
    plt.axis('off')

    # Filter responses
    for idx, (name, kernel) in enumerate(filters):
        plt.subplot(n_rows, n_cols, idx + 3)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        filtered[mask == 0] = 0
        plt.imshow(filtered, cmap='gray')
        plt.title(f'Filter {idx + 1}')  # Simplified title
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    print("Starting texture analysis...")

    # Load images (adjust S according to your needs)
    S = 5  # Number of images to process

    # Create Gabor filters with fewer combinations
    filters = []
    num_theta = 2  # Reduced from 4
    num_freq = 2  # Reduced from 3
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
            filters.append((f'gabor_{theta}_{freq}', kernel))

    # Load SB images
    sb_images = []
    sb_segs = []
    for img_name in sorted(os.listdir(SLATY_BACKED_IMG_DIR))[:S]:
        img_path = os.path.join(SLATY_BACKED_IMG_DIR, img_name)
        seg_path = os.path.join(SLATY_BACKED_SEG_DIR, img_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is not None and seg is not None:
            sb_images.append(img)
            sb_segs.append(seg)

    # Load GW images
    gw_images = []
    gw_segs = []
    for img_name in sorted(os.listdir(GLAUCOUS_WINGED_IMG_DIR))[:S]:
        img_path = os.path.join(GLAUCOUS_WINGED_IMG_DIR, img_name)
        seg_path = os.path.join(GLAUCOUS_WINGED_SEG_DIR, img_name)
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        if img is not None and seg is not None:
            gw_images.append(img)
            gw_segs.append(seg)

    # Analyze each region
    for region_name, color in REGION_COLORS.items():
        print(f"\nAnalyzing {region_name} region...")

        # Extract features for each species
        sb_region_features = []
        gw_region_features = []

        # Process Slaty-backed images
        for idx, (img, seg) in enumerate(zip(sb_images, sb_segs)):
            tolerance = 10
            lower = np.array([max(c - tolerance, 0) for c in color])
            upper = np.array([min(c + tolerance, 255) for c in color])
            mask = cv2.inRange(seg, lower, upper)

            features = extract_texture_features(img, mask, filters)
            if features:
                sb_region_features.append(features)

                # Visualize Gabor responses for first image
                if idx == 0:
                    plot_gabor_responses(img, mask, filters,
                                         f'Slaty-backed Gull - {region_name} Gabor Responses')

        # Process Glaucous-winged images
        for idx, (img, seg) in enumerate(zip(gw_images, gw_segs)):
            tolerance = 10
            lower = np.array([max(c - tolerance, 0) for c in color])
            upper = np.array([min(c + tolerance, 255) for c in color])
            mask = cv2.inRange(seg, lower, upper)

            features = extract_texture_features(img, mask, filters)
            if features:
                gw_region_features.append(features)

                # Visualize Gabor responses for first image
                if idx == 0:
                    plot_gabor_responses(img, mask, filters,
                                         f'Glaucous-winged Gull - {region_name} Gabor Responses')

        # Compare and visualize features
        if sb_region_features and gw_region_features:
            print(f"\nStatistical comparison for {region_name}:")

            # Basic statistics
            basic_features = ['intensity_mean', 'intensity_std', 'intensity_skew', 'intensity_kurt']
            for feature in basic_features:
                plot_feature_comparison(sb_region_features, gw_region_features,
                                        feature, region_name)

            # Selected Gabor features
            gabor_features = [f'gabor_0_0_mean', f'gabor_0_0_energy']
            for feature in gabor_features:
                plot_feature_comparison(sb_region_features, gw_region_features,
                                        feature, region_name)

            # Print summary statistics
            print("\nSummary Statistics:")
            for feature in basic_features + gabor_features:
                sb_values = [f[feature] for f in sb_region_features]
                gw_values = [f[feature] for f in gw_region_features]

                print(f"\n{feature}:")
                print(f"Slaty-backed: mean={np.mean(sb_values):.2f}, std={np.std(sb_values):.2f}")
                print(f"Glaucous-winged: mean={np.mean(gw_values):.2f}, std={np.std(gw_values):.2f}")

                t_stat, p_value = ttest_ind(sb_values, gw_values)
                print(f"T-statistic={t_stat:.3f}, p-value={p_value:.3f}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()