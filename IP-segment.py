import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk


def compute_curvature(contour):
    """
    Compute a simple discrete curvature along the contour.
    For each point, the curvature is approximated as the angle between the vectors
    from the previous and next points divided by the distance between those points.
    """
    # Reshape contour to (N,2)
    pts = contour[:, 0, :]  # contour shape: (N,1,2)
    N = pts.shape[0]
    curvatures = []
    for i in range(N):
        p_prev = pts[(i - 1) % N]
        p = pts[i]
        p_next = pts[(i + 1) % N]
        v1 = p - p_prev
        v2 = p_next - p
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            continue
        # Compute the angle between v1 and v2
        dot = np.dot(v1, v2)
        cos_angle = np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        # Distance between p_prev and p_next
        d = np.linalg.norm(p_next - p_prev)
        if d == 0:
            continue
        curvature = angle / d
        curvatures.append(curvature)
    return np.array(curvatures)


def compute_contour_features(contour):
    """
    Compute a set of shape features from the given contour.
    """
    features = {}

    # Basic geometric features
    area = cv2.contourArea(contour)
    features["area"] = area
    perimeter = cv2.arcLength(contour, True)
    features["perimeter"] = perimeter

    x, y, w, h = cv2.boundingRect(contour)
    features["bounding_width"] = w
    features["bounding_height"] = h
    features["aspect_ratio"] = float(w) / h if h != 0 else 0
    features["extent"] = float(area) / (w * h) if (w * h) != 0 else 0

    # Convex hull and solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    features["convex_hull_area"] = hull_area
    features["solidity"] = float(area) / hull_area if hull_area != 0 else 0

    # Equivalent diameter
    features["equivalent_diameter"] = np.sqrt(4 * area / np.pi) if area > 0 else 0

    # Fit ellipse to obtain orientation and axis lengths (if possible)
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        features["orientation"] = angle
        features["major_axis_length"] = major_axis
        features["minor_axis_length"] = minor_axis
        features["ellipse_aspect_ratio"] = major_axis / minor_axis if minor_axis != 0 else 0
    else:
        features["orientation"] = np.nan
        features["major_axis_length"] = np.nan
        features["minor_axis_length"] = np.nan
        features["ellipse_aspect_ratio"] = np.nan

    # Curvature features along the contour
    curvatures = compute_curvature(contour)
    if curvatures.size > 0:
        features["mean_curvature"] = float(np.mean(curvatures))
        features["std_curvature"] = float(np.std(curvatures))
    else:
        features["mean_curvature"] = np.nan
        features["std_curvature"] = np.nan

    # Measure of contour irregularity:
    hull_perimeter = cv2.arcLength(hull, True)
    features["hull_perimeter"] = hull_perimeter
    features["perimeter_diff_ratio"] = (hull_perimeter - perimeter) / perimeter if perimeter != 0 else 0

    # Hu Moments (log-transformed)
    moments = cv2.moments(contour)
    huMoments = cv2.HuMoments(moments).flatten()
    for i, hu in enumerate(huMoments):
        # Use log scale for better comparability
        features[f"hu_moment_{i + 1}"] = -1 * np.sign(hu) * np.log10(abs(hu)) if hu != 0 else 0

    return features


class StructureAnalyzer:
    def __init__(self, white_thresh=250, max_images=None):
        """
        Args:
            white_thresh: Threshold for determining white background (0-255).
                          Pixels with intensity above this (in grayscale) are considered background.
            max_images: Maximum number of images to process per folder (if None, process all).
        """
        self.white_thresh = white_thresh
        self.max_images = max_images

    def process_image(self, image_path):
        """
        Process one image to extract the segmented bird's contour and compute shape features.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Since the background is white, threshold to get the bird (non-white regions)
        ret, thresh = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY_INV)

        # Find contours from the binary mask
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Warning: No contours found in {image_path}")
            return None

        # Assume the largest contour is the bird.
        contour = max(contours, key=cv2.contourArea)
        features = compute_contour_features(contour)
        return features

    def process_folder(self, folder_path):
        """
        Process all valid image files in the folder (up to self.max_images) and return results.
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(valid_extensions)]
        image_files = sorted(image_files)
        if self.max_images is not None:
            image_files = image_files[:self.max_images]

        results = []
        for img_file in image_files:
            print(f"Processing {os.path.basename(img_file)}...")
            feat = self.process_image(img_file)
            if feat is not None:
                feat["image"] = os.path.basename(img_file)
                results.append(feat)
        return results

    def save_results_to_csv(self, species1, species2, results1, results2, output_file='structure_analysis_results.csv'):
        """
        Combine results from both species and save to a CSV file.
        """
        df1 = pd.DataFrame(results1)
        df1["Species"] = species1
        df2 = pd.DataFrame(results2)
        df2["Species"] = species2
        df = pd.concat([df1, df2], ignore_index=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        return df

    def visualize_feature(self, df, feature_name):
        """
        Create a boxplot comparing the given feature between species.
        """
        plt.figure(figsize=(8, 6))
        species = df["Species"].unique()
        data = [df[df["Species"] == sp][feature_name].dropna() for sp in species]
        plt.boxplot(data, labels=species, patch_artist=True,
                    boxprops=dict(facecolor="lightgreen", color="green"),
                    medianprops=dict(color="red"))
        plt.ylabel(feature_name)
        plt.title(f"Comparison of {feature_name} between Species")
        plt.show()


def main():
    # Hide the Tkinter root window
    root = Tk()
    root.withdraw()

    # Ask the user to select two folders containing segmented bird images (one per species)
    print("Select folder for the first species...")
    folder1 = filedialog.askdirectory(title="Select folder for first species")
    print("Select folder for the second species...")
    folder2 = filedialog.askdirectory(title="Select folder for second species")

    species1 = os.path.basename(folder1.rstrip(os.sep))
    species2 = os.path.basename(folder2.rstrip(os.sep))

    # Set maximum number of images to process per folder
    max_images = 10  # Change this value as needed

    analyzer = StructureAnalyzer(white_thresh=250, max_images=max_images)

    print(f"\nProcessing images from species: {species1}")
    results1 = analyzer.process_folder(folder1)
    print(f"Processed {len(results1)} images from {species1}.")

    print(f"\nProcessing images from species: {species2}")
    results2 = analyzer.process_folder(folder2)
    print(f"Processed {len(results2)} images from {species2}.")

    # Save combined results to CSV
    df = analyzer.save_results_to_csv(species1, species2, results1, results2)

    # Example visualization: compare area and mean curvature between species
    analyzer.visualize_feature(df, "area")
    analyzer.visualize_feature(df, "mean_curvature")


if __name__ == "__main__":
    main()
