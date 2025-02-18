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
    from the previous and next points, divided by the distance between those points.
    """
    pts = contour[:, 0, :]  # shape: (N, 2)
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
        # Angle between v1 and v2
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
    perimeter = cv2.arcLength(contour, True)
    features["area"] = area
    features["perimeter"] = perimeter

    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    features["bounding_width"] = w
    features["bounding_height"] = h
    features["aspect_ratio"] = float(w) / h if h != 0 else 0
    features["extent"] = float(area) / (w * h) if (w * h) != 0 else 0

    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    features["convex_hull_area"] = hull_area
    features["solidity"] = float(area) / hull_area if hull_area != 0 else 0

    # Equivalent diameter
    features["equivalent_diameter"] = np.sqrt(4 * area / np.pi) if area > 0 else 0

    # Fit ellipse if sufficient points
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

    # Curvature features
    curvatures = compute_curvature(contour)
    if curvatures.size > 0:
        features["mean_curvature"] = float(np.mean(curvatures))
        features["std_curvature"] = float(np.std(curvatures))
    else:
        features["mean_curvature"] = np.nan
        features["std_curvature"] = np.nan

    # Compare contour perimeter to hull perimeter (irregularity measure)
    hull_perimeter = cv2.arcLength(hull, True)
    features["hull_perimeter"] = hull_perimeter
    features["perimeter_diff_ratio"] = (
        (hull_perimeter - perimeter) / perimeter if perimeter != 0 else 0
    )

    # Hu Moments (log-transformed)
    moments = cv2.moments(contour)
    huMoments = cv2.HuMoments(moments).flatten()
    for i, hu in enumerate(huMoments):
        # Use log scale for better comparability
        features[f"hu_moment_{i + 1}"] = (
            -1 * np.sign(hu) * np.log10(abs(hu)) if hu != 0 else 0
        )

    return features


def visualize_processing(original_bgr, thresh, contour):
    """
    Display (in a single figure):
      1) The original image,
      2) The thresholded mask,
      3) The original image with the largest contour overlaid
    """

    # Convert BGR -> RGB for matplotlib
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # Draw contour on a copy of the original
    overlay = original_bgr.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Plot with subplots
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(thresh_rgb)
    axs[1].set_title("Thresholded Mask")
    axs[1].axis("off")

    axs[2].imshow(overlay_rgb)
    axs[2].set_title("Contour Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


class StructureAnalyzer:
    def __init__(self,
                 white_thresh=250,
                 max_images=None,
                 show_visuals=True):
        """
        Args:
            white_thresh: Threshold for white background (0-255).
            max_images: Maximum number of images to process per folder (None => all).
            show_visuals: If True, show step-by-step images for each file.
        """
        self.white_thresh = white_thresh
        self.max_images = max_images
        self.show_visuals = show_visuals

    def process_image(self, image_path):
        """
        Process one image to get the largest contour and compute shape features.
        Also optionally visualize the steps.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold to get the bird (non-white region)
        #   - anything below white_thresh is "bird" => we invert
        _, thresh = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY_INV)

        # Find external contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Warning: No contours found in {image_path}")
            return None

        # Largest contour => the bird
        contour = max(contours, key=cv2.contourArea)

        # Optionally visualize the steps
        if self.show_visuals:
            visualize_processing(img, thresh, contour)

        # Compute features
        features = compute_contour_features(contour)
        return features

    def process_folder(self, folder_path):
        """
        Process valid image files in the folder (up to self.max_images).
        Returns a list of feature dictionaries.
        """
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in sorted(os.listdir(folder_path))
                 if f.lower().endswith(valid_exts)]
        if self.max_images is not None:
            files = files[:self.max_images]

        results = []
        for filename in files:
            img_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            feat = self.process_image(img_path)
            if feat is not None:
                feat["image"] = filename
                results.append(feat)
        return results

    def save_results_to_csv(self, species1, species2, results1, results2, output_file="structure_analysis_results.csv"):
        """
        Combine results from both species and save to CSV.
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
        plt.figure(figsize=(8, 5))
        species_list = df["Species"].unique()
        data = [df[df["Species"] == sp][feature_name].dropna() for sp in species_list]
        plt.boxplot(data, labels=species_list, patch_artist=True,
                    boxprops=dict(facecolor="lightgreen", color="green"),
                    medianprops=dict(color="red"))
        plt.ylabel(feature_name)
        plt.title(f"Comparison of {feature_name} between Species")
        plt.show()


def main():
    # Hide the Tk root window
    root = Tk()
    root.withdraw()

    print("Select the MAIN folder containing two subfolders (one per species).")
    main_folder = filedialog.askdirectory(title="Select main folder")

    # List subdirectories
    subfolders = [d for d in sorted(os.listdir(main_folder))
                  if os.path.isdir(os.path.join(main_folder, d))]

    if len(subfolders) < 2:
        print("Error: Main folder must have at least two subfolders for species.")
        print("Found subfolders:", subfolders)
        return

    # For example, pick the first two subfolders as species1 and species2
    species1_folder = os.path.join(main_folder, subfolders[0])
    species2_folder = os.path.join(main_folder, subfolders[1])

    species1 = subfolders[0]
    species2 = subfolders[1]

    # Create analyzer
    # show_visuals=True => it will display the image, threshold, and contour for each image
    analyzer = StructureAnalyzer(white_thresh=250, max_images=5, show_visuals=True)

    print(f"\nAnalyzing species 1: {species1}  in folder: {species1_folder}")
    results1 = analyzer.process_folder(species1_folder)
    print(f"Processed {len(results1)} images from {species1}.")

    print(f"\nAnalyzing species 2: {species2}  in folder: {species2_folder}")
    results2 = analyzer.process_folder(species2_folder)
    print(f"Processed {len(results2)} images from {species2}.")

    # Save combined results to CSV
    df = analyzer.save_results_to_csv(species1, species2, results1, results2,
                                      output_file="structure_analysis_results.csv")

    # Example: visualize area and mean curvature
    analyzer.visualize_feature(df, "area")
    analyzer.visualize_feature(df, "mean_curvature")


if __name__ == "__main__":
    main()
