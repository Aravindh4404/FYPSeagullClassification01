import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk


class ToneAnalyzer:
    def __init__(self, white_threshold=230):
        """
        Args:
            white_threshold: Pixel values >= this (in all channels) are considered white.
                             You may tweak this value if needed.
        """
        self.white_threshold = white_threshold

    def white_balance(self, img):
        """
        A simple white-balance adjustment that uses the white background.
        It computes the average background value in each channel (using a threshold)
        and scales the channel so that the background becomes (close to) pure white.
        """
        # Define lower/upper bounds for white (tolerance)
        lower = np.array([self.white_threshold] * 3, dtype=np.uint8)
        upper = np.array([255] * 3, dtype=np.uint8)
        bg_mask = cv2.inRange(img, lower, upper)

        # If no background is found, skip correction.
        if cv2.countNonZero(bg_mask) == 0:
            return img

        # Get indices where the background is detected.
        indices = np.where(bg_mask > 0)
        # Compute mean for each channel over the background pixels.
        b_mean = np.mean(img[:, :, 0][indices])
        g_mean = np.mean(img[:, :, 1][indices])
        r_mean = np.mean(img[:, :, 2][indices])

        # Compute scaling factors so that the mean becomes 255.
        scale_b = 255.0 / b_mean if b_mean > 0 else 1.0
        scale_g = 255.0 / g_mean if g_mean > 0 else 1.0
        scale_r = 255.0 / r_mean if r_mean > 0 else 1.0

        # Apply scaling to each channel.
        img_float = img.astype(np.float32)
        img_float[:, :, 0] *= scale_b
        img_float[:, :, 1] *= scale_g
        img_float[:, :, 2] *= scale_r

        # Clip to valid range and convert back to uint8.
        balanced = np.clip(img_float, 0, 255).astype(np.uint8)
        return balanced

    def extract_roi_mask(self, img):
        """
        Since the images are segmented with a white background, we assume that
        any pixel not almost white belongs to the region of interest.
        """
        lower = np.array([self.white_threshold] * 3, dtype=np.uint8)
        upper = np.array([255] * 3, dtype=np.uint8)
        # Create a mask for the white background.
        white_mask = cv2.inRange(img, lower, upper)
        # Invert to get the segmented (non-white) region.
        roi_mask = cv2.bitwise_not(white_mask)
        return roi_mask

    def analyze_image(self, img_path):
        """
        Loads an image, applies white-balance normalization, extracts the ROI,
        and computes the mean and standard deviation of the grayscale intensity.
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image {img_path}")
            return None

        # Apply white-balance normalization.
        balanced = self.white_balance(img)

        # Extract the segmented (non-white) region.
        roi_mask = self.extract_roi_mask(balanced)
        if cv2.countNonZero(roi_mask) == 0:
            print(f"Warning: No ROI detected in {img_path}")
            return None

        # Convert to grayscale.
        gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
        roi_values = gray[roi_mask > 0]

        # Compute average tone (mean intensity) and standard deviation.
        mean_intensity = np.mean(roi_values)
        std_intensity = np.std(roi_values)
        return mean_intensity, std_intensity

    def process_folder(self, folder_path):
        """
        Iterates over all valid image files in the folder and analyzes them.
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(valid_extensions)]
        results = []
        for img_file in image_files:
            result = self.analyze_image(img_file)
            if result is not None:
                mean_intensity, std_intensity = result
                results.append({
                    "image": os.path.basename(img_file),
                    "mean_intensity": mean_intensity,
                    "std_intensity": std_intensity
                })
        return results

    def save_results_to_csv(self, species1, species2, results1, results2, output_file='tone_analysis_results.csv'):
        """
        Combines the results from both species and saves them as a CSV file.
        """
        df1 = pd.DataFrame(results1)
        df1["Species"] = species1
        df2 = pd.DataFrame(results2)
        df2["Species"] = species2
        df = pd.concat([df1, df2], ignore_index=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        return df

    def visualize_results(self, df):
        """
        Creates a boxplot of the mean intensity values for the two species.
        """
        plt.figure(figsize=(8, 6))
        species = df["Species"].unique()
        data = [df[df["Species"] == sp]["mean_intensity"] for sp in species]
        plt.boxplot(data, labels=species, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="blue"),
                    medianprops=dict(color="red"))
        plt.ylabel("Mean Grayscale Intensity (0-255)")
        plt.title("Tone Analysis Comparison Between Species")
        plt.show()


def main():
    # Hide the Tkinter root window.
    root = Tk()
    root.withdraw()

    # Ask user to select the two folders (each containing segmented images for one species).
    print("Select folder for the first species...")
    folder1 = filedialog.askdirectory(title="Select folder for first species")
    print("Select folder for the second species...")
    folder2 = filedialog.askdirectory(title="Select folder for second species")

    species1 = os.path.basename(folder1.rstrip(os.sep))
    species2 = os.path.basename(folder2.rstrip(os.sep))

    analyzer = ToneAnalyzer(white_threshold=230)

    print(f"\nProcessing images from species: {species1}")
    results1 = analyzer.process_folder(folder1)
    print(f"Processed {len(results1)} images from {species1}.")

    print(f"\nProcessing images from species: {species2}")
    results2 = analyzer.process_folder(folder2)
    print(f"Processed {len(results2)} images from {species2}.")

    # Save combined results to a CSV file.
    df = analyzer.save_results_to_csv(species1, species2, results1, results2)

    # Visualize the tone distribution for both species.
    analyzer.visualize_results(df)


if __name__ == "__main__":
    main()
