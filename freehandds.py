import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog, Tk
from typing import List, Dict
import pandas as pd


class DarknessAnalyzer:
    def __init__(self, target_size=(800, 600)):
        self.regions = []
        self.drawing = False
        self.current_region = []
        self.masks = []
        self.image = None
        self.drawing_image = None
        self.mask = None
        self.target_size = target_size

    def resize_image(self, image):
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        aspect = w / h
        if aspect > target_w / target_h:
            new_w = target_w
            new_h = int(target_w / aspect)
        else:
            new_h = target_h
            new_w = int(target_h * aspect)

        return cv2.resize(image, (new_w, new_h))

    def select_parent_folder(self) -> str:
        """Let the user select a single parent folder.
           All subfolders inside this folder are automatically selected."""
        root = Tk()
        root.withdraw()
        print("Select the parent folder that contains the species subfolders...")
        parent_folder = filedialog.askdirectory(title="Select Parent Folder")
        return parent_folder

    def get_image_files(self, folder_path: str, max_images: int = None) -> List[str]:
        """Get list of image files from folder"""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(valid_extensions)]
        return image_files[:max_images] if max_images else image_files

    def save_results_to_csv(self, results_dict: Dict):
        """Save the analysis results to a CSV file"""
        data = []
        for species_key, species_data in results_dict.items():
            species_name = species_data['name']
            for image_result in species_data['results']:
                image_name = image_result['image']
                for region in image_result['regions']:
                    data.append({
                        'Species': species_name,
                        'Image': image_name,
                        'Region': f"Region {region['region_id']}",
                        'Darkness_Value': region['darkness_value']
                    })
        df = pd.DataFrame(data)
        output_file = 'darkness_analysis_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    def select_regions_for_image(self, img_path: str):
        """Select regions for a single image"""
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"Error: Could not load image from {img_path}")
            return None

        self.regions = []
        self.masks = []
        self.image = self.resize_image(original_image)

        window_name = 'Region Selection'
        cv2.namedWindow(window_name)
        self.drawing_image = self.image.copy()
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        cv2.setMouseCallback(window_name, self.mouse_callback)

        print(f"\nSelect regions for {os.path.basename(img_path)}:")
        print("- Hold left mouse button to draw")
        print("- Release to complete region")
        print("- Press 'a' to accept, 'r' to reset, or 'q' to skip this image")

        while True:
            display_image = self.drawing_image.copy()
            cv2.putText(display_image, os.path.basename(img_path),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                return None
            elif key == ord('r'):
                self.drawing_image = self.image.copy()
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.masks = []
                self.regions = []
            elif key == ord('a') and len(self.regions) > 0:
                # Scale regions back to original image size.
                scale_x = original_image.shape[1] / self.image.shape[1]
                scale_y = original_image.shape[0] / self.image.shape[0]
                scaled_regions = [[(int(x * scale_x), int(y * scale_y))
                                   for x, y in region]
                                  for region in self.regions]
                return scaled_regions

        return None

    def process_parent_folder(self, parent_folder: str, max_images: int = None):
        """
        Process all images in all subfolders of the parent folder.
        Each subfolder is assumed to represent a species.
        """
        # Get a list of subfolders in the parent folder.
        subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder)
                      if os.path.isdir(os.path.join(parent_folder, d))]
        if not subfolders:
            print("No subfolders found in the selected parent folder.")
            return

        results_dict = {}
        for subfolder in subfolders:
            species_name = os.path.basename(subfolder)
            results_dict[subfolder] = {'name': species_name, 'results': []}
            images = self.get_image_files(subfolder, max_images)
            print(f"\nProcessing species: {species_name} ({len(images)} images)")
            for img_path in images:
                print(f"\nProcessing image: {os.path.basename(img_path)}")
                regions = self.select_regions_for_image(img_path)
                if regions:
                    img = cv2.imread(img_path)
                    if img is not None:
                        results = self.analyze_regions(img, regions)
                        results_dict[subfolder]['results'].append({
                            'image': os.path.basename(img_path),
                            'regions': results
                        })
                        self.visualize_individual_image(img, results, os.path.basename(img_path))
        # Visualize comparative analysis across species.
        if results_dict:
            self.visualize_comparison(results_dict)
            self.save_results_to_csv(results_dict)

        cv2.destroyAllWindows()

    def analyze_regions(self, image: np.ndarray, regions: List) -> List[Dict]:
        """Calculate darkness metrics for selected regions with normalization"""
        results = []
        # Convert image to grayscale.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize the grayscale image using min‑max normalization.
        min_val = np.min(gray_image)
        max_val = np.max(gray_image)
        if max_val > min_val:
            normalized_gray = ((gray_image - min_val) / (max_val - min_val)) * 255
            normalized_gray = normalized_gray.astype(np.uint8)
        else:
            normalized_gray = gray_image

        for i, region_points in enumerate(regions):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            points = np.array(region_points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

            # Use the normalized grayscale image for analysis.
            region = cv2.bitwise_and(normalized_gray, normalized_gray, mask=mask)
            non_zero_pixels = region[mask > 0]

            if len(non_zero_pixels) > 0:
                avg_intensity = np.mean(non_zero_pixels)
                results.append({
                    'region_id': i + 1,
                    'darkness_value': avg_intensity,
                    'contour': region_points
                })

        return results

    def visualize_individual_image(self, image: np.ndarray, results: List[Dict], image_name: str):
        """Visualize results for a single image"""
        display_image = self.resize_image(image)
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        plt.imshow(display_image_rgb)
        scale_x = display_image.shape[1] / image.shape[1]
        scale_y = display_image.shape[0] / image.shape[0]
        for result in results:
            scaled_contour = [(int(x * scale_x), int(y * scale_y))
                              for x, y in result['contour']]
            points = np.array(scaled_contour)
            plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=2)
            x, y = np.mean(points[:, 0]), np.mean(points[:, 1])
            plt.text(x, y, f"Region {result['region_id']}\n{result['darkness_value']:.1f}",
                     color='red', fontsize=10, ha='center')
        plt.title(f"Regions - {image_name}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        region_nums = [r['region_id'] for r in results]
        darkness_values = [r['darkness_value'] for r in results]
        bars = plt.bar(region_nums, darkness_values)
        plt.title(f"Darkness Values - {image_name}")
        plt.xlabel("Region Number")
        plt.ylabel("Darkness Value (0-255)\n(lower = darker)")
        plt.ylim(0, 255)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

    def visualize_comparison(self, results_dict: Dict):
        """
        Visualize comparative analysis across all species.
        For each image, if multiple regions were selected, the darkness values are averaged.
        Then, for each species (subfolder), an overall average and standard deviation is computed,
        and a final bar chart with one bar per species is displayed.
        """
        species_names = []
        species_means = []
        species_stds = []
        species_counts = []

        # Loop through each species (subfolder).
        for key, species_data in results_dict.items():
            species_name = species_data['name']
            image_averages = []
            for image_result in species_data['results']:
                region_values = [region['darkness_value'] for region in image_result['regions']]
                if region_values:
                    image_avg = np.mean(region_values)
                    image_averages.append(image_avg)
            if image_averages:
                overall_mean = np.mean(image_averages)
                overall_std = np.std(image_averages, ddof=1) if len(image_averages) > 1 else 0
                species_names.append(species_name)
                species_means.append(overall_mean)
                species_stds.append(overall_std)
                species_counts.append(len(image_averages))

        # Plot a bar chart with one bar per species.
        fig, ax = plt.subplots(figsize=(max(8, len(species_names)*1.5), 6))
        bars = ax.bar(species_names, species_means, yerr=species_stds, capsize=10, alpha=0.7)
        ax.set_ylabel('Average Darkness Value (0-255)\n(lower = darker)')
        ax.set_title('Comparative Average Darkness Analysis Across Species')
        ax.set_ylim(0, 255)
        # Add count labels on each bar.
        for bar, count in zip(bars, species_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={count}', ha='center', va='bottom', color='black', fontweight='bold')
        plt.tight_layout()
        plt.show()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for freehand drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_region = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_region.append((x, y))
                if len(self.current_region) >= 2:
                    cv2.line(self.drawing_image,
                             self.current_region[-2],
                             self.current_region[-1],
                             (0, 255, 0), 2)
                cv2.imshow('Region Selection', self.drawing_image)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and len(self.current_region) > 2:
                self.drawing = False
                region_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                points = np.array(self.current_region, dtype=np.int32)
                cv2.fillPoly(region_mask, [points], 255)
                M = cv2.moments(points)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(self.drawing_image, f'Region {len(self.masks) + 1}',
                                (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                self.masks.append(region_mask)
                self.regions.append(self.current_region)
                self.current_region = []


def main():
    analyzer = DarknessAnalyzer(target_size=(800, 600))
    parent_folder = analyzer.select_parent_folder()
    analyzer.process_parent_folder(parent_folder, max_images=5)  # Process up to 5 images per species subfolder


if __name__ == "__main__":
    main()
