# WORKS

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

        # Calculate aspect ratio
        aspect = w / h

        # Calculate new dimensions
        if aspect > target_w / target_h:  # Width is limiting factor
            new_w = target_w
            new_h = int(target_w / aspect)
        else:  # Height is limiting factor
            new_h = target_h
            new_w = int(target_h * aspect)

        return cv2.resize(image, (new_w, new_h))

    def select_folders(self) -> tuple:
        """Let user select two folders for comparison"""
        root = Tk()
        root.withdraw()

        print("Select first species folder...")
        folder1 = filedialog.askdirectory(title="Select first species folder")
        print("Select second species folder...")
        folder2 = filedialog.askdirectory(title="Select second species folder")

        return folder1, folder2

    def get_image_files(self, folder_path: str, max_images: int = None) -> List[str]:
        """Get list of image files from folder"""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(valid_extensions)]

        if max_images:
            return image_files[:max_images]
        return image_files

    def save_results_to_csv(self, results_dict: Dict):
        """Save the analysis results to a CSV file"""
        data = []

        # Process results for both folders
        for folder_key in ['folder1', 'folder2']:
            folder_name = results_dict[folder_key]['name']
            for image_result in results_dict[folder_key]['results']:
                image_name = image_result['image']
                for region in image_result['regions']:
                    data.append({
                        'Species': folder_name,
                        'Image': image_name,
                        'Region': f"Region {region['region_id']}",
                        'Darkness_Value': region['darkness_value']
                    })

        # Create DataFrame and save to CSV
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

        # Reset regions for new image
        self.regions = []
        self.masks = []

        # Resize image
        self.image = self.resize_image(original_image)

        # Create window only if it doesn't exist
        window_name = 'Region Selection'
        cv2.namedWindow(window_name)
        self.drawing_image = self.image.copy()
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        # Set mouse callback
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print(f"\nSelect regions for {os.path.basename(img_path)}:")
        print("- Hold left mouse button and draw to create regions")
        print("- Release mouse button to complete a region")
        print("- Press 'a' to accept regions")
        print("- Press 'r' to reset selections")
        print("- Press 'q' to skip this image")

        while True:
            # Show the current image name in the window
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
                # Scale regions back to original image size
                scale_x = original_image.shape[1] / self.image.shape[1]
                scale_y = original_image.shape[0] / self.image.shape[0]
                scaled_regions = [[(int(x * scale_x), int(y * scale_y))
                                   for x, y in region]
                                  for region in self.regions]
                return scaled_regions

        return None

    def process_folders(self, folder1: str, folder2: str, max_images: int = None):
        """Process all images in both folders"""
        # Get image lists
        images1 = self.get_image_files(folder1, max_images)
        images2 = self.get_image_files(folder2, max_images)

        # Process images and collect results
        results_dict = {
            'folder1': {'name': os.path.basename(folder1), 'results': []},
            'folder2': {'name': os.path.basename(folder2), 'results': []}
        }

        try:
            # Process first folder
            print(f"\nProcessing {results_dict['folder1']['name']}...")
            for img_path in images1:
                print(f"\nProcessing {os.path.basename(img_path)}")
                regions = self.select_regions_for_image(img_path)
                if regions:
                    img = cv2.imread(img_path)
                    if img is not None:
                        results = self.analyze_regions(img, regions)
                        results_dict['folder1']['results'].append({
                            'image': os.path.basename(img_path),
                            'regions': results
                        })
                        self.visualize_individual_image(img, results, os.path.basename(img_path))

            # Process second folder
            print(f"\nProcessing {results_dict['folder2']['name']}...")
            for img_path in images2:
                print(f"\nProcessing {os.path.basename(img_path)}")
                regions = self.select_regions_for_image(img_path)
                if regions:
                    img = cv2.imread(img_path)
                    if img is not None:
                        results = self.analyze_regions(img, regions)
                        results_dict['folder2']['results'].append({
                            'image': os.path.basename(img_path),
                            'regions': results
                        })
                        self.visualize_individual_image(img, results, os.path.basename(img_path))

            if results_dict['folder1']['results'] and results_dict['folder2']['results']:
                self.visualize_results(results_dict)
                self.save_results_to_csv(results_dict)

        finally:
            cv2.destroyAllWindows()

    def analyze_regions(self, image: np.ndarray, regions: List) -> List[Dict]:
        """Calculate darkness metrics for selected regions"""
        results = []
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for i, region_points in enumerate(regions):
            # Create mask for this region
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            points = np.array(region_points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

            # Get the region using the mask
            region = cv2.bitwise_and(gray_image, gray_image, mask=mask)
            non_zero_pixels = region[mask > 0]

            if len(non_zero_pixels) > 0:
                avg_intensity = np.mean(non_zero_pixels)
                results.append({
                    'region_id': i + 1,
                    'darkness_value': avg_intensity,  # 0-255 scale
                    'contour': region_points
                })

        return results

    def visualize_individual_image(self, image: np.ndarray, results: List[Dict], image_name: str):
        """Visualize results for a single image"""
        # Resize image for display
        display_image = self.resize_image(image)

        # Create figure with two subplots
        plt.figure(figsize=(15, 5))

        # Plot the image with regions
        plt.subplot(1, 2, 1)
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        plt.imshow(display_image_rgb)

        # Scale regions for display
        scale_x = display_image.shape[1] / image.shape[1]
        scale_y = display_image.shape[0] / image.shape[0]

        for result in results:
            # Scale contour points
            scaled_contour = [(int(x * scale_x), int(y * scale_y))
                              for x, y in result['contour']]
            points = np.array(scaled_contour)
            plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=2)

            # Add text
            x, y = np.mean(points[:, 0]), np.mean(points[:, 1])
            plt.text(x, y, f"Region {result['region_id']}\n{result['darkness_value']:.1f}",
                     color='red', fontsize=10, ha='center')

        plt.title(f"Regions - {image_name}")
        plt.axis('off')

        # Plot darkness values
        plt.subplot(1, 2, 2)
        region_nums = [r['region_id'] for r in results]
        darkness_values = [r['darkness_value'] for r in results]

        bars = plt.bar(region_nums, darkness_values)
        plt.title(f"Region Darkness Values - {image_name}")
        plt.xlabel("Region Number")
        plt.ylabel("Darkness Value (0-255)")
        plt.ylim(0, 255)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def visualize_results(self, results_dict: Dict):
        """Visualize the average results across all images"""
        folder1_name = results_dict['folder1']['name']
        folder2_name = results_dict['folder2']['name']

        # Get the maximum number of regions used across all images
        max_regions = max(
            max(len(r['regions']) for r in results_dict['folder1']['results']),
            max(len(r['regions']) for r in results_dict['folder2']['results'])
        )

        # Calculate averages for each region
        folder1_avgs = []
        folder2_avgs = []

        for region_idx in range(max_regions):
            # Calculate average for folder1
            folder1_values = [r['regions'][region_idx]['darkness_value']
                              for r in results_dict['folder1']['results']
                              if region_idx < len(r['regions'])]
            folder1_avg = np.mean(folder1_values) if folder1_values else 0

            # Calculate average for folder2
            folder2_values = [r['regions'][region_idx]['darkness_value']
                              for r in results_dict['folder2']['results']
                              if region_idx < len(r['regions'])]
            folder2_avg = np.mean(folder2_values) if folder2_values else 0

            folder1_avgs.append(folder1_avg)
            folder2_avgs.append(folder2_avg)

        # Create plot
        plt.figure(figsize=(12, 6))
        x = np.arange(max_regions)
        width = 0.35

        bars1 = plt.bar(x - width / 2, folder1_avgs, width, label=folder1_name)
        bars2 = plt.bar(x + width / 2, folder2_avgs, width, label=folder2_name)

        plt.xlabel('Region Number')
        plt.ylabel('Average Darkness Value (0-255)')
        plt.title('Comparison of Region Darkness Between Species')
        plt.xticks(x, [f'Region {i + 1}' for i in range(max_regions)])
        plt.legend()

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only add label if there's a value
                    plt.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:.1f}',
                             ha='center', va='bottom')

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
                cv2.imshow('Draw Regions (Press "a" to accept, "r" to reset, "q" to skip)',
                           self.drawing_image)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and len(self.current_region) > 2:
                self.drawing = False

                # Create mask for this region
                region_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                points = np.array(self.current_region, dtype=np.int32)
                cv2.fillPoly(region_mask, [points], 255)

                # Add region number text
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
    # Create analyzer instance with target size for display
    analyzer = DarknessAnalyzer(target_size=(800, 600))

    # Let user select folders
    folder1, folder2 = analyzer.select_folders()

    # Set maximum number of images to process from each folder
    max_images = 1  # Change this value to process more or fewer images

    # Process the folders
    analyzer.process_folders(folder1, folder2, max_images)


if __name__ == "__main__":
    main()