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

        return image_files[:max_images] if max_images else image_files

    def save_results_to_csv(self, results_dict: Dict):
        """Save the analysis results to a CSV file"""
        data = []

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
        print("- 'a' to accept, 'r' to reset, 'q' to skip")

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
                scale_x = original_image.shape[1] / self.image.shape[1]
                scale_y = original_image.shape[0] / self.image.shape[0]
                scaled_regions = [[(int(x * scale_x), int(y * scale_y))
                                   for x, y in region]
                                  for region in self.regions]
                return scaled_regions

        return None

    def process_folders(self, folder1: str, folder2: str, max_images: int = None):
        """Process all images in both folders"""
        images1 = self.get_image_files(folder1, max_images)
        images2 = self.get_image_files(folder2, max_images)

        results_dict = {
            'folder1': {'name': os.path.basename(folder1), 'results': []},
            'folder2': {'name': os.path.basename(folder2), 'results': []}
        }

        try:
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
                self.visualize_comparison(results_dict)
                self.save_results_to_csv(results_dict)

        finally:
            cv2.destroyAllWindows()

    def analyze_regions(self, image: np.ndarray, regions: List) -> List[Dict]:
        """Calculate darkness metrics for selected regions with normalization"""
        results = []
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize the grayscale image using min-max normalization
        min_val = np.min(gray_image)
        max_val = np.max(gray_image)
        if max_val > min_val:
            # Scale values to range 0-255
            normalized_gray = ((gray_image - min_val) / (max_val - min_val)) * 255
            normalized_gray = normalized_gray.astype(np.uint8)
        else:
            normalized_gray = gray_image

        for i, region_points in enumerate(regions):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            points = np.array(region_points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

            # Use the normalized grayscale image for analysis
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
                     f'{height:.1f}',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def visualize_comparison(self, results_dict: Dict):
        """Visualize comparative analysis between folders"""
        folder1_name = results_dict['folder1']['name']
        folder2_name = results_dict['folder2']['name']

        # Collect all unique region indices present in either folder
        folder1_regions = set()
        for result in results_dict['folder1']['results']:
            folder1_regions.update(range(len(result['regions'])))

        folder2_regions = set()
        for result in results_dict['folder2']['results']:
            folder2_regions.update(range(len(result['regions'])))

        all_regions = sorted(folder1_regions.union(folder2_regions))

        # Prepare data for plotting
        regions = []
        folder1_means, folder1_stds, folder1_counts = [], [], []
        folder2_means, folder2_stds, folder2_counts = [], [], []

        for region_idx in all_regions:
            # Folder1 calculations
            f1_values = [
                img_result['regions'][region_idx]['darkness_value']
                for img_result in results_dict['folder1']['results']
                if region_idx < len(img_result['regions'])
            ]
            f1_mean = np.mean(f1_values) if f1_values else np.nan
            f1_std = np.std(f1_values, ddof=1) if f1_values else np.nan
            f1_count = len(f1_values)

            # Folder2 calculations
            f2_values = [
                img_result['regions'][region_idx]['darkness_value']
                for img_result in results_dict['folder2']['results']
                if region_idx < len(img_result['regions'])
            ]
            f2_mean = np.mean(f2_values) if f2_values else np.nan
            f2_std = np.std(f2_values, ddof=1) if f2_values else np.nan
            f2_count = len(f2_values)

            if not np.isnan(f1_mean) or not np.isnan(f2_mean):
                regions.append(f"Region {region_idx+1}")
                folder1_means.append(f1_mean)
                folder1_stds.append(f1_std)
                folder1_counts.append(f1_count)
                folder2_means.append(f2_mean)
                folder2_stds.append(f2_std)
                folder2_counts.append(f2_count)

        # Plotting
        x = np.arange(len(regions))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        rects1 = ax.bar(x - width/2, folder1_means, width,
                        yerr=folder1_stds, label=folder1_name,
                        alpha=0.7, capsize=5)
        rects2 = ax.bar(x + width/2, folder2_means, width,
                        yerr=folder2_stds, label=folder2_name,
                        alpha=0.7, capsize=5)

        ax.set_ylabel('Darkness Value (0-255)\n(lower = darker)')
        ax.set_title('Comparative Darkness Analysis by Region')
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.legend()

        # Add count labels
        for rect, count in zip(rects1, folder1_counts):
            if not np.isnan(rect.get_height()):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2, height/2,
                        f'n={count}', ha='center', va='center',
                        color='white', fontweight='bold')

        for rect, count in zip(rects2, folder2_counts):
            if not np.isnan(rect.get_height()):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2, height/2,
                        f'n={count}', ha='center', va='center',
                        color='white', fontweight='bold')

        plt.ylim(0, 255)
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
    folder1, folder2 = analyzer.select_folders()
    analyzer.process_folders(folder1, folder2, max_images=5)  # Process up to 5 images per folder


if __name__ == "__main__":
    main()
