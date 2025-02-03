# DOES NOT WORK PROPERLY

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class DarknessAnalyzer:
    def __init__(self):
        self.regions = []
        self.drawing = False
        self.current_region = []
        self.masks = []
        self.image = None
        self.drawing_image = None
        self.mask = None

    def select_regions(self, img_path):
        """Initialize window and handle region selection"""
        # Read image
        self.image = cv2.imread(img_path)
        if self.image is None:
            print(f"Error: Could not load image from {img_path}")
            return None

        # Create window
        window_name = 'Draw Regions (Press "a" to analyze, "r" to reset, "q" to quit)'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        self.drawing_image = self.image.copy()
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        print("\nInstructions:")
        print("- Hold left mouse button and draw to create regions")
        print("- Release mouse button to complete a region")
        print("- Press 'a' to analyze selected regions")
        print("- Press 'r' to reset selections")
        print("- Press 'q' to quit")

        while True:
            cv2.imshow(window_name, self.drawing_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
            elif key == ord('r'):
                self.drawing_image = self.image.copy()
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.masks = []
                self.regions = []
            elif key == ord('a'):
                cv2.destroyAllWindows()
                return self.analyze_regions()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for freehand drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_region = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_region.append((x, y))
                # Draw line between last two points
                if len(self.current_region) >= 2:
                    cv2.line(self.drawing_image,
                             self.current_region[-2],
                             self.current_region[-1],
                             (0, 255, 0), 2)
                cv2.imshow('Draw Regions (Press "a" to analyze, "r" to reset, "q" to quit)',
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

    def analyze_regions(self):
        """Calculate darkness metrics for selected regions"""
        results = []
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        for i, mask in enumerate(self.masks):
            # Get the region using the mask
            region = cv2.bitwise_and(gray_image, gray_image, mask=mask)
            non_zero_pixels = region[mask > 0]

            if len(non_zero_pixels) > 0:
                avg_intensity = np.mean(non_zero_pixels)  # Intensity in 0-255 format

                results.append({
                    'region_id': i + 1,
                    'avg_intensity': avg_intensity,
                    'contour': self.regions[i]
                })

        return results


def process_folder(folder_path, max_images_to_process):
    """Process all images in a folder and analyze selected regions"""
    analyzer = DarknessAnalyzer()
    results = []

    # Get list of images in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:max_images_to_process]  # Limit number of images

    for idx, img_file in enumerate(image_files):
        print(f"\nProcessing image {idx + 1}/{len(image_files)}: {img_file}")
        img_path = os.path.join(folder_path, img_file)
        region_results = analyzer.select_regions(img_path)

        if region_results:
            results.append({
                'image_name': img_file,
                'region_results': region_results,
                'avg_darkness': np.mean([r['avg_intensity'] for r in region_results])
            })

    return results


def compare_folders(folder1, folder2, max_images_to_process):
    """Compare darkness levels between two folders"""
    print(f"Processing folder 1: {folder1}")
    folder1_results = process_folder(folder1, max_images_to_process)

    print(f"\nProcessing folder 2: {folder2}")
    folder2_results = process_folder(folder2, max_images_to_process)

    # Compare average darkness
    folder1_avg = np.mean([r['avg_darkness'] for r in folder1_results])
    folder2_avg = np.mean([r['avg_darkness'] for r in folder2_results])

    print("\nComparison Results:")
    print(f"Average darkness in {folder1}: {folder1_avg:.2f}")
    print(f"Average darkness in {folder2}: {folder2_avg:.2f}")

    if folder1_avg < folder2_avg:
        print(f"{folder2} has darker regions on average.")
    else:
        print(f"{folder1} has darker regions on average.")

    # Plot comparison
    plt.bar([folder1, folder2], [folder1_avg, folder2_avg], color=['blue', 'orange'])
    plt.title("Average Darkness Comparison")
    plt.ylabel("Average Intensity (0-255, lower is darker)")
    plt.show()


def main():
    # Set folder paths
    slaty_backed_gull_folder = r"D:\FYP DATASETS USED\Dataset HQ\HQSBNGW\train\Glaucous_Winged_Gull"  # Update with your path
    glaucous_winged_gull_folder = r"D:\FYP DATASETS USED\Dataset HQ\HQSBNGW\train\Slaty_Backed_Gull"  # Update with your path

    # Set maximum number of images to process per folder
    max_images_to_process = 10  # Change this as needed

    # Compare folders
    compare_folders(slaty_backed_gull_folder, glaucous_winged_gull_folder, max_images_to_process)


if __name__ == "__main__":
    main()