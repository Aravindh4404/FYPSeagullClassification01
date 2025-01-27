import cv2
import numpy as np
import matplotlib.pyplot as plt


class DarknessAnalyzer:
    def __init__(self):
        self.regions = []
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.selected_regions = []
        self.image = None
        self.drawing_image = None

    def select_regions(self, img_path):
        """Initialize window and handle region selection"""
        # Read image
        self.image = cv2.imread(img_path)
        if self.image is None:
            print(f"Error: Could not load image from {img_path}")
            return

        # Create window
        window_name = 'Select Regions (Press "a" to analyze, "r" to reset, "q" to quit)'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        self.drawing_image = self.image.copy()

        print("\nInstructions:")
        print("- Click and drag to select regions")
        print("- Press 'a' to analyze selected regions")
        print("- Press 'r' to reset selections")
        print("- Press 'q' to quit")

        while True:
            cv2.imshow(window_name, self.drawing_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.drawing_image = self.image.copy()
                self.selected_regions = []
            elif key == ord('a'):
                cv2.destroyAllWindows()
                self.analyze_and_visualize()
                break

        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_img = self.drawing_image.copy()
                cv2.rectangle(temp_img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Select Regions (Press "a" to analyze, "r" to reset, "q" to quit)', temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)

            # Draw permanent rectangle
            cv2.rectangle(self.drawing_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.drawing_image, f'Region {len(self.selected_regions) + 1}',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            self.selected_regions.append((x1, y1, x2, y2))

    def analyze_regions(self):
        """Calculate darkness metrics for selected regions"""
        results = []
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        for i, (x1, y1, x2, y2) in enumerate(self.selected_regions):
            region = gray_image[y1:y2, x1:x2]
            avg_intensity = np.mean(region)
            relative_darkness = 1 - (avg_intensity / 255.0)

            results.append({
                'region_id': i + 1,
                'relative_darkness': relative_darkness,
                'avg_intensity': avg_intensity,
                'bbox': (x1, y1, x2, y2)
            })

        return results

    def analyze_and_visualize(self):
        """Analyze selected regions and display results"""
        if not self.selected_regions:
            print("No regions selected!")
            return

        results = self.analyze_regions()

        # Convert BGR to RGB for matplotlib
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(15, 5))

        # Original image with regions
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            plt.gca().add_patch(plt.Rectangle((x1, y1),
                                              x2 - x1,
                                              y2 - y1,
                                              fill=False,
                                              color='red',
                                              linewidth=2))
            plt.text(x1, y1 - 5, f"Region {result['region_id']}\nDarkness: {result['relative_darkness']:.2f}",
                     color='red', fontsize=10)
        plt.title("Selected Regions with Darkness Scores")
        plt.axis('off')

        # Darkness comparison bar plot
        plt.subplot(1, 2, 2)
        region_nums = [r['region_id'] for r in results]
        darkness_scores = [r['relative_darkness'] for r in results]
        plt.bar(region_nums, darkness_scores)
        plt.title("Relative Darkness by Region")
        plt.xlabel("Region Number")
        plt.ylabel("Darkness Score (0=white, 1=black)")

        plt.tight_layout()
        plt.show()

        # Print detailed results
        print("\nRegion Analysis Results:")
        for result in results:
            print(f"\nRegion {result['region_id']}:")
            print(f"  Relative Darkness: {result['relative_darkness']:.3f}")
            print(f"  Average Intensity: {result['avg_intensity']:.3f}")
            x1, y1, x2, y2 = result['bbox']
            print(f"  Location: (x: {x1}-{x2}, y: {y1}-{y2})")


def main():
    # Use your image path here
    image_path = r'D:\Glaucous-winged Gull - Larus glaucescens - Media Search - Macaulay Library and eBird\1200 - 2024-10-22T213907.711.jpg'  # Updated path

    # Create analyzer instance and start selection
    analyzer = DarknessAnalyzer()
    analyzer.select_regions(image_path)

if __name__ == "__main__":
    main()

