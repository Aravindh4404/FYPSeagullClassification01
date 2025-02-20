import cv2
import numpy as np
import matplotlib.pyplot as plt


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
            return

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
                break
            elif key == ord('r'):
                self.drawing_image = self.image.copy()
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                self.masks = []
                self.regions = []
            elif key == ord('a'):
                cv2.destroyAllWindows()
                self.analyze_and_visualize()
                break

        cv2.destroyAllWindows()

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
                avg_intensity = np.mean(non_zero_pixels)
                relative_darkness = 1 - (avg_intensity / 255.0)

                # Get bounding box for visualization
                points = np.array(self.regions[i], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(points)

                results.append({
                    'region_id': i + 1,
                    'relative_darkness': relative_darkness,
                    'avg_intensity': avg_intensity,
                    'contour': self.regions[i],
                    'bbox': (x, y, x + w, y + h)
                })

        return results

    def analyze_and_visualize(self):
        """Analyze selected regions and display results"""
        if not self.masks:
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
            # Draw the actual contour instead of rectangle
            points = np.array(result['contour'])
            plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=2)

            # Add text at the bounding box top-left corner
            x1, y1, _, _ = result['bbox']
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

def main():
    # Use your image path here
    image_path = r'D:\Glaucous-winged Gull - Larus glaucescens - Media Search - Macaulay Library and eBird\1200 - 2024-10-22T213907.711.jpg'  # Updated path

    # Create analyzer instance and start selection
    analyzer = DarknessAnalyzer()
    analyzer.select_regions(image_path)


if __name__ == "__main__":
    main()