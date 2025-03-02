import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from Features_Analysis.config import SLATY_BACKED_IMG_DIR, SLATY_BACKED_SEG_DIR
from Features_Analysis.config import extract_region_mask, extract_region

def main():
    # List available files in the image and segmentation directories
    img_files = sorted(os.listdir(SLATY_BACKED_IMG_DIR))
    seg_files = sorted(os.listdir(SLATY_BACKED_SEG_DIR))

    if not img_files or not seg_files:
        print("No images or segmentation files found!")
        return

    # For testing, use the first image file that exists in both directories.
    # (Assuming file names match between original images and segmentation images)
    filename = img_files[0]
    sample_img_path = Path(SLATY_BACKED_IMG_DIR) / filename
    sample_seg_path = Path(SLATY_BACKED_SEG_DIR) / filename

    sample_img = cv2.imread(str(sample_img_path))
    sample_seg = cv2.imread(str(sample_seg_path))

    if sample_img is None or sample_seg is None:
        print("Error loading sample image or segmentation image!")
        return

    # Test extraction of masks
    wingtip_mask = extract_region_mask(sample_seg, "wingtip")
    entire_bird_mask = extract_region_mask(sample_seg, "entire_bird")

    # Test extraction of region (wingtip)
    wingtip_region, _ = extract_region(sample_img, sample_seg, "wingtip")

    # Plot the results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(sample_seg, cv2.COLOR_BGR2RGB))
    plt.title("Segmentation Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(wingtip_mask, cmap="gray")
    plt.title("Wingtip Mask")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(entire_bird_mask, cmap="gray")
    plt.title("Entire Bird Mask")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(wingtip_region, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Wingtip Region")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
