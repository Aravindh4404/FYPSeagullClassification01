###############################################################################
# CONFIGURATION
###############################################################################

from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent

SLATY_BACKED_IMG_DIR = BASE_DIR / "dataset" / "Original_Images" / "Slaty_Backed_Gull"
SLATY_BACKED_SEG_DIR = BASE_DIR / "dataset" / "Colored_Images" / "Slaty_Backed_Gull"

GLAUCOUS_WINGED_IMG_DIR = BASE_DIR / "dataset" / "Original_Images" / "Glaucous_Winged_Gull"
GLAUCOUS_WINGED_SEG_DIR = BASE_DIR / "dataset" / "Colored_Images" / "Glaucous_Winged_Gull"

# Number of images per species to process
S = 5

# Define the BGR colors for each region based on your RGB swatches:
REGION_COLORS = {
    "wingtip": (0, 255, 0),  # Green in RGB → (0, 255, 0) in BGR
    "wing": (0, 0, 255),  # Red in RGB → (0, 0, 255) in BGR
    "head": (255, 255, 0),  # Yellow in RGB → (0, 255, 255) in BGR
    "body": (0, 255, 255)  # Sky Blue in RGB → (235, 206, 135) in BGR
}

ENTIRE_BIRD_COLORS = list(REGION_COLORS.values())


def get_region_color(region_name):
    """
    Get the BGR color code for a specific region.

    Parameters:
        region_name (str): Name of the region (wingtip, wing, head, body)

    Returns:
        tuple: BGR color code
    """
    if region_name not in REGION_COLORS:
        raise ValueError(f"Region '{region_name}' not found. Available regions: {list(REGION_COLORS.keys())}")
    return REGION_COLORS[region_name]


def extract_region_mask(segmentation_img, region_name):
    """
    Extract a binary mask for a specific region from a segmentation image.

    Parameters:
        segmentation_img (numpy.ndarray): Segmentation image (BGR)
        region_name (str): Name of the region to extract

    Returns:
        numpy.ndarray: Binary mask where the region is 255 and background is 0
    """
    if region_name == "entire_bird":
        # Create a mask for the entire bird (all regions)
        mask = np.zeros(segmentation_img.shape[:2], dtype=np.uint8)
        for color in ENTIRE_BIRD_COLORS:
            region_mask = cv2.inRange(segmentation_img, color, color)
            mask = cv2.bitwise_or(mask, region_mask)
        return mask

    # Get the color for the specific region
    region_color = get_region_color(region_name)

    # Create a mask where the pixel value exactly matches the region color
    return cv2.inRange(segmentation_img, region_color, region_color)


def extract_region(original_img, segmentation_img, region_name):
    """
    Extract a specific region from an original image using the segmentation map.

    Parameters:
        original_img (numpy.ndarray): Original image (BGR)
        segmentation_img (numpy.ndarray): Segmentation image (BGR)
        region_name (str): Name of the region to extract

    Returns:
        numpy.ndarray: Extracted region with background set to black
    """
    # Check image dimensions match
    if original_img.shape[:2] != segmentation_img.shape[:2]:
        raise ValueError("Original and segmentation images must have the same dimensions")

    # Get the binary mask for the region
    mask = extract_region_mask(segmentation_img, region_name)

    # Apply the mask to the original image
    result = cv2.bitwise_and(original_img, original_img, mask=mask)

    return result, mask