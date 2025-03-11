###############################################################################
# CONFIGURATION
###############################################################################
import os
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

# Dictionary of species -> (image directory, segmentation directory)
SPECIES = {
    "Slaty_Backed_Gull": {
        "img_dir": SLATY_BACKED_IMG_DIR,
        "seg_dir": SLATY_BACKED_SEG_DIR,
    },
    "Glaucous_Winged_Gull": {
        "img_dir": GLAUCOUS_WINGED_IMG_DIR,
        "seg_dir": GLAUCOUS_WINGED_SEG_DIR,
    },
}


###############################################################################
# HELPER FUNCTIONS
###############################################################################
def get_image_paths(species):
    """
    For the given species, returns a list of (image_path, segmentation_path) pairs
    that share the same base filename.
    """
    img_dir = SPECIES[species]["img_dir"]
    seg_dir = SPECIES[species]["seg_dir"]

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    seg_files = sorted([f for f in os.listdir(seg_dir)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    paired_files = []
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        match_seg = [f for f in seg_files if os.path.splitext(f)[0] == base_name]
        if match_seg:
            paired_files.append((
                os.path.join(img_dir, img_file),
                os.path.join(seg_dir, match_seg[0])
            ))
    return paired_files


def get_region_masks(segmentation):
    """
    Returns a dictionary {region_name: mask} and {region_name: stats}
    from the color-coded segmentation image.
    """
    region_masks = {}
    region_stats = {}

    for region_name in REGION_COLORS:
        mask = extract_region_mask(segmentation, region_name)
        region_masks[region_name] = mask

        # Optional: find region center
        pixels = cv2.countNonZero(mask)
        if pixels > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    region_stats[region_name] = {"center": (cx, cy), "pixels": pixels}
    return region_masks, region_stats


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

# use this below
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