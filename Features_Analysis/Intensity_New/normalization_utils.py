import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from Features_Analysis.config import *

def to_grayscale(image):
    """
    Ensures the image is in grayscale before processing.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()

def minmax_normalize(image):
    """
    Scales grayscale pixel intensities to [0, 255] using per-image Min-Max normalization.
    This is the primary normalization method used throughout the analysis.
    """
    gray = to_grayscale(image)
    min_val = np.min(gray)
    max_val = np.max(gray)
    norm = (gray - min_val) / (max_val - min_val + 1e-8)  # scale to [0,1]
    return (norm * 255).astype(np.uint8)

def extract_region(original_img, segmentation_img, region_type):
    """
    Extracts a specific region (wing or wingtip) from the image using the segmentation mask.
    
    Args:
        original_img: The original image
        segmentation_img: The segmentation mask image
        region_type: Either "wing" or "wingtip"
        
    Returns:
        tuple: (region_image, region_mask)
    """
    # Convert segmentation image to grayscale if it's not already
    if len(segmentation_img.shape) == 3:
        seg_gray = cv2.cvtColor(segmentation_img, cv2.COLOR_BGR2GRAY)
    else:
        seg_gray = segmentation_img.copy()
    
    # Create binary mask for the specified region
    if region_type == "wing":
        # Wing is typically the largest connected component
        _, binary = cv2.threshold(seg_gray, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Find the largest component (excluding background)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            region_mask = (labels == largest_label).astype(np.uint8) * 255
        else:
            region_mask = np.zeros_like(seg_gray)
    elif region_type == "wingtip":
        # Wingtip is typically at the edge of the wing
        _, binary = cv2.threshold(seg_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find the wing mask first
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            wing_mask = (labels == largest_label).astype(np.uint8) * 255
            
            # Dilate the wing mask to find the edge
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(wing_mask, kernel, iterations=1)
            
            # The wingtip is the difference between dilated and original
            region_mask = cv2.bitwise_xor(dilated, wing_mask)
        else:
            region_mask = np.zeros_like(seg_gray)
    else:
        raise ValueError(f"Unknown region type: {region_type}")
    
    # Apply the mask to the original image
    region_image = cv2.bitwise_and(original_img, original_img, mask=region_mask)
    
    return region_image, region_mask

def load_and_normalize_image(image_path, seg_path=None):
    """
    Loads an image and its segmentation mask, then normalizes the image.
    
    Args:
        image_path: Path to the original image
        seg_path: Path to the segmentation mask (optional)
        
    Returns:
        tuple: (normalized_image, segmentation_image)
    """
    # Load original image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error loading image: {image_path}")
        return None, None
    
    # Convert to grayscale and normalize
    gray_img = to_grayscale(original_img)
    normalized_img = minmax_normalize(gray_img)
    
    # Load segmentation image if provided
    segmentation_img = None
    if seg_path:
        segmentation_img = cv2.imread(seg_path)
        if segmentation_img is None:
            print(f"Error loading segmentation image: {seg_path}")
            return normalized_img, None
    
    return normalized_img, segmentation_img 