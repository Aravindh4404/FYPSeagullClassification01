# normalize_intensity_analysis.py
import cv2
import numpy as np
import sys
from pathlib import Path

# Add the root directory to Python path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
sys.path.append(str(root_dir))

from image_normalization import normalize_image, extract_region_from_normalized
from Features_Analysis.config import *  # Import configuration functions


def normalize_and_analyze_intensity(image_path, seg_path, species, file_name, region_type="wing"):
    """
    Normalizes the entire image first, then performs intensity analysis on the specified region.

    Args:
        image_path (str): Path to the original image
        seg_path (str): Path to the segmentation mask
        species (str): Species name
        file_name (str): Name of the file
        region_type (str): Type of region to analyze ("wing" or "wingtip")

    Returns:
        dict: Dictionary containing normalized intensity analysis results
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Convert to grayscale and normalize the entire image first
    gray_img, normalized_img = normalize_image(original_img)

    # Extract region from the normalized image
    region, mask = extract_region_from_normalized(normalized_img, segmentation_img, region_type)

    # Get pixels within the mask
    region_pixels = normalized_img[mask > 0]

    if len(region_pixels) == 0:
        print(f"No {region_type} region found in {file_name}")
        return None

    # Calculate intensity metrics
    results = {
        'image_name': file_name,
        'species': species,
        'mean_intensity': np.mean(region_pixels),
        'std_intensity': np.std(region_pixels),
        'median_intensity': np.median(region_pixels),
        'min_intensity': np.min(region_pixels),
        'max_intensity': np.max(region_pixels),
        'pixel_count': len(region_pixels)
    }

    return results


def analyze_normalized_wing_intensity(image_path, seg_path, species, file_name):
    """
    Analyzes normalized wing intensity for a single image.
    """
    return normalize_and_analyze_intensity(image_path, seg_path, species, file_name, "wing")


def analyze_normalized_wingtip_intensity(image_path, seg_path, species, file_name):
    """
    Analyzes normalized wingtip intensity for a single image.
    """
    return normalize_and_analyze_intensity(image_path, seg_path, species, file_name, "wingtip")


def analyze_normalized_wingtip_darkness(image_path, seg_path, species, file_name, wing_mean):
    """
    Analyzes normalized wingtip darkness compared to wing mean intensity.
    """
    # Load images
    original_img = cv2.imread(image_path)
    segmentation_img = cv2.imread(seg_path)

    if original_img is None or segmentation_img is None:
        print(f"Error loading images: {image_path} or {seg_path}")
        return None

    # Convert to grayscale and normalize the entire image first
    gray_img, normalized_img = normalize_image(original_img)

    # Extract wingtip region from the normalized image
    wingtip_region, wingtip_mask = extract_region_from_normalized(normalized_img, segmentation_img, "wingtip")

    # Get normalized pixels within the mask
    wingtip_pixels = normalized_img[wingtip_mask > 0]

    if len(wingtip_pixels) == 0:
        print(f"No wingtip region found in {file_name}")
        return None

    # Calculate darkness metrics using normalized values
    darker_pixels = np.sum(wingtip_pixels < wing_mean)
    total_pixels = len(wingtip_pixels)
    percentage_darker = (darker_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    return {
        'image_name': file_name,
        'species': species,
        'percentage_darker': percentage_darker,
        'darker_pixel_count': darker_pixels,
        'total_wingtip_pixels': total_pixels
    }