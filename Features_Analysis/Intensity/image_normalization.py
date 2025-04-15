# image_normalization.py
import cv2
import numpy as np


def minmax_normalize(image):
    """
    Apply min-max normalization to scale pixel values to 0-255 range.

    Args:
        image (numpy.ndarray): Input grayscale image

    Returns:
        numpy.ndarray: Normalized image with values in 0-255 range
    """
    # Ensure the image is in the right format
    if len(image.shape) > 2 and image.shape[2] > 1:
        raise ValueError("Input image should be grayscale (single channel)")

    # Handle edge case when all pixels have the same value
    if np.max(image) == np.min(image):
        return np.zeros_like(image)

    # Apply min-max normalization
    normalized = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255.0
    return normalized.astype(np.uint8)


def normalize_image(original_img):
    """
    Convert image to grayscale and normalize it.

    Args:
        original_img (numpy.ndarray): Original BGR image

    Returns:
        tuple: (grayscale image, normalized grayscale image)
    """
    # Convert to grayscale
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Apply normalization
    normalized_img = minmax_normalize(gray_img)

    return gray_img, normalized_img


def extract_region_from_normalized(normalized_img, segmentation_img, region_type="wing"):
    """
    Extract a specific region from a normalized image using a segmentation mask.

    Args:
        normalized_img (numpy.ndarray): Normalized grayscale image
        segmentation_img (numpy.ndarray): Segmentation mask image
        region_type (str): Type of region to extract ("wing" or "wingtip")

    Returns:
        tuple: (region, mask) where region contains the pixels and mask is the binary mask
    """
    # Create mask based on region_type
    if region_type == "wing":
        # Create a mask for the wing region (assuming specific color in segmentation)
        lower = np.array([0, 0, 100])  # Adjust these values based on your segmentation color scheme
        upper = np.array([100, 100, 255])
        mask = cv2.inRange(segmentation_img, lower, upper)
    elif region_type == "wingtip":
        # Create a mask for the wingtip region
        lower = np.array([100, 0, 0])  # Adjust these values based on your segmentation color scheme
        upper = np.array([255, 100, 100])
        mask = cv2.inRange(segmentation_img, lower, upper)
    else:
        raise ValueError(f"Unsupported region type: {region_type}")

    # Apply mask to the normalized image
    result = cv2.bitwise_and(normalized_img, normalized_img, mask=mask)

    return result, mask