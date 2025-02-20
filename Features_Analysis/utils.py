import os
import cv2
import numpy as np
from scipy.stats import skew, kurtosis


###############################################################################
# HELPER FUNCTIONS
###############################################################################

def load_images_and_seg_maps(sb_img_dir, sb_seg_dir, gw_img_dir, gw_seg_dir, s):
    sb_images = sorted(os.listdir(sb_img_dir))[:s]
    sb_segs = sorted(os.listdir(sb_seg_dir))[:s]
    gw_images = sorted(os.listdir(gw_img_dir))[:s]
    gw_segs = sorted(os.listdir(gw_seg_dir))[:s]

    iSB, sSB, iGW, sGW = [], [], [], []

    for img_name, seg_name in zip(sb_images, sb_segs):
        img_path, seg_path = os.path.join(sb_img_dir, img_name), os.path.join(sb_seg_dir, seg_name)
        img, seg = cv2.imread(img_path), cv2.imread(seg_path)

        if seg is not None and seg.shape[-1] == 4:
            seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)

        if img is not None and seg is not None:
            iSB.append(img)
            sSB.append(seg)
        else:
            print(f"[Warning] Could not load: {img_name}, {seg_name}")

    for img_name, seg_name in zip(gw_images, gw_segs):
        img_path, seg_path = os.path.join(gw_img_dir, img_name), os.path.join(gw_seg_dir, seg_name)
        img, seg = cv2.imread(img_path), cv2.imread(seg_path)

        if seg is not None and seg.shape[-1] == 4:
            seg = cv2.cvtColor(seg, cv2.COLOR_BGRA2BGR)

        if img is not None and seg is not None:
            iGW.append(img)
            sGW.append(seg)
        else:
            print(f"[Warning] Could not load: {img_name}, {seg_name}")

    return iSB, sSB, iGW, sGW


def extract_region_pixels(image, seg_map, region_color, tolerance=10):
    if image is None or seg_map is None:
        return np.array([])

    lower = np.array([max(c - tolerance, 0) for c in region_color], dtype=np.uint8)
    upper = np.array([min(c + tolerance, 255) for c in region_color], dtype=np.uint8)
    mask = cv2.inRange(seg_map, lower, upper)

    selected_pixels = image[mask > 0]

    if selected_pixels.size == 0:
        print(f"[Info] No pixels found for region color {region_color} (Tolerance: {tolerance})")

    return selected_pixels, mask


def compute_statistics(pixel_values):
    if pixel_values.size == 0:
        return {'mean': np.nan, 'std': np.nan, 'median': np.nan, 'skew': np.nan, 'kurtosis': np.nan}

    if len(pixel_values.shape) == 2 and pixel_values.shape[1] == 3:
        pixel_values = pixel_values.mean(axis=1)

    return {
        'mean': np.mean(pixel_values),
        'std': np.std(pixel_values),
        'median': np.median(pixel_values),
        'skew': skew(pixel_values),
        'kurtosis': kurtosis(pixel_values)
    }