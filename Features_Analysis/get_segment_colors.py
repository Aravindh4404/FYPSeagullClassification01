import cv2
import numpy as np

# Load a sample segmentation map (change path as needed)
seg_map = cv2.imread(r"C:\Users\Aravindh P\OneDrive - University of Nottingham Malaysia\FYP\FYPSeagullClassification01\Features_Analysis\Colored_Images\Glaucous_Winged_Gull\001.jpg")

# Get unique colors
unique_colors = np.unique(seg_map.reshape(-1, 3), axis=0)
print("Unique colors in segmentation map (BGR):", unique_colors)
