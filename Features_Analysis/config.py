###############################################################################
# CONFIGURATION
###############################################################################

# Paths to your data (adjust as needed)
SLATY_BACKED_IMG_DIR = r"./dataset/Original_Images/Slaty_Backed_Gull/"
SLATY_BACKED_SEG_DIR = r"./dataset/Colored_Images/Slaty_Backed_Gull/"

GLAUCOUS_WINGED_IMG_DIR = r"./dataset/Original_Images/Glaucous_Winged_Gull/"
GLAUCOUS_WINGED_SEG_DIR = r"./dataset/Colored_Images/Glaucous_Winged_Gull/"

# Number of images per species to process
S = 5

# Define the BGR colors for each region based on your RGB swatches:
REGION_COLORS = {
    "wingtip": (0, 255, 0),      # Green in RGB → (0, 255, 0) in BGR
    # "wing":    (0, 0, 255),      # Red in RGB → (0, 0, 255) in BGR
    # "head":    (255, 255, 0),    # Yellow in RGB → (0, 255, 255) in BGR
    # "body":    (0, 255, 255)   # Sky Blue (e.g., RGB (135,206,235)) → (235,206,135) in BGR
}