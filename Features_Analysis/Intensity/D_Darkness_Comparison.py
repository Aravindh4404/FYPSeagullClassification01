import matplotlib.pyplot as plt
from Features_Analysis.config import *

# Select an image to analyze
image_name = "005.png"
species = "Glaucous_Winged_Gull"

# Construct paths
image_path = f"../Dataset/Original_Images/{species}/{image_name}"
seg_path = f"../Dataset/Colored_Images/{species}/{image_name}"

# Load images
original_img = cv2.imread(image_path)
segmentation_img = cv2.imread(seg_path)

if original_img is None or segmentation_img is None:
    print(f"Error loading images: {image_path} or {seg_path}")
else:
    # Extract wing and wingtip regions
    wing_region, wing_mask = extract_region(original_img, segmentation_img, "wing")
    wingtip_region, wingtip_mask = extract_region(original_img, segmentation_img, "wingtip")

    # Convert to grayscale
    gray_wing = cv2.cvtColor(wing_region, cv2.COLOR_BGR2GRAY)

    # Get wing pixels (non-zero)
    wing_pixels = gray_wing[wing_mask > 0]

    if len(wing_pixels) == 0:
        print(f"No wing region found in {image_name}")
    else:
        # Calculate mean intensity of the wing
        mean_wing_intensity = np.mean(wing_pixels)

        # Get grayscale of original image for wingtip analysis
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # Get wingtip pixels
        wingtip_pixels = gray_img[wingtip_mask > 0]

        if len(wingtip_pixels) == 0:
            print(f"No wingtip region found in {image_name}")
        else:
            # Find wingtip pixels darker than the mean wing intensity
            darker_pixels = wingtip_pixels[wingtip_pixels < mean_wing_intensity]

            # Calculate percentage of darker pixels
            percentage_darker = (len(darker_pixels) / len(wingtip_pixels)) * 100

            # Print results
            print(f"Image: {image_name}")
            print(f"Species: {species}")
            print(f"Mean wing intensity: {mean_wing_intensity:.2f}")
            print(f"Wing pixel count: {len(wing_pixels)}")
            print(f"Wingtip pixel count: {len(wingtip_pixels)}")
            print(f"Darker wingtip pixels: {len(darker_pixels)}")
            print(f"Percentage of wingtip pixels darker than mean wing intensity: {percentage_darker:.2f}%")

            # Create a mask for darker pixels in the wingtip
            darker_mask = np.zeros_like(gray_img)
            darker_mask[np.logical_and(wingtip_mask > 0, gray_img < mean_wing_intensity)] = 255

            # Highlight the darker pixels in the original image
            highlighted_img = original_img.copy()
            highlighted_img[darker_mask > 0] = [255, 165, 0]  # Red in BGR

            # Display the visualization
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Darker Wingtip Pixels\n{percentage_darker:.2f}% of Wingtip")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"wingtip_darkness_{image_name.split('.')[0]}.png")
            plt.show()
