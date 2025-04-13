import matplotlib.pyplot as plt
import numpy as np
import cv2

def normalize_image(image, mean, std):
    """
    Normalize the image using the given mean and standard deviation.
    Args:
        image (numpy.ndarray): Input image in RGB format.
        mean (list): Mean values for each channel (R, G, B).
        std (list): Standard deviation values for each channel (R, G, B).
    Returns:
        numpy.ndarray: Normalized image.
    """
    image = image / 255.0  # Scale pixel values to [0, 1]
    mean = np.array(mean)
    std = np.array(std)
    normalized = (image - mean) / std
    return normalized

def visualize_normalization(image_path, mean, std):
    """
    Visualize the normalization process for a single image.
    Args:
        image_path (str): Path to the input image.
        mean (list): Mean values for each channel (R, G, B).
        std (list): Standard deviation values for each channel (R, G, B).
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Normalize the image
    normalized_image = normalize_image(image, mean, std)

    # Plot the original and normalized images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Clip normalized image to [0, 1] for visualization
    normalized_image_clipped = np.clip(normalized_image, 0, 1)
    axes[1].imshow(normalized_image_clipped)
    axes[1].set_title("Normalized Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    image_path = "dataset/image.jpg"  # Replace with the actual image path in your dataset
    mean = [0.485, 0.456, 0.406]  # Example mean values (ImageNet)
    std = [0.229, 0.224, 0.225]   # Example std values (ImageNet)

    visualize_normalization(image_path, mean, std)

