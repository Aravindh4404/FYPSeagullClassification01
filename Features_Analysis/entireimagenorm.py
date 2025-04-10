import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from skimage import exposure
import os
from glob import glob


def load_image(image_path):
    """Load an image and convert to RGB format."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def normalize_min_max(image):
    """Min-max normalization: scales values to [0, 1]."""
    norm_img = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # Process each channel
        channel = image[:, :, i].astype(np.float32)
        min_val = np.min(channel)
        max_val = np.max(channel)
        if max_val > min_val:
            norm_img[:, :, i] = (channel - min_val) / (max_val - min_val)
    return norm_img


def normalize_z_score(image):
    """Z-score normalization: (x - mean) / std."""
    norm_img = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # Process each channel
        channel = image[:, :, i].astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        if std > 0:
            norm_img[:, :, i] = (channel - mean) / std
    # Rescale to [0, 1] for display
    return (norm_img - np.min(norm_img)) / (np.max(norm_img) - np.min(norm_img))


def normalize_histogram_equalization(image):
    """Apply histogram equalization to each channel."""
    norm_img = np.zeros_like(image)
    for i in range(3):
        norm_img[:, :, i] = cv2.equalizeHist(image[:, :, i])
    return norm_img / 255.0


def normalize_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    norm_img = np.zeros_like(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(3):
        norm_img[:, :, i] = clahe.apply(image[:, :, i])
    return norm_img / 255.0


def normalize_adaptive_equalization(image):
    """Apply adaptive equalization from skimage."""
    img_float = image.astype(np.float32) / 255.0
    img_eq = exposure.equalize_adapthist(img_float, clip_limit=0.03)
    return img_eq


def compare_normalizations(image_path, output_dir=None):
    """Compare different normalization techniques on a single image."""
    # Load the image
    original_img = load_image(image_path)

    # Apply different normalization techniques
    min_max_norm = normalize_min_max(original_img)
    z_score_norm = normalize_z_score(original_img)
    hist_eq_norm = normalize_histogram_equalization(original_img)
    clahe_norm = normalize_clahe(original_img)
    adaptive_eq_norm = normalize_adaptive_equalization(original_img)

    # Create a figure with a grid layout for all normalizations
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Display all images with proper titles
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_img / 255.0)
    ax1.set_title('Original')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(min_max_norm)
    ax2.set_title('Min-Max Normalization')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(z_score_norm)
    ax3.set_title('Z-Score Normalization')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(hist_eq_norm)
    ax4.set_title('Histogram Equalization')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(clahe_norm)
    ax5.set_title('CLAHE')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(adaptive_eq_norm)
    ax6.set_title('Adaptive Equalization')
    ax6.axis('off')

    image_name = os.path.basename(image_path)
    plt.suptitle(f'Normalization Comparison - {image_name}', fontsize=16)
    plt.tight_layout()

    # Save the figure if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"norm_comparison_{image_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")

    plt.show()
    plt.close()


def process_directory(input_dir, output_dir=None):
    """Process all images in a directory."""
    # Find all image files in the directory
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(input_dir, ext)))

    print(f"Found {len(image_files)} image files in {input_dir}")

    # Process each image
    for img_path in image_files:
        try:
            print(f"Processing {img_path}...")
            compare_normalizations(img_path, output_dir)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    # Define dataset paths
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script

    # Datasets paths
    datasets = {
        "cub": os.path.join(base_dir, "datasets", "CUB_200_2011", "images"),
        "nabirds": os.path.join(base_dir, "datasets", "nabirds", "images"),
        "birdsnap": os.path.join(base_dir, "datasets", "birdsnap", "images"),
        "inat_birds": os.path.join(base_dir, "datasets", "inat_birds", "images"),
    }

    # Output directory for normalized comparison images
    output_base_dir = os.path.join(base_dir, "normalization_results")

    # Process example images from each dataset
    for dataset_name, dataset_path in datasets.items():
        if not os.path.exists(dataset_path):
            print(f"Dataset path not found: {dataset_path}")
            continue

        print(f"\nProcessing examples from {dataset_name} dataset...")

        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Get a list of image files in the dataset
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            # Look in dataset dir and one level of subdirectories
            image_files.extend(glob(os.path.join(dataset_path, ext)))
            image_files.extend(glob(os.path.join(dataset_path, "*", ext)))

        if not image_files:
            print(f"No images found in {dataset_path}")
            continue

        # Process a limited number of images from each dataset
        # (to avoid processing potentially thousands of images)
        max_images_per_dataset = 5  # Adjust as needed
        selected_images = image_files[:max_images_per_dataset]

        print(f"Processing {len(selected_images)} images from {dataset_name}")
        for img_path in selected_images:
            try:
                print(f"  Processing {os.path.basename(img_path)}...")
                compare_normalizations(img_path, dataset_output_dir)
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")

    print("\nDone processing all datasets!")

    # If you want to process a specific dataset or image instead:
    """
    # Process specific dataset
    dataset_name = "cub"  # Change to desired dataset name
    if dataset_name in datasets and os.path.exists(datasets[dataset_name]):
        dataset_output_dir = os.path.join(output_base_dir, dataset_name)
        process_directory(datasets[dataset_name], dataset_output_dir)
    else:
        print(f"Dataset {dataset_name} not found")

    # Or process a specific image
    specific_image = os.path.join(datasets["cub"], "example_category", "example_image.jpg")
    if os.path.exists(specific_image):
        dataset_output_dir = os.path.join(output_base_dir, "specific_examples")
        compare_normalizations(specific_image, dataset_output_dir)
    """