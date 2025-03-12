import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import shutil


# ------------------------------------
# Define the SEBlock (Squeeze-and-Excitation)
# ------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ------------------------------------
# Define the modified VGG16 model with SEBlock attention
# ------------------------------------
class VGG16Improved(nn.Module):
    def __init__(self, num_classes=2, freeze_layers=True):
        super(VGG16Improved, self).__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Optionally freeze early layers
        if freeze_layers:
            for param in self.vgg.features[:24].parameters():
                param.requires_grad = False

        # Use the SEBlock attention module
        self.attention = SEBlock(channel=512, reduction=16)

        # Pass a dummy input through features to determine the flattened size
        dummy = torch.zeros(1, 3, 224, 224)
        dummy = self.vgg.features(dummy)
        dummy = nn.AdaptiveAvgPool2d((7, 7))(dummy)
        flattened_size = dummy.view(1, -1).size(1)  # Typically 25088

        # Replace the classifier with a custom one using the computed flattened_size
        self.vgg.classifier = nn.Sequential(
            nn.Linear(flattened_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.attention(x)
        x = nn.AdaptiveAvgPool2d((7, 7))(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        return x


# ------------------------------------
# Model version with batch normalization for loading legacy models
# ------------------------------------
class VGG16ImprovedLegacy(nn.Module):
    def __init__(self, num_classes=2, freeze_layers=True):
        super(VGG16ImprovedLegacy, self).__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Optionally freeze early layers
        if freeze_layers:
            for param in self.vgg.features[:24].parameters():
                param.requires_grad = False

        # Use the SEBlock attention module
        self.attention = SEBlock(channel=512, reduction=16)

        # Modify classifier to match the saved model architecture with BatchNorm
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.attention(x)
        x = nn.AdaptiveAvgPool2d((7, 7))(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        return x


def load_model_with_architecture_detection(model_path, device):
    """
    Detects the model architecture from the state dict and loads accordingly.
    """
    # Load the state dict to check its structure
    state_dict = torch.load(model_path, map_location=device)

    # Print a few keys to help with debugging
    print(f"[DEBUG] Model path: {model_path}")
    print(f"[DEBUG] Sample keys in state dict: {list(state_dict.keys())[:5]}")

    # Check for BatchNorm layers in classifier
    has_batchnorm = any('vgg.classifier.1.running_mean' in key for key in state_dict.keys())

    if has_batchnorm:
        print("[INFO] Detected legacy model with BatchNorm layers")
        model = VGG16ImprovedLegacy()
    else:
        print("[INFO] Using standard VGG16Improved model")
        model = VGG16Improved()

    # Try loading with strict=True first, fall back to non-strict if it fails
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[INFO] Model loaded with strict=True")
    except RuntimeError as e:
        print(f"[WARNING] Strict loading failed: {e}")
        print("[INFO] Attempting to load with strict=False")
        model.load_state_dict(state_dict, strict=False)
        print("[INFO] Model loaded with strict=False")

    return model


# ------------------------------------
# Main script for saving correctly predicted images
# ------------------------------------
def save_correctly_predicted_images():
    # Define paths
    source_dir = r"D:\FYP\Black BG\Black Background"
    best_model_path = r"D:\FYP\MODELS\VGGModel\Improved_20250210\best_model_vgg_improved_20250210.pth"
    target_dir = r"D:\FYP\CorrectlyPredicted_Images"  # New directory for saving correctly predicted images

    # Create target directory structure
    os.makedirs(target_dir, exist_ok=True)

    # Define class names
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # Create target class directories
    for class_name in class_names:
        os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    model = load_model_with_architecture_detection(best_model_path, device)
    model.to(device)
    model.eval()

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Class to index mapping
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # Counters for statistics
    correct_count = {class_name: 0 for class_name in class_names}
    total_count = {class_name: 0 for class_name in class_names}

    # Process images from each class
    for class_name in class_names:
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)

        print(f"\n[INFO] Processing images from class: {class_name}")

        # Check if source directory exists
        if not os.path.exists(source_class_dir):
            print(f"[WARNING] Source directory does not exist: {source_class_dir}")
            continue

        # Get all image files
        image_files = [f for f in os.listdir(source_class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        for img_file in image_files:
            img_path = os.path.join(source_class_dir, img_file)
            total_count[class_name] += 1

            # Load and preprocess the image
            try:
                image = Image.open(img_path).convert('RGB')
                img_tensor = transform(image).unsqueeze(0).to(device)

                # Make prediction
                with torch.no_grad():
                    output = model(img_tensor)
                    _, predicted = torch.max(output, 1)

                predicted_class_idx = predicted.item()
                predicted_class = class_names[predicted_class_idx]
                true_class_idx = class_to_idx[class_name]

                # Check if prediction is correct
                if predicted_class_idx == true_class_idx:
                    correct_count[class_name] += 1
                    # Copy the image to the target directory
                    target_path = os.path.join(target_class_dir, img_file)
                    shutil.copy2(img_path, target_path)
                    print(f"[CORRECT] {img_file} -> {target_class_dir}")
                else:
                    print(f"[INCORRECT] {img_file} - Predicted as {predicted_class}, True: {class_name}")

            except Exception as e:
                print(f"[ERROR] Could not process {img_path}: {str(e)}")

    # Print summary of results
    print("\n" + "=" * 50)
    print("Summary of Correctly Predicted Images:")
    print("=" * 50)

    overall_correct = 0
    overall_total = 0

    for class_name in class_names:
        accuracy = correct_count[class_name] / total_count[class_name] if total_count[class_name] > 0 else 0
        print(f"{class_name}: {correct_count[class_name]}/{total_count[class_name]} correct ({accuracy:.2%})")
        print(f"Saved to: {os.path.join(target_dir, class_name)}")
        overall_correct += correct_count[class_name]
        overall_total += total_count[class_name]

    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    print("\nOverall: {}/{} correct ({:.2%})".format(
        overall_correct, overall_total, overall_accuracy
    ))
    print(f"All correctly predicted images saved to: {target_dir}")


if __name__ == "__main__":
    save_correctly_predicted_images()
