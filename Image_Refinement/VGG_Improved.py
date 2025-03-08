import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


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
        self.vgg = models.vgg16(pretrained=True)

        # Optionally freeze early layers
        if freeze_layers:
            for param in self.vgg.features[:24].parameters():
                param.requires_grad = False

        # Use the SEBlock attention module
        self.attention = SEBlock(channel=512, reduction=16)

        # Modify classifier to match the saved model architecture with BatchNorm
        # Updated dimensions to match the saved model (512, 256 instead of 4096, 1024)
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 512),  # Changed from 4096
            nn.BatchNorm1d(512),    # Changed from 4096
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),    # Changed from 1024
            nn.BatchNorm1d(256),    # Changed from 1024
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
# Function to check model architecture and choose appropriate loading strategy
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
# Evaluation function
# ------------------------------------
def evaluate_model(
        model_path,
        data_dir,
        class_names,
        output_dir="./outputs_confusion_matrices",
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Loads a model checkpoint and evaluates it on the dataset at data_dir.
    Generates and saves a confusion matrix.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model with architecture detection
    model = load_model_with_architecture_detection(model_path, device)
    model.to(device)
    model.eval()

    # Data transforms and loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    # Inference loop
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Build confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix with Seaborn
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                     xticklabels=class_names,
                     yticklabels=class_names)
    plt.title(f"Confusion Matrix\nModel: {os.path.basename(model_path)}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    plot_path = os.path.join(output_dir, f"confmat_{model_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Confusion matrix saved to {plot_path}")

    # Save per-class statistics
    per_class_correct = cm.diagonal()
    per_class_total = cm.sum(axis=1)
    per_class_acc = per_class_correct / (per_class_total + 1e-8)

    metrics_df = pd.DataFrame({
        "class_name": class_names,
        "correct": per_class_correct,
        "total": per_class_total,
        "accuracy": per_class_acc
    })
    csv_path = os.path.join(output_dir, f"metrics_{model_name}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[INFO] Metrics CSV saved to {csv_path}")

    return model, plot_path


# ------------------------------------
# Main script
# ------------------------------------
def main():
    # Define your data directory and class names
    data_dir = r"D:\FYP\Black BG\Black Background"
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # Specify model paths (original paths from your script)
    model_paths = [
        r"D:\FYP\MODELS\VGGModel\Improved_20250209\final_model_vgg_improved_20250209.pth",
        r"D:\FYP\MODELS\VGGModel\Improved_20250210\best_model_vgg_improved_20250210.pth",
    ]

    # Set output directory for confusion matrices
    output_dir = r"D:\FYP\ConfusionMatrixOutputs\VGGW"

    # Loop through model paths and evaluate each model
    for mp in model_paths:
        if not os.path.isfile(mp):
            print(f"[WARNING] Model file not found: {mp}")
            continue

        print(f"\n[INFO] Evaluating model: {mp}")
        model, cm_plot_path = evaluate_model(
            model_path=mp,
            data_dir=data_dir,
            class_names=class_names,
            output_dir=output_dir,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )


if __name__ == "__main__":
    main()