import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
from PIL import Image
import shutil


# Define a lightweight SE block for channel-wise attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        se = torch.mean(x, (2, 3))  # Global Average Pooling
        se = torch.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        se = se.view(batch, channels, 1, 1)
        return x * se


# Define an improved CNN architecture with SE Block
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBlock(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ------------------------------------
# Grad-CAM Implementation
# ------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        model_output = self.model(input_image)
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()

        self.model.zero_grad()
        model_output[0, target_class].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(pooled_gradients.shape[0]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10  # Added small epsilon to avoid division by zero
        return heatmap


# ------------------------------------
# Utility Functions
# ------------------------------------
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor


def overlay_heatmap(image_path, heatmap, output_path):
    image = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    # No text overlay - just save the image directly
    cv2.imwrite(output_path, superimposed_img)


# ------------------------------------
# Main Function
# ------------------------------------
def generate_gradcam_and_confusion_matrix(model_path, data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    print(f"Using device: {device}")
    print(f"Processing data from: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Initialize model
    model = ImprovedCNN().to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Check if this is a checkpoint dictionary with different key formats
    if "model_state_dict" in checkpoint:
        print("Loading model from checkpoint dictionary (model_state_dict)...")
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("Loading model from checkpoint dictionary (state_dict)...")
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback to direct loading if it's just the state dict
        print("Loading model directly...")
        model.load_state_dict(checkpoint)

    model.eval()

    # Setup Grad-CAM for ImprovedCNN
    # Use the last convolutional layer (before MaxPool2d)
    # This is the third SEBlock in the sequence (index would be 10)
    target_layer = model.conv_layers[10]  # This is the SEBlock after the third conv layer
    grad_cam = GradCAM(model, target_layer)

    # Data transformations (128x128 for ImprovedCNN)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # ImprovedCNN expects 128x128 input size
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader
    from torchvision import datasets
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Processing {len(dataset)} images from {len(dataset.classes)} classes.")
    print(f"Class mapping: {dataset.class_to_idx}")

    all_preds = []
    all_labels = []

    # Create output directories
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, "correct", class_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "misclassified", class_name), exist_ok=True)

    # Process images
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

        # Generate Grad-CAM
        heatmap = grad_cam.generate_cam(inputs)

        image_path = dataset.imgs[i][0]
        file_name = os.path.basename(image_path)
        true_class = class_names[labels.item()]
        predicted_class = class_names[preds.item()]
        is_correct = preds.item() == labels.item()

        if is_correct:
            # Save correctly classified
            class_dir = os.path.join(output_dir, "correct", true_class)
            output_path = os.path.join(class_dir, file_name)
            overlay_heatmap(image_path, heatmap, output_path)
        else:
            # Save misclassified
            misclassified_dir = os.path.join(output_dir, "misclassified", true_class)

            # Save Grad-CAM visualization
            output_path = os.path.join(misclassified_dir, file_name)
            overlay_heatmap(image_path, heatmap, output_path)

            # Save original copy
            original_copy_path = os.path.join(misclassified_dir, "original_" + file_name)
            shutil.copy2(image_path, original_copy_path)

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images.")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix - ImprovedCNN")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    cm_output_path = os.path.join(output_dir, "improved_cnn_confusion_matrix.png")
    plt.savefig(cm_output_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Calculate accuracy
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Save class-wise accuracy
    class_correct = [0, 0]
    class_total = [0, 0]

    for label, pred in zip(all_labels, all_preds):
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1

    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_accuracy = class_correct[i] / class_total[i] * 100
            print(f"Accuracy of {class_names[i]}: {class_accuracy:.2f}%")

    print(f"Confusion matrix saved to: {cm_output_path}")
    print(f"Results saved to: {output_dir}")


# ------------------------------------
# Main Execution
# ------------------------------------
if __name__ == "__main__":
    MODEL_PATH = r"D:\MODELS\CustomModel\HQ2_20241121_173047\final_model_20241121_173047.pth"
    DATA_DIR = r"D:\FYPSeagullClassification01\Test_Results\Test_Data"
    OUTPUT_DIR = r"D:\FYP\GradALL\best_model_20241121_173047"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_gradcam_and_confusion_matrix(MODEL_PATH, DATA_DIR, OUTPUT_DIR)