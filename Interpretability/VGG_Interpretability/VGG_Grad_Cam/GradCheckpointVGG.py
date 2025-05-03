import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
from PIL import Image
import shutil  # Added for file copying


class VGG16Modified(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(VGG16Modified, self).__init__()
        from torchvision.models import vgg16, VGG16_Weights

        # Load pre-trained VGG16 model
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Replace the classifier with a custom classification layer
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.vgg(x)


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
        heatmap /= np.max(heatmap)
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
    cv2.imwrite(output_path, superimposed_img)


# ------------------------------------
# Main Function
# ------------------------------------
def generate_gradcam_and_confusion_matrix(model_path, data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # Initialize model
    model = VGG16Modified().to(device)

    # Load checkpoint - FIXED: Properly extract model state dict from checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Check if this is a checkpoint dictionary with model_state_dict
    if "model_state_dict" in checkpoint:
        print("Loading model from checkpoint dictionary...")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback to direct loading if it's just the state dict
        print("Loading model directly...")
        model.load_state_dict(checkpoint)

    model.eval()

    # Setup Grad-CAM
    target_layer = model.vgg.features[-1]
    grad_cam = GradCAM(model, target_layer)

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []

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
            os.makedirs(class_dir, exist_ok=True)
            output_path = os.path.join(class_dir, file_name)

            overlay_heatmap(image_path, heatmap, output_path)

            # Remove confidence text
            # img = cv2.imread(output_path)
            # confidence_text = f"Confidence: {confidence.item():.2f}"
            # cv2.putText(img, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.imwrite(output_path, img)
        else:
            # Save misclassified
            misclassified_dir = os.path.join(output_dir, "misclassified", true_class)
            os.makedirs(misclassified_dir, exist_ok=True)

            # Save Grad-CAM visualization
            output_path = os.path.join(misclassified_dir, file_name)
            overlay_heatmap(image_path, heatmap, output_path)

            # Remove confidence text, optionally keep predicted class
            # img = cv2.imread(output_path)
            # confidence_text = f"Confidence: {confidence.item():.2f}"
            # predicted_text = f"Predicted: {predicted_class}"
            # cv2.putText(img, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(img, predicted_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.imwrite(output_path, img)

            # Save original copy
            original_copy_path = os.path.join(misclassified_dir, "original_" + file_name)
            shutil.copy2(image_path, original_copy_path)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix - VGGModified")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    cm_output_path = os.path.join(output_dir, "vggmodified_confusion_matrix.png")
    plt.savefig(cm_output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {cm_output_path}")
    print(f"Results saved to: {output_dir}")


# ------------------------------------
# Main Execution
# ------------------------------------
if __name__ == "__main__":
    MODEL_PATH = r"D:\MODELS\VGGModel\HQ3latst_20250210\best_model_vgg_20250210.pth"
    DATA_DIR = r"D:\FYP\FYP DATASETS USED\Dataset HQ\HQ3\train"
    OUTPUT_DIR = r"D:\FYP\GradALL\best_model_vgg_20250210"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_gradcam_and_confusion_matrix(MODEL_PATH, DATA_DIR, OUTPUT_DIR)