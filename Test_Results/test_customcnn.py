import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import cv2
from PIL import Image
import shutil

# ------------------------------------
# Model Architecture Definition
# ------------------------------------
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
        heatmap /= np.max(heatmap)
        return heatmap

# ------------------------------------
# Utility Functions
# ------------------------------------
def sanitize_filename(filename):
    """Clean filename by removing/replacing problematic characters"""
    import unicodedata
    import re

    filename = unicodedata.normalize('NFKD', filename)
    filename = re.sub(r'[\u202f\u00a0\u2009\u200b\u2060\ufeff]', ' ', filename)
    filename = re.sub(r'[^\w\s\-_\.]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    filename = filename.strip('_')
    return filename

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def overlay_heatmap(image_path, heatmap, output_path):
    try:
        pil_image = Image.open(image_path).convert('RGB')
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return False

        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, superimposed_img)
        return True

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return False

def save_metrics(output_dir, all_labels, all_preds, class_names):
    overall_accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file_path = os.path.join(metrics_dir, "classification_metrics.txt")
    with open(metrics_file_path, "w") as f:
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nAccuracy per Class:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}: {class_accuracies[i]:.4f}\n")

    return metrics_file_path

def save_gradcam_all_images(output_dir, image_path, heatmap, file_name, true_class, predicted_class, confidence):
    gradcam_all_dir = os.path.join(output_dir, "gradcam_all")
    os.makedirs(gradcam_all_dir, exist_ok=True)

    clean_filename = sanitize_filename(file_name)
    output_path = os.path.join(gradcam_all_dir, clean_filename)

    success = overlay_heatmap(image_path, heatmap, output_path)
    if success:
        return output_path
    else:
        print(f"Failed to save Grad-CAM for {file_name}")
        return None

def save_misclassified_list(output_dir, misclassified_images):
    misclassified_dir = os.path.join(output_dir, "misclassified")
    os.makedirs(misclassified_dir, exist_ok=True)
    list_file_path = os.path.join(misclassified_dir, "misclassified_images.txt")
    with open(list_file_path, "w") as f:
        for img_info in misclassified_images:
            img_path = img_info["path"]
            true_class = img_info["true_class"]
            predicted_class = img_info["predicted_class"]
            confidence = img_info["confidence"]
            f.write(f"Path: {img_path}, True: {true_class}, Predicted: {predicted_class}, Confidence: {confidence:.4f}\n")
    return list_file_path

def safe_cv2_operations(image_path, output_path, operations_func):
    """Safely perform OpenCV operations on an image with proper error handling"""
    try:
        pil_image = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        if img is None:
            print(f"Warning: Could not load image {image_path}")
            return False

        img = operations_func(img)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return True

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return False

# ------------------------------------
# Main Function
# ------------------------------------
def generate_gradcam_and_confusion_matrix(model_path, data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # Initialize and load model
    model = ImprovedCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Setup Grad-CAM - target the last convolutional layer
    target_layer = model.conv_layers[-2]  # The last Conv2d layer before pooling
    grad_cam = GradCAM(model, target_layer)

    # Data transformations (128x128 to match the model's expected input)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []
    misclassified_images = []
    skipped_images = []

    # Create necessary directories
    os.makedirs(os.path.join(output_dir, "correct"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "misclassified"), exist_ok=True)

    # Process images
    for i, (inputs, labels) in enumerate(dataloader):
        try:
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
            clean_file_name = sanitize_filename(file_name)
            true_class = class_names[labels.item()]
            predicted_class = class_names[preds.item()]
            is_correct = preds.item() == labels.item()
            confidence_value = confidence.item()

            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                skipped_images.append(image_path)
                continue

            # Save all Grad-CAM images
            gradcam_output = save_gradcam_all_images(
                output_dir, image_path, heatmap, file_name,
                true_class, predicted_class, confidence_value
            )

            if gradcam_output is None:
                skipped_images.append(image_path)
                continue

            if is_correct:
                class_dir = os.path.join(output_dir, "correct", true_class)
                os.makedirs(class_dir, exist_ok=True)
                output_path = os.path.join(class_dir, clean_file_name)

                def add_confidence_text(img):
                    confidence_text = f"Confidence: {confidence_value:.2f}"
                    cv2.putText(img, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    return img

                if overlay_heatmap(image_path, heatmap, output_path):
                    safe_cv2_operations(output_path, output_path, add_confidence_text)
            else:
                misclassified_images.append({
                    "path": image_path,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "confidence": confidence_value
                })

                misclassified_dir = os.path.join(output_dir, "misclassified", true_class)
                os.makedirs(misclassified_dir, exist_ok=True)

                output_path = os.path.join(misclassified_dir, clean_file_name)
                if overlay_heatmap(image_path, heatmap, output_path):
                    def add_prediction_text(img):
                        confidence_text = f"Confidence: {confidence_value:.2f}"
                        predicted_text = f"Predicted: {predicted_class}"
                        cv2.putText(img, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(img, predicted_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        return img

                    safe_cv2_operations(output_path, output_path, add_prediction_text)

                    original_copy_path = os.path.join(misclassified_dir, "original_" + clean_file_name)
                    try:
                        shutil.copy2(image_path, original_copy_path)
                    except Exception as e:
                        print(f"Warning: Could not copy original image {image_path}: {str(e)}")
                else:
                    print(f"Failed to create Grad-CAM overlay for {image_path}")

        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
            skipped_images.append(dataset.imgs[i][0] if i < len(dataset.imgs) else f"Image {i}")
            continue

    # Print summary of skipped images
    if skipped_images:
        print(f"\nWarning: {len(skipped_images)} images were skipped due to errors:")
        for img_path in skipped_images:
            print(f"  - {img_path}")

    # Generate confusion matrix
    if all_labels and all_preds:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title("Confusion Matrix - ImprovedCNN")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        cm_output_path = os.path.join(output_dir, "metrics", "improved_cnn_confusion_matrix.png")
        os.makedirs(os.path.dirname(cm_output_path), exist_ok=True)
        plt.savefig(cm_output_path, bbox_inches='tight', dpi=300)
        plt.close()

        metrics_file_path = save_metrics(output_dir, all_labels, all_preds, class_names)
        misclassified_list_path = save_misclassified_list(output_dir, misclassified_images)

        print(f"Confusion matrix saved to: {cm_output_path}")
        print(f"Metrics saved to: {metrics_file_path}")
        print(f"Misclassified images list saved to: {misclassified_list_path}")
        print(f"Results saved to: {output_dir}")
    else:
        print("No images were successfully processed!")

# ------------------------------------
# Main Execution
# ------------------------------------
if __name__ == "__main__":
    MODEL_PATH = r"D:\MODELS\CustomModel\HQ2_20241121_173047\best_model_20241121_173047.pth"
    DATA_DIR = r"D:\FYPSeagullClassification01\Test_Results\Test_Data"
    OUTPUT_DIR = r"D:\FYPSeagullClassification01\Test_Results\Test_Results\best_model_20241121_173047"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_gradcam_and_confusion_matrix(MODEL_PATH, DATA_DIR, OUTPUT_DIR)