import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import cv2
from PIL import Image
import shutil  # Added for file copying


# Use Pre-trained VGG-16 model and modify it for binary classification
class VGG16Modified(nn.Module):
    def __init__(self):
        super(VGG16Modified, self).__init__()
        from torchvision.models import VGG16_Weights
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Replace the classifier with a custom binary classification layer
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 2)
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


def save_metrics(output_dir, all_labels, all_preds, class_names):
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate accuracy per class
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    # Create directory for metrics
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Save metrics to a text file
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
    output_path = os.path.join(gradcam_all_dir, file_name)
    overlay_heatmap(image_path, heatmap, output_path)
    
    # Add prediction info
    img = cv2.imread(output_path)
    confidence_text = f"Confidence: {confidence:.2f}"
    true_class_text = f"True: {true_class}"
    predicted_text = f"Predicted: {predicted_class}"
    
    cv2.putText(img, true_class_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, predicted_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, confidence_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, img)
    return output_path


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


# ------------------------------------
# Main Function
# ------------------------------------
def generate_gradcam_and_confusion_matrix(model_path, data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # Initialize and load model
    model = VGG16Modified().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    misclassified_images = []  # Track misclassified images

    # Create necessary directories
    os.makedirs(os.path.join(output_dir, "correct"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "misclassified"), exist_ok=True)

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
        confidence_value = confidence.item()

        # Save all Grad-CAM images regardless of correctness
        save_gradcam_all_images(
            output_dir, image_path, heatmap, file_name, 
            true_class, predicted_class, confidence_value
        )

        if is_correct:
            # Save correctly classified
            class_dir = os.path.join(output_dir, "correct", true_class)
            os.makedirs(class_dir, exist_ok=True)
            output_path = os.path.join(class_dir, file_name)

            overlay_heatmap(image_path, heatmap, output_path)

            # Add confidence score
            img = cv2.imread(output_path)
            confidence_text = f"Confidence: {confidence_value:.2f}"
            cv2.putText(img, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(output_path, img)
        else:
            # Track misclassified image info
            misclassified_images.append({
                "path": image_path,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": confidence_value
            })
            
            # Save misclassified
            misclassified_dir = os.path.join(output_dir, "misclassified", true_class)
            os.makedirs(misclassified_dir, exist_ok=True)

            # Save Grad-CAM visualization
            output_path = os.path.join(misclassified_dir, file_name)
            overlay_heatmap(image_path, heatmap, output_path)

            # Add prediction info
            img = cv2.imread(output_path)
            confidence_text = f"Confidence: {confidence_value:.2f}"
            predicted_text = f"Predicted: {predicted_class}"
            cv2.putText(img, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, predicted_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(output_path, img)

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

    cm_output_path = os.path.join(output_dir, "metrics", "vggmodified_confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_output_path), exist_ok=True)
    plt.savefig(cm_output_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Save metrics to text file
    metrics_file_path = save_metrics(output_dir, all_labels, all_preds, class_names)
    
    # Save list of misclassified images
    misclassified_list_path = save_misclassified_list(output_dir, misclassified_images)

    print(f"Confusion matrix saved to: {cm_output_path}")
    print(f"Metrics saved to: {metrics_file_path}")
    print(f"Misclassified images list saved to: {misclassified_list_path}")
    print(f"Results saved to: {output_dir}")


# ------------------------------------
# Main Execution
# ------------------------------------
if __name__ == "__main__":
    MODEL_PATH = r"D:\MODELS\VGGModel\HQ3latst_20250210\best_model_vgg_20250210.pth"
    DATA_DIR = r"D:\FYPSeagullClassification01\Test_Results\Test_Data"
    OUTPUT_DIR = r"D:\FYPSeagullClassification01\Test_Results\Test_Results\best_model_vgg_20250210"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_gradcam_and_confusion_matrix(MODEL_PATH, DATA_DIR, OUTPUT_DIR)
