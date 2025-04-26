import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms
from sklearn.metrics import confusion_matrix, accuracy_score


# Function to extract model name from path
def get_model_name_from_path(model_path):
    base_name = os.path.basename(model_path)
    model_name = os.path.splitext(base_name)[0]  # Get filename without extension
    return model_name


# Enhanced ViT Model with Grad-CAM
class EnhancedViT(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512):
        super(EnhancedViT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

        self.embed_dim = self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else self.vit.head.in_features

        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        tokens = self.vit.forward_features(x)
        # Save gradients for Grad-CAM
        tokens.register_hook(self.save_tokens_grad)
        self.tokens = tokens

        attn_scores = self.attention_layer(tokens)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_feature = torch.sum(attn_weights * tokens, dim=1)
        out = self.classifier(weighted_feature)
        return out

    def save_tokens_grad(self, grad):
        self.tokens_grad = grad


# GradCAM Computation
def compute_gradcam(model, input_tensor, target_class=None):
    model.eval()
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    score = output[0, target_class]
    score.backward()

    # Remove the class token for Grad-CAM calculation
    tokens = model.tokens[:, 1:, :]
    tokens_grad = model.tokens_grad[:, 1:, :]

    batch_size, num_tokens, embed_dim = tokens.shape
    h = w = int(np.sqrt(num_tokens))

    tokens_reshaped = tokens.reshape(batch_size, h, w, embed_dim).permute(0, 3, 1, 2)
    tokens_grad_reshaped = tokens_grad.reshape(batch_size, h, w, embed_dim).permute(0, 3, 1, 2)

    weights = tokens_grad_reshaped.mean(dim=(2, 3), keepdim=True)
    cam = (weights * tokens_reshaped).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))

    confidence = torch.softmax(output, dim=1)[0, target_class].item()
    return cam, target_class, confidence


# Image Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image.resize((224, 224)))
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_image


# Function to save metrics to a text file
def save_metrics(output_dir, all_true, all_pred, class_names):
    overall_accuracy = accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred)
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


# Function to save misclassified images list
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
            f.write(
                f"Path: {img_path}, True: {true_class}, Predicted: {predicted_class}, Confidence: {confidence:.4f}\n")
    return list_file_path


# Function to plot and save the confusion matrix
def plot_and_save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Annotate each cell with the count
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Process folder and save Grad-CAM visualizations, metrics, and misclassified list
def process_folder(model, root_folder, class_names, base_output_folder, model_path, max_images=110):
    all_true = []
    all_pred = []
    misclassified_images = []

    # Extract model name and create specific output directory
    model_name = get_model_name_from_path(model_path)
    output_folder = os.path.join(base_output_folder, model_name)
    os.makedirs(output_folder, exist_ok=True)

    # Create directory for all Grad-CAM results
    gradcam_all_dir = os.path.join(output_folder, "gradcam_all")
    os.makedirs(gradcam_all_dir, exist_ok=True)

    # Create directories for correct/incorrect predictions
    correct_dir = os.path.join(output_folder, "correct")
    misclassified_dir = os.path.join(output_folder, "misclassified")
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(misclassified_dir, exist_ok=True)

    # Loop through each class folder in the root folder
    for class_idx, class_folder in enumerate(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Create corresponding folders
        output_class_folder = os.path.join(output_folder, class_folder)
        correct_class_dir = os.path.join(correct_dir, class_folder)
        misclassified_class_dir = os.path.join(misclassified_dir, class_folder)

        os.makedirs(output_class_folder, exist_ok=True)
        os.makedirs(correct_class_dir, exist_ok=True)
        os.makedirs(misclassified_class_dir, exist_ok=True)

        images_processed = 0
        for image_file in os.listdir(class_path):
            if images_processed >= max_images:
                break

            image_path = os.path.join(class_path, image_file)
            try:
                input_tensor, original_image = preprocess_image(image_path)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

            cam, pred_class, confidence = compute_gradcam(model, input_tensor)
            if cam is None:
                continue

            # Save ground truth and prediction for confusion matrix computation
            all_true.append(class_idx)
            all_pred.append(pred_class)
            is_correct = pred_class == class_idx

            # Create Grad-CAM overlay visualization
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

            # Plot the three-panel figure
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(cam, cmap='jet')
            plt.title("Grad-CAM Map")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
            true_label = class_names[class_idx]
            plt.title(f"Overlay\nTrue: {true_label}\nPredicted: {pred_label}, Conf: {confidence:.2%}")
            plt.axis("off")

            plt.tight_layout()

            # Save to gradcam_all directory
            all_save_path = os.path.join(gradcam_all_dir, f"{class_folder}_{os.path.splitext(image_file)[0]}.png")
            plt.savefig(all_save_path)

            # Save to class-specific directory
            class_save_path = os.path.join(output_class_folder, f"gradcam_{os.path.splitext(image_file)[0]}.png")
            plt.savefig(class_save_path)

            # Save to correct/incorrect directory
            if is_correct:
                correct_save_path = os.path.join(correct_class_dir, f"gradcam_{os.path.splitext(image_file)[0]}.png")
                plt.savefig(correct_save_path)
            else:
                # Track misclassified image
                misclassified_images.append({
                    "path": image_path,
                    "true_class": true_label,
                    "predicted_class": pred_label,
                    "confidence": confidence
                })

                misclass_save_path = os.path.join(misclassified_class_dir,
                                                  f"gradcam_{os.path.splitext(image_file)[0]}.png")
                plt.savefig(misclass_save_path)

            plt.close()
            images_processed += 1

    # Generate and save confusion matrix
    cm = confusion_matrix(all_true, all_pred)
    cm_save_path = os.path.join(output_folder, "metrics", "confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_save_path), exist_ok=True)
    plot_and_save_confusion_matrix(cm, class_names, cm_save_path)

    # Save metrics to text file
    metrics_file_path = save_metrics(output_folder, all_true, all_pred, class_names)

    # Save list of misclassified images
    misclassified_list_path = save_misclassified_list(output_folder, misclassified_images)

    print(f"Confusion matrix saved to: {cm_save_path}")
    print(f"Metrics saved to: {metrics_file_path}")
    print(f"Misclassified images list saved to: {misclassified_list_path}")
    print(f"Results saved to: {output_folder}")

    return all_true, all_pred


# Example Usage
if __name__ == "__main__":
    MODEL_PATH = r"D:/FYP/MODELS/VIT/VIT2_HQ3_20250208/final_model_vit_20250208.pth"
    DATA_DIR = r"D:\FYPSeagullClassification01\Test_Results\Test_Data"
    BASE_OUTPUT_DIR = r"D:\FYPSeagullClassification01\Test_Results\Test_Results\final_model_vit_20250208"

    model = EnhancedViT()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # Process the folder and collect true and predicted labels
    true_labels, pred_labels = process_folder(model, DATA_DIR, class_names, BASE_OUTPUT_DIR, MODEL_PATH, max_images=110)
