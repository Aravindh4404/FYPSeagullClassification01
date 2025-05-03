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
from sklearn.metrics import confusion_matrix

# InterpretableViT with Grad-CAM Hooks
class InterpretableViTGradCAM(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()
        self.embed_dim = self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else 768

        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)
        )

        self.tokens = None
        self.tokens_grad = None

    def forward(self, x):
        tokens = self.vit.forward_features(x)
        tokens.register_hook(self._save_tokens_grad)
        self.tokens = tokens

        cls_token = tokens[:, 0, :]
        patch_tokens = tokens[:, 1:, :]

        attn_scores = self.attention_layer(patch_tokens)
        attn_weights = torch.softmax(attn_scores, dim=1)

        weighted_patch = torch.sum(attn_weights * patch_tokens, dim=1)
        combined = torch.cat([cls_token, weighted_patch], dim=1)

        logits = self.classifier(combined)
        return logits, attn_weights

    def _save_tokens_grad(self, grad):
        self.tokens_grad = grad

# Grad-CAM Computation
def compute_gradcam(model, input_tensor, target_class=None):
    model.eval()
    output, _ = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    model.zero_grad()
    score = output[0, target_class]
    score.backward()

    tokens = model.tokens
    tokens_grad = model.tokens_grad

    if tokens is None or tokens_grad is None:
        print("No tokens or token gradients found. Make sure hooks are working.")
        return None, target_class, 0.0

    patch_tokens = tokens[:, 1:, :]
    patch_tokens_grad = tokens_grad[:, 1:, :]
    grads_mean = patch_tokens_grad.mean(dim=2, keepdim=True)
    cam = (patch_tokens * grads_mean).sum(dim=-1)
    cam = F.relu(cam)
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    cam_2d = cam.reshape(14, 14).detach().cpu().numpy()
    cam_2d = cv2.resize(cam_2d, (224, 224))
    confidence = torch.softmax(output, dim=1)[0, target_class].item()
    return cam_2d, target_class, confidence

# Image Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    original_image = np.array(img.resize((224, 224)))
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor, original_image

# Grad-CAM Visualization and Saving with Confusion Matrix Data Collection
def process_folder(model, root_folder, class_names, output_folder, max_images=15):
    all_true = []
    all_pred = []

    # Loop through each class folder in the root folder
    for class_idx, class_folder in enumerate(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Create corresponding folder in the output directory
        output_class_folder = os.path.join(output_folder, class_folder)
        os.makedirs(output_class_folder, exist_ok=True)

        images_processed = 0
        for image_file in os.listdir(class_path):
            if images_processed >= max_images:
                break

            image_path = os.path.join(class_path, image_file)
            input_tensor, original_image = preprocess_image(image_path)
            cam, pred_class, confidence = compute_gradcam(model, input_tensor)
            if cam is None:
                continue

            # Save ground truth and prediction for confusion matrix computation
            all_true.append(class_idx)
            all_pred.append(pred_class)

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
            plt.title(f"Overlay\nPredicted: {pred_label}, Conf: {confidence:.2%}")
            plt.axis("off")

            plt.tight_layout()

            # Save the figure instead of displaying it
            save_path = os.path.join(output_class_folder, f"gradcam_{os.path.splitext(image_file)[0]}.png")
            plt.savefig(save_path)
            plt.close()

            images_processed += 1

    return all_true, all_pred

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

# Example Usage
if __name__ == "__main__":
    model = InterpretableViTGradCAM(dropout_rate=0.3, hidden_dim=512)
    model.load_state_dict(torch.load("D:/MODELS/VIT/InterpretableViT_20250213/final_model_vit_20250213.pth", map_location="cpu"))
    model.eval()
    # InterpretableViT_20250213 / final_model_vit_20250213.pth
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]
    root_folder = r"D:\FYPSeagullClassification01\Test_Results\Test_Data"
    output_folder = r"D:\FYP\GradALL\final_model_vit_20250213"
    os.makedirs(output_folder, exist_ok=True)

    # Process the folder and collect true and predicted labels
    true_labels, pred_labels = process_folder(model, root_folder, class_names, output_folder, max_images=110)

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("Confusion Matrix:")
    print(cm)

    # Save the confusion matrix figure
    cm_save_path = os.path.join(output_folder, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, class_names, cm_save_path)
