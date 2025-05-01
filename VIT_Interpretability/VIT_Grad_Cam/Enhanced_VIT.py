import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms
from sklearn.metrics import confusion_matrix


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


# Process images in folder and save Grad-CAM for correctly predicted images.
# Also record predictions for the confusion matrix.
def process_folder(model, root_folder, class_names, save_folder, max_images=15):
    y_true = []
    y_pred = []

    # Process each class folder (sorted to ensure consistency)
    for class_folder in sorted(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        # Determine ground-truth label index based on folder name
        if class_folder in class_names:
            true_class = class_names.index(class_folder)
        else:
            # Skip folders not matching known class names
            continue

        # Create output subfolder for current class under the results folder
        output_dir = os.path.join(save_folder, class_folder)
        os.makedirs(output_dir, exist_ok=True)

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

            # Record true and predicted labels
            y_true.append(true_class)
            y_pred.append(pred_class)

            # Save only if the image is correctly predicted
            if pred_class == true_class:
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

                # Save the overlay image with a new filename
                save_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + '_gradcam.png')
                Image.fromarray(overlay).save(save_path)

            images_processed += 1

    return y_true, y_pred


if __name__ == "__main__":
    # Initialize and load the model
    model = EnhancedViT()
    model_path = 'D:/FYP/MODELS/VIT/VIT2_HQ3_20250208/final_model_vit_20250208.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Define class names (these should correspond to the folder names)
    class_names = ['Glaucous_Winged_Gull', 'Slaty_Backed_Gull']
    # root_folder = "D:/FYP/FYP DATASETS USED/Dataset HQ/HQ3/train"
    root_folder = r"D:\FYPSeagullClassification01\Test_Results\Test_Data"
    output_folder = r"D:\FYP\GradALL\final_model_vit_20250208"

    # Create a results folder using the model name
    model_name = os.path.basename(os.path.dirname(model_path))
    results_folder = r"D:\FYP\GradALL\final_model_vit_20250208"
    os.makedirs(results_folder, exist_ok=True)

    # Process the folder, saving Grad-CAM overlays only for correctly predicted images,
    # and accumulate predictions for the confusion matrix.
    y_true, y_pred = process_folder(model, root_folder, class_names, results_folder, max_images=110)

    # Generate and save the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set(xticks=tick_marks, yticks=tick_marks, xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Save confusion matrix plot
    cm_save_path = os.path.join(results_folder, "confusion_matrix.png")
    plt.savefig(cm_save_path)
    plt.show()
