import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms
import cv2
from sklearn.metrics import confusion_matrix

# InterpretableViT with properly configured hooks for attribution methods
class InterpretableViT(nn.Module):
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

        # Storage for activations and gradients
        self.tokens = None
        self.tokens_grad = None

    def forward(self, x):
        tokens = self.vit.forward_features(x)
        self.tokens = tokens
        tokens.register_hook(self._save_tokens_grad)

        cls_token = tokens[:, 0, :]
        patch_tokens = tokens[:, 1:, :]

        attn_scores = self.attention_layer(patch_tokens)
        attn_weights = torch.softmax(attn_scores, dim=1)

        weighted_patch = torch.sum(attn_weights * patch_tokens, dim=1)
        combined = torch.cat([cls_token, weighted_patch], dim=1)

        logits = self.classifier(combined)
        return logits, attn_weights

    def _save_tokens_grad(self, grad):
        """Hook to save the gradient of tokens"""
        self.tokens_grad = grad


# Integrated Gradients implementation
def compute_integrated_gradients(model, input_tensor, target_class=None, steps=50):
    """
    Computes Integrated Gradients attribution for the given input and target class.
    """
    model.eval()

    # Forward pass to get the prediction
    output, _ = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Get confidence
    prob = torch.nn.functional.softmax(output, dim=1)
    confidence = prob[0, target_class].item()

    # Create a baseline input (black image)
    baseline = torch.zeros_like(input_tensor)

    # Initialize accumulated gradients
    accumulated_grads = None

    # Compute integrated gradients
    for step in range(steps):
        alpha = step / (steps - 1)
        interpolated_input = baseline + alpha * (input_tensor - baseline)
        interpolated_input.requires_grad_(True)

        # Forward pass
        output, _ = model(interpolated_input)

        # Backward pass
        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # Access token gradients
        if model.tokens_grad is None:
            print(f"No gradients found at step {step}. Make sure hooks are working.")
            continue

        # Extract patch token gradients (excluding CLS token)
        patch_grads = model.tokens_grad[0, 1:, :].sum(dim=1)

        # Accumulate gradients
        if accumulated_grads is None:
            accumulated_grads = patch_grads.detach().clone()
        else:
            accumulated_grads += patch_grads.detach().clone()

    if accumulated_grads is None:
        print("No gradients accumulated. Make sure hooks are working.")
        return None, target_class, confidence

    accumulated_grads = accumulated_grads / steps

    # Reshape to patch grid (14x14 for ViT base)
    attribution_map = accumulated_grads.reshape(14, 14).detach().cpu().numpy()

    # Normalize for visualization
    attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-8)

    # Resize to image size
    attribution_map = cv2.resize(attribution_map, (224, 224))

    return attribution_map, target_class, confidence


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
    input_tensor.requires_grad_()  # Enable gradient tracking
    return input_tensor, original_image


# Process folder using only Integrated Gradients
def process_folder(model, root_folder, class_names, output_folder, max_images=15):
    """
    Process a folder of images using Integrated Gradients for attribution.
    """
    all_true = []
    all_pred = []

    os.makedirs(output_folder, exist_ok=True)

    # Loop through each class folder
    for class_idx, class_folder in enumerate(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        output_class_folder = os.path.join(output_folder, class_folder)
        os.makedirs(output_class_folder, exist_ok=True)

        images_processed = 0
        for image_file in os.listdir(class_path):
            if images_processed >= max_images:
                break

            image_path = os.path.join(class_path, image_file)
            input_tensor, original_image = preprocess_image(image_path)

            # Get prediction
            output, _ = model(input_tensor)
            pred_class = output.argmax(dim=1).item()
            all_true.append(class_idx)
            all_pred.append(pred_class)

            # Compute Integrated Gradients
            attribution, _, confidence = compute_integrated_gradients(model, input_tensor, pred_class)

            # Create overlay visualization
            heatmap = cv2.applyColorMap(np.uint8(255 * attribution), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

            # Plot original, attribution, and overlay
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(attribution, cmap='jet')
            plt.title("Integrated Gradients")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
            plt.title(f"Overlay\nPred: {pred_label}, Conf: {confidence:.2%}")
            plt.axis("off")

            plt.tight_layout()
            save_path = os.path.join(output_class_folder, f"interpretation_{os.path.splitext(image_file)[0]}.png")
            plt.savefig(save_path)
            plt.close()

            images_processed += 1

    return all_true, all_pred


# For testing a single image using only Integrated Gradients
def test_single_image(model, image_path):
    input_tensor, original_image = preprocess_image(image_path)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Forward pass to get prediction
    output, _ = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    print(f"Predicted class: {pred_class}, confidence: {confidence:.2%}")

    # Compute Integrated Gradients
    attribution, _, _ = compute_integrated_gradients(model, input_tensor, pred_class)

    plt.subplot(1, 3, 2)
    plt.imshow(attribution, cmap='jet')
    plt.title("Integrated Gradients")
    plt.axis("off")

    # Create overlay visualization
    heatmap = cv2.applyColorMap(np.uint8(255 * attribution), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    pred_label = "Glaucous_Winged_Gull" if pred_class == 0 else "Slaty_Backed_Gull"
    plt.title(f"Overlay\nPred: {pred_label}, Conf: {confidence:.2%}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = InterpretableViT(dropout_rate=0.3, hidden_dim=512)
    model.load_state_dict(
        torch.load("D:/FYP/MODELS/VIT/InterpretableViT_20250213/final_model_vit_20250213.pth", map_location=device))
    model.to(device)
    model.eval()

    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]
    root_folder = r"D:\FYP\Black BG\Black Background"
    output_folder = r"D:\FYP\Model_Comparison\VIT\final_model_vit_20250213"

    # Process folder and collect true and predicted labels using Integrated Gradients only
    true_labels, pred_labels = process_folder(model, root_folder, class_names, output_folder, max_images=110)

    # Compute and save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
    plt.close()

    # For testing a single image (uncomment the following line and provide a valid image path)
    # test_single_image(model, r"D:\FYP\Black BG\Black Background\YourImage.jpg")