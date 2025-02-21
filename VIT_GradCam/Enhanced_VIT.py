import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms


# Define the Enhanced ViT model (with a small modification for GradCAM)
class EnhancedViT(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512):
        super(EnhancedViT, self).__init__()
        # Load a pre-trained ViT backbone and remove its classification head.
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

        if hasattr(self.vit, 'embed_dim'):
            self.embed_dim = self.vit.embed_dim
        else:
            self.embed_dim = self.vit.head.in_features

        # An attention mechanism to pool the patch tokens.
        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

        # Custom classifier head.
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)  # Binary classification (2 classes)
        )

    def forward(self, x):
        # Get patch token embeddings; shape: [B, num_tokens, embed_dim]
        tokens = self.vit.forward_features(x)
        # Register a hook on the tokens to capture gradients during backprop.
        tokens.register_hook(self.save_tokens_grad)
        self.tokens = tokens  # Save activations for later use in GradCAM.

        # Compute attention scores and aggregate tokens.
        attn_scores = self.attention_layer(tokens)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_feature = torch.sum(attn_weights * tokens, dim=1)
        out = self.classifier(weighted_feature)
        return out

    def save_tokens_grad(self, grad):
        self.tokens_grad = grad


# GradCAM computation function.
def compute_gradcam(model, input_tensor, target_class=None):
    model.eval()
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero all existing gradients.
    model.zero_grad()
    # Compute the score for the target class.
    score = output[0, target_class]
    score.backward()

    # Retrieve the patch token activations and gradients.
    # (Assuming the first token is the [CLS] token and should be ignored.)
    tokens = model.tokens[:, 1:, :]  # shape: [1, num_tokens-1, embed_dim]
    tokens_grad = model.tokens_grad[:, 1:, :]  # same shape

    # For a 224x224 image with patch size 16, there are 14x14 patches.
    batch_size, num_tokens, embed_dim = tokens.shape
    h = w = int(np.sqrt(num_tokens))  # Should be 14 for ViT-base.

    # Reshape tokens and gradients into a spatial map.
    tokens_reshaped = tokens.reshape(batch_size, h, w, embed_dim).permute(0, 3, 1, 2)
    tokens_grad_reshaped = tokens_grad.reshape(batch_size, h, w, embed_dim).permute(0, 3, 1, 2)

    # Compute channel-wise weights by global-average-pooling the gradients.
    weights = tokens_grad_reshaped.mean(dim=(2, 3), keepdim=True)  # shape: [1, embed_dim, 1, 1]

    # Combine weights with the activation maps.
    cam = (weights * tokens_reshaped).sum(dim=1, keepdim=True)  # shape: [1, 1, h, w]
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().detach().numpy()

    # Normalize the CAM to [0, 1] and resize to the input image size (224, 224).
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))

    # Get the confidence score for the predicted class.
    confidence = torch.softmax(output, dim=1)[0, target_class].item()

    return cam, target_class, confidence


# Preprocess a single image.
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


# Visualize the GradCAM result for one image.
def visualize_gradcam(model, image_path, class_names):
    input_tensor, original_image = preprocess_image(image_path)
    cam, pred_class, confidence = compute_gradcam(model, input_tensor)

    # Create a heatmap from the CAM.
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    # Plot the original image, CAM map, and overlay.
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("GradCAM Map")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay\nPredicted: {class_names[pred_class]}, Conf: {confidence:.2%}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Initialize the model.
    model = EnhancedViT()
    # Optionally, load your trained weights:
    model.load_state_dict(
        torch.load('D:/FYP/MODELS/VIT/VIT2_HQ3_20250208/final_model_vit_20250208.pth', map_location='cpu'))
    class_names = ['Glaucous_Winged_Gull', 'Slaty_Backed_Gull']

    # Set the path to a single test image.
    image_path = 'D:/FYP/FYP DATASETS USED/Dataset HQ/slaty only/3O4A1180.JPG'
    visualize_gradcam(model, image_path, class_names)
