import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms

############################################
# 1. InterpretableViT with Grad-CAM Hooks  #
############################################
class InterpretableViTGradCAM(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512):
        """
        This model:
          - Uses a pre-trained ViT backbone
          - Removes the original classification head
          - Learns an attention over patch tokens
          - Aggregates patch tokens
          - Concatenates CLS + aggregated patch tokens
          - Passes through an MLP classifier
          - Stores patch token embeddings for Grad-CAM
        """
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # Remove the original classification head

        # Determine embedding dimension
        self.embed_dim = self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else 768

        # Attention layer on patch tokens
        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

        # Classifier head (CLS + aggregated patch tokens => 2*embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)
        )

        # For Grad-CAM storage
        self.tokens = None       # Will hold patch embeddings (CLS + patch tokens)
        self.tokens_grad = None  # Will hold gradients of those embeddings

    def forward(self, x):
        # Extract token embeddings: shape [B, N+1, embed_dim]
        tokens = self.vit.forward_features(x)

        # Register hook to capture gradient wrt 'tokens'
        tokens.register_hook(self._save_tokens_grad)
        self.tokens = tokens  # store for later Grad-CAM

        # The first token is CLS
        cls_token = tokens[:, 0, :]    # shape [B, embed_dim]
        # Remaining tokens are patch tokens
        patch_tokens = tokens[:, 1:, :]  # shape [B, N, embed_dim]

        # Compute learned attention over patch tokens
        attn_scores = self.attention_layer(patch_tokens)  # [B, N, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, N, 1]

        # Weighted average of patch tokens
        weighted_patch = torch.sum(attn_weights * patch_tokens, dim=1)  # [B, embed_dim]

        # Concatenate CLS + weighted patch
        combined = torch.cat([cls_token, weighted_patch], dim=1)  # [B, 2*embed_dim]

        # Classifier
        logits = self.classifier(combined)  # [B, 2]

        return logits, attn_weights

    def _save_tokens_grad(self, grad):
        """This hook captures the gradient of the 'tokens' tensor."""
        self.tokens_grad = grad


############################################
# 2. Grad-CAM Computation
############################################
def compute_gradcam(model, input_tensor, target_class=None):
    """
    - Forward pass
    - Backprop on target_class
    - Compute patch-level Grad-CAM from 'model.tokens' and 'model.tokens_grad'
    """
    model.eval()

    # Forward pass
    output, _ = model(input_tensor)  # we don't need the attention weights here
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero gradients
    model.zero_grad()
    # Compute gradient for the chosen class
    score = output[0, target_class]
    score.backward()

    # Retrieve tokens and their gradients
    tokens = model.tokens  # shape: [B, N+1, embed_dim]
    tokens_grad = model.tokens_grad

    if tokens is None or tokens_grad is None:
        print("No tokens or token gradients found. Make sure hooks are working.")
        return None, target_class, 0.0

    # Exclude the CLS token (index=0) to focus on patch tokens
    patch_tokens = tokens[:, 1:, :]       # shape [B, 196, embed_dim] for 224×224, patch=16
    patch_tokens_grad = tokens_grad[:, 1:, :]  # same shape

    # We'll do standard Grad-CAM: globally average the gradients across channels,
    # then multiply by the patch token embeddings to get a patch-level importance.
    # shape [B, 196, embed_dim]
    grads_mean = patch_tokens_grad.mean(dim=2, keepdim=True)  # [B, 196, 1]
    # Weighted combination
    cam = (patch_tokens * grads_mean).sum(dim=-1)  # => [B, 196]

    # ReLU & normalize
    cam = F.relu(cam)
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    # shape => [1, 196], for a single image

    # Reshape to 14×14
    cam_2d = cam.reshape(14, 14).detach().cpu().numpy()

    # Upsample to 224×224
    cam_2d = cv2.resize(cam_2d, (224, 224))

    # Confidence
    confidence = torch.softmax(output, dim=1)[0, target_class].item()

    return cam_2d, target_class, confidence


############################################
# 3. Image Preprocessing
############################################
def preprocess_image(image_path):
    """
    Returns:
      input_tensor: shape [1, 3, 224, 224]
      original_image: a (224, 224, 3) NumPy array for plotting
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # If you trained with mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], keep it;
        # otherwise, use standard ImageNet normalization for timm:
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    original_image = np.array(img.resize((224, 224)))
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor, original_image


############################################
# 4. Grad-CAM Visualization
############################################
def visualize_gradcam(model, image_path, class_names):
    """
    - Preprocess image
    - Compute Grad-CAM
    - Overlay heatmap
    """
    # 1. Preprocess
    input_tensor, original_image = preprocess_image(image_path)

    # 2. Grad-CAM
    cam, pred_class, confidence = compute_gradcam(model, input_tensor)

    if cam is None:
        print("CAM was not computed.")
        return

    # 3. Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 4. Overlay heatmap on original image
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    # 5. Plot
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
    if pred_class < len(class_names):
        pred_label = class_names[pred_class]
    else:
        pred_label = f"Class {pred_class}"
    plt.title(f"Overlay\nPredicted: {pred_label}, Conf: {confidence:.2%}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


############################################
# 5. Example Usage
############################################
if __name__ == "__main__":
    # Example: Create model, load trained weights, run Grad-CAM on one image
    model = InterpretableViTGradCAM(dropout_rate=0.3, hidden_dim=512)
    # Load your existing checkpoint:
    model.load_state_dict(torch.load("D:/FYP/MODELS/VIT/InterpretableViT_20250213/final_model_vit_20250213.pth", map_location="cpu"))
    model.eval()

    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]
    image_path = "D:/FYP/FYP DATASETS USED/Dataset HQ/slaty only/3O4A1180.JPG"

    visualize_gradcam(model, image_path, class_names)
