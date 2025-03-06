import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms

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

# Visualize GradCAM for Folder Batch
def process_folder(model, root_folder, class_names, max_images=15):
    for class_idx, class_folder in enumerate(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        images_processed = 0
        for image_file in os.listdir(class_path):
            if images_processed >= max_images:
                break

            image_path = os.path.join(class_path, image_file)
            input_tensor, original_image = preprocess_image(image_path)
            cam, pred_class, confidence = compute_gradcam(model, input_tensor)
            if cam is None:
                continue

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

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
            pred_label = class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
            plt.title(f"Overlay\nPredicted: {pred_label}, Conf: {confidence:.2%}")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            images_processed += 1

if __name__ == "__main__":
    model = EnhancedViT()
    model.load_state_dict(torch.load(
        'D:/FYP/MODELS/VIT/VIT2_HQ3_20250208/final_model_vit_20250208.pth',
        map_location='cpu'))
    model.eval()

    class_names = ['Glaucous_Winged_Gull', 'Slaty_Backed_Gull']
    # root_folder = "D:/FYP/FYP DATASETS USED/Dataset HQ/HQ3/train"
    root_folder = "D:/FYP/Black BG/Black Background"
    process_folder(model, root_folder, class_names, max_images=30)