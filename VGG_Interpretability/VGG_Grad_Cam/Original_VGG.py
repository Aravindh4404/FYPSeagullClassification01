import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path

# 1) Define the VGG16Modified Class
class VGG16Modified(nn.Module):
    def __init__(self):
        super(VGG16Modified, self).__init__()
        from torchvision.models import VGG16_Weights
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)  # Binary classification
        )

    def forward(self, x):
        return self.vgg(x)

# 2) Load Model Checkpoint
# checkpoint_path = "D:/FYP/MODELS/VGGModel/HQ3_20250218/checkpoint_model_vgg_20250218.pth" #latest model trained which is using high epochs. may be overfitting most gradcam are ok
checkpoint_path = "D:/FYP/MODELS/VGGModel/HQ2ltst_20241123/best_model_vgg_20241123.pth" # one of the best performing old original models
 #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16Modified().to(device)
model.eval()
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# 3) Class Names and Transformations
class_names = ['Glaucous_Winged_Gull', 'Slaty_Backed_Gull']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4) Grad-CAM Computation
def generate_gradcam(model, image_tensor, target_layer):
    model.eval()
    features = []
    grads = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1)
    predicted_class_idx = torch.argmax(probs, dim=1).item()
    predicted_confidence = probs[0, predicted_class_idx].item()

    model.zero_grad()
    class_score = outputs[0, predicted_class_idx]
    class_score.backward()

    gradient = grads[0].detach().cpu().numpy()[0]
    feature_map = features[0].detach().cpu().numpy()[0]

    forward_handle.remove()
    backward_handle.remove()

    weights = np.mean(gradient, axis=(1, 2))
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= (cam.max() + 1e-9)

    return cam, predicted_class_idx, predicted_confidence

# 5) Image Preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device), np.array(image.resize((224, 224)))

# 6) Process Folder with Max Image Limit
def process_folder(dataset_path, class_names, max_images=15):
    dataset_path = Path(dataset_path)
    target_layer = model.vgg.features[-1]

    for class_idx, class_folder in enumerate(os.listdir(dataset_path)):
        class_path = dataset_path / class_folder
        if not class_path.is_dir():
            continue

        images_processed = 0
        for image_file in os.listdir(class_path):
            if images_processed >= max_images:
                break

            image_path = class_path / image_file
            image_tensor, original_image = preprocess_image(image_path)
            cam, predicted_class_idx, confidence = generate_gradcam(model, image_tensor, target_layer)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.clip(0.5 * original_image + 0.5 * heatmap, 0, 255).astype(np.uint8)

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
            pred_label = class_names[predicted_class_idx]
            plt.title(f"Overlay\nPredicted: {pred_label}, Conf: {confidence:.2%}")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            images_processed += 1

# 7) Main Execution
if __name__ == "__main__":
    dataset_path = "D:/FYP/Black BG/Black Background"
    process_folder(dataset_path, class_names, max_images=15)
