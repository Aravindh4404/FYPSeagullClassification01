import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
from PIL import Image


class VGGModified(nn.Module):
    def __init__(self):
        super(VGGModified, self).__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(4096, 2)
            )
        )

    def forward(self, x):
        return self.vgg(x)


class ExpectedGradCAM:
    def __init__(self, model, target_layer, n_samples=50, sigma=0.15):
        self.model = model
        self.target_layer = target_layer
        self.n_samples = n_samples
        self.sigma = sigma

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    # In ExpectedGradCAM class
    def generate_cam(self, input_image, target_class=None):
        b, c, h, w = input_image.size()
        cam = torch.zeros((b, h, w), device=input_image.device)

        for noisy_image in self.generate_noisy_images(input_image):
            self.model.zero_grad()
            output = self.model(noisy_image)

            if target_class is None:
                target_class = output.argmax(dim=1)

            self.model.zero_grad()
            output[0, target_class].backward()

            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam += F.interpolate(torch.sum(weights * self.activations, dim=1, keepdim=True),
                                 size=(h, w), mode='bicubic', align_corners=False).squeeze(1)

        cam /= self.n_samples
        cam = F.relu(cam)

        # Enhanced normalization with contrast stretching
        if cam.max() > 0:
            cam = cam / cam.max()
            cam = torch.clamp(cam * 2.5, 0, 1)  # Increase contrast
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy()

    # In overlay_heatmap function
    def overlay_heatmap(image_path, heatmap, output_path):
        image = cv2.imread(image_path)

        # Convert heatmap to visible range
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        # Create mask for better visibility
        _, mask = cv2.threshold(heatmap, 15, 255, cv2.THRESH_BINARY)
        heatmap = cv2.bitwise_and(heatmap, mask)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)  # Adjust blending

        cv2.imwrite(output_path, superimposed_img)

    def generate_noisy_images(self, input_image):
        return [input_image + torch.randn_like(input_image) * self.sigma for _ in range(self.n_samples)]


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor


def overlay_heatmap(image_path, heatmap, output_path):
    image = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize heatmap to 0-255 range and convert to uint8
    heatmap = np.uint8(255 * heatmap)

    # Ensure heatmap is single-channel
    if len(heatmap.shape) > 2:
        heatmap = heatmap[:, :, 0]

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, superimposed_img)


def generate_gradcam_and_confusion_matrix(model_path, data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    model = VGGModified().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    target_layer = model.vgg.features[-1]
    expected_gradcam = ExpectedGradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

        heatmap = expected_gradcam.generate_cam(inputs)

        if preds.item() == labels.item():
            image_path = dataset.imgs[i][0]
            file_name = os.path.basename(image_path)
            class_name = class_names[preds.item()]
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            output_path = os.path.join(class_dir, file_name)
            overlay_heatmap(image_path, heatmap, output_path)

            img = cv2.imread(output_path)
            confidence_text = f"Confidence: {confidence.item():.2f}"
            cv2.putText(img, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(output_path, img)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix - VGGModified")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    cm_output_path = os.path.join(output_dir, "vggmodified_confusion_matrix.png")
    plt.savefig(cm_output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {cm_output_path}")
    print(f"Expected Grad-CAM visualizations for correctly predicted images saved to: {output_dir}")


if __name__ == "__main__":
    MODEL_PATH = r"D:\FYP\MODELS\VGGModel\HQ3latst_20250210\best_model_vgg_20250210.pth"
    DATA_DIR = r"D:\FYP\FYP DATASETS USED\test"
    OUTPUT_DIR = r"D:\FYP\GradCAM_Output\best_model_vgg_20250210exp"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_gradcam_and_confusion_matrix(MODEL_PATH, DATA_DIR, OUTPUT_DIR)
