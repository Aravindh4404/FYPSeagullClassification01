import os
import shutil
import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageFont


class VGGModifiedWrapper(torch.nn.Module):
    """Wrapper to access intermediate features for Grad-CAM"""

    def __init__(self, model):
        super().__init__()
        self.vgg = model.vgg.features
        self.classifier = model.vgg.classifier
        self.target_layer = self.vgg[-2]  # Last conv layer before pooling

    def forward(self, x):
        x = self.vgg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def generate_gradcam(model_path, data_dir, output_dir, device='cuda'):
    # Initialize model with Grad-CAM compatibility
    original_model = VGGModified()
    try:
        original_model.load_state_dict(torch.load(model_path))
    except RuntimeError:
        original_model.load_state_dict(torch.load(model_path, strict=False))

    model = VGGModifiedWrapper(original_model)
    model.eval().to(device)

    # Grad-CAM configuration
    target_layers = [model.target_layer]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    # Data pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    misclassified_dir = os.path.join(output_dir, "misclassified")
    os.makedirs(misclassified_dir, exist_ok=True)

    for idx, (img_tensor, true_label) in enumerate(dataset):
        input_tensor = img_tensor.unsqueeze(0).to(device)
        rgb_img = np.array(Image.open(dataset.imgs[idx][0]).convert('RGB').resize((224, 224))) / 255.0

        # Prediction with confidence
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred_prob, pred_label = torch.max(probs, 0)

        # Generate Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Add confidence text
        pil_img = Image.fromarray(visualization)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()
        text = f"{class_names[pred_label]}: {pred_prob:.2f}"
        draw.text((10, 10), text, fill=(255, 0, 0), font=font)

        # Save results
        filename = os.path.basename(dataset.imgs[idx][0])
        if pred_label != true_label:
            class_dir = os.path.join(misclassified_dir, f"True_{class_names[true_label]}",
                                     f"Pred_{class_names[pred_label]}")
            os.makedirs(class_dir, exist_ok=True)
            save_path = os.path.join(class_dir, filename)
        else:
            save_path = os.path.join(output_dir, filename)

        pil_img.save(save_path)


if __name__ == "__main__":
    MODEL_PATH = r"D:\FYP\MODELS\VGGModel\HQ3latst_20250210\best_model_vgg_20250210.pth"
    DATA_DIR = r"D:\ALLIMAGESLATEST"
    OUTPUT_DIR = r"D:\FYP\GradCAM_Results"

    generate_gradcam(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR
    )
