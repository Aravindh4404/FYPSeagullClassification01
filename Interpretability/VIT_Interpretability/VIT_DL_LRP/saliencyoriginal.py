import os
import glob
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#############################################
# 1. Define the Model                       #
#############################################
class ViTModified(nn.Module):
    def __init__(self):
        super(ViTModified, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.vit(x)


#############################################
# 2. Saliency Map Implementation            #
#############################################
class SaliencyMapGenerator:
    """
    Generates saliency maps for a given model by computing gradients
    of the output with respect to the input image.
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_saliency_map(self, input_tensor, target_class=None):
        # Enable gradients for the input
        input_tensor.requires_grad_()

        # Forward pass
        output = self.model(input_tensor)

        # If target class not specified, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1)

        # Zero all existing gradients
        self.model.zero_grad()

        # Backward pass to get gradients
        class_score = output[0, target_class]
        class_score.backward()

        # Get the gradients
        gradients = input_tensor.grad.detach().cpu()[0]

        # Take the absolute value and max across RGB channels
        saliency_map = torch.abs(gradients).max(dim=0)[0]

        # Normalize to [0, 1]
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

        return saliency_map.numpy()


#############################################
# 3. Preprocessing and Visualization        #
#############################################
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image.resize((224, 224)))
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_image


def save_saliency_visualization(saliency_map, original_image, class_name, confidence, output_path):
    # Resize saliency map to match image size
    saliency_map = cv2.resize(saliency_map, (224, 224))

    # Create heatmap and overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    # Plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(saliency_map, cmap='jet')
    axs[1].set_title("Saliency Map")
    axs[1].axis("off")

    axs[2].imshow(overlay)
    axs[2].set_title(f"Overlay\nPredicted: {class_name}, Conf: {confidence:.2%}")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


#############################################
# 4. Main Driver: Process Dataset           #
#############################################
if __name__ == "__main__":
    # ----- Configuration -----
    # Path to the trained model weights
    model_weights_path = r"D:\MODELS\VIT\VITModified_HQ3_20250503\final_model_vit_20250503.pth"
    # Model name (used for naming the output folder)
    model_name = "final_model_vit_20250503_saliency"
    # Dataset directory: each subfolder is assumed to be a class (e.g., "Glaucous_Winged_Gull", "Slaty_Backed_Gull")
    dataset_dir = r"D:\FYPSeagullClassification01\Test_Results\Test_Data"
    # Base directory to store outputs (correct predictions and confusion matrix)
    output_base_dir = os.path.join("D:\FYP\GradALL", model_name)
    os.makedirs(output_base_dir, exist_ok=True)

    # Define class names (adjust these if your folder names differ)
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # ----- Load Model and Setup Saliency Map Generator -----
    model = ViTModified()
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    model.eval()
    saliency_generator = SaliencyMapGenerator(model)

    # ----- Loop over Dataset -----
    # Assume each subdirectory in dataset_dir corresponds to one class
    true_labels = []
    pred_labels = []

    # Optional: Configure device (CPU/GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for class_folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, class_folder)
        if not os.path.isdir(folder_path):
            continue

        # Create output folder for this class
        output_class_dir = os.path.join(output_base_dir, class_folder)
        os.makedirs(output_class_dir, exist_ok=True)

        # Process image files (adjust extensions as needed)
        image_paths = glob.glob(os.path.join(folder_path, "*.*"))
        for image_path in image_paths:
            try:
                input_tensor, original_image = preprocess_image(image_path)
                input_tensor = input_tensor.to(device)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

            # Get prediction first
            with torch.no_grad():
                output = model(input_tensor)

            # Determine prediction and confidence
            pred_class_idx = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_class_idx].item()
            predicted_label = class_names[pred_class_idx]

            # Record true and predicted labels for confusion matrix
            true_labels.append(class_folder)
            pred_labels.append(predicted_label)

            # Save visualization only if prediction is correct
            if predicted_label == class_folder:
                # Generate saliency map for the predicted class
                saliency_map = saliency_generator.generate_saliency_map(input_tensor.cpu(), target_class=pred_class_idx)

                # Save file with the same base name as the original image
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_class_dir, f"saliency_{base_name}")
                save_saliency_visualization(saliency_map, original_image, predicted_label, confidence, output_path)
                print(f"Saved saliency map for correct prediction: {output_path}")

    # ----- Compute and Save Confusion Matrix -----
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    cm_output_path = os.path.join(output_base_dir, "confusion_matrix.png")
    plt.savefig(cm_output_path)
    plt.close()
    print(f"Confusion matrix saved at: {cm_output_path}")