import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms
from sklearn.metrics import confusion_matrix


# InterpretableViT model definition (keeping the class definition for loading the model)
class EnhancedViT(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()
        self.embed_dim = 768  # Fixed embed_dim

        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

        # Use 1536 as the input dimension for the classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(1536),  # Match the saved model dimension
            nn.Dropout(dropout_rate),
            nn.Linear(1536, hidden_dim),  # Match the saved model dimension
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        tokens = self.vit.forward_features(x)

        cls_token = tokens[:, 0, :]
        patch_tokens = tokens[:, 1:, :]

        attn_scores = self.attention_layer(patch_tokens)
        attn_weights = torch.softmax(attn_scores, dim=1)

        weighted_patch = torch.sum(attn_weights * patch_tokens, dim=1)
        # Concatenate to match the 1536 dimension
        combined = torch.cat([cls_token, weighted_patch], dim=1)

        logits = self.classifier(combined)
        return logits, attn_weights


# Image Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor


# Process folder for confusion matrix only
def process_folder_for_confusion_matrix(model, root_folder, device="cpu"):
    all_true = []
    all_pred = []

    for class_idx, class_folder in enumerate(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        print(f"Processing class: {class_folder} (class_idx={class_idx})")

        for image_file in os.listdir(class_path):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            image_path = os.path.join(class_path, image_file)
            try:
                input_tensor = preprocess_image(image_path)
                input_tensor = input_tensor.to(device)

                with torch.no_grad():
                    output, _ = model(input_tensor)
                    pred_class = output.argmax(dim=1).item()

                all_true.append(class_idx)
                all_pred.append(pred_class)

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

    return all_true, all_pred


# Function to plot and save the confusion matrix
def plot_and_save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
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
    print(f"Confusion matrix saved to {save_path}")


# Main execution
if __name__ == "__main__":
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = "D:/FYP/MODELS/VIT/EnhancedViT_20250213/final_model_vit_20250213.pth"
    model = (EnhancedViT())
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    # Define class names and paths
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]
    root_folder = r"D:\FYP\Black BG\Black Background"
    output_folder = r"D:\FYP\ConfusionMatrix_Output\ViT_20250213"
    os.makedirs(output_folder, exist_ok=True)

    # Process the folder and collect true and predicted labels
    print(f"Processing images in {root_folder}")
    true_labels, pred_labels = process_folder_for_confusion_matrix(model, root_folder, device)

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("Confusion Matrix:")
    print(cm)

    # Calculate accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Save the confusion matrix figure
    cm_save_path = os.path.join(output_folder, "confusion_matrix_vit.png")
    plot_and_save_confusion_matrix(cm, class_names, cm_save_path)
