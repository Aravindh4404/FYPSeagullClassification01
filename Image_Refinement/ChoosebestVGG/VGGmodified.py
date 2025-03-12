import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ------------------------------------
# Custom VGGModified Architecture
# ------------------------------------
class VGGModified(nn.Module):
    def __init__(self):
        super(VGGModified, self).__init__()
        self.vgg = models.vgg16(pretrained=False)

        # Modified classifier architecture
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 1536),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.vgg(x)


# ------------------------------------
# Confusion Matrix Generation Only
# ------------------------------------
def generate_confusion_matrix(model_path, data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # Load model with correct architecture
    model = VGGModified()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    # Inference
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot and save
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f"Confusion Matrix\n{os.path.basename(model_path)}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")
    return output_path


# ------------------------------------
# Main Execution
# ------------------------------------
if __name__ == "__main__":
    # Configure these paths
    MODEL_PATH = r"D:\FYP\MODELS\VGGModel\HQ3latst_20250210\best_model_vgg_20250210.pth"
    DATA_DIR = r"D:\FYP\Black BG\Black Background"
    OUTPUT_DIR = r"D:\FYP\ConfusionMatrix_Output\VGGModified"

    # Generate confusion matrix
    cm_path = generate_confusion_matrix(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR
    )
