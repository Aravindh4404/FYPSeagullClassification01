import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


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


def generate_confusion_matrix(model_path, data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    model = VGGModified()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    except RuntimeError:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    wrong_paths = []
    wrong_true = []
    wrong_pred = []

    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store predictions and labels
            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)

            # Identify misclassified samples
            incorrect_mask = preds_cpu != labels_cpu
            for idx_in_batch in np.where(incorrect_mask)[0]:
                global_idx = total_samples + idx_in_batch
                img_path = dataset.imgs[global_idx][0]
                wrong_paths.append(img_path)
                wrong_true.append(labels_cpu[idx_in_batch])
                wrong_pred.append(preds_cpu[idx_in_batch])

            total_samples += inputs.size(0)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - VGGModified")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "vggmodified_confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Move misclassified images
    wrong_dir = os.path.join(output_dir, "misclassified")
    for img_path, true_idx, pred_idx in zip(wrong_paths, wrong_true, wrong_pred):
        true_class = class_names[true_idx]
        pred_class = class_names[pred_idx]

        dest_dir = os.path.join(wrong_dir, f"True_{true_class}", f"Pred_{pred_class}")
        os.makedirs(dest_dir, exist_ok=True)

        # Handle duplicate filenames
        filename = os.path.basename(img_path)
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            counter = 1
            while True:
                new_name = f"{base}_{counter}{ext}"
                new_dest = os.path.join(dest_dir, new_name)
                if not os.path.exists(new_dest):
                    dest_path = new_dest
                    break
                counter += 1

        shutil.move(img_path, dest_path)

    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Misclassified images moved to: {wrong_dir}")
    return cm_path, wrong_dir


if __name__ == "__main__":
    MODEL_PATH = r"D:\FYP\MODELS\VGGModel\HQ3latst_20250210\best_model_vgg_20250210.pth"
    DATA_DIR = r"D:\ALLIMAGESLATEST\HQ3FULL"
    OUTPUT_DIR = r"D:\FYP\ConfusionMatrix_Output\VGGModifiedfull"

    generate_confusion_matrix(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR
    )

# 659 127