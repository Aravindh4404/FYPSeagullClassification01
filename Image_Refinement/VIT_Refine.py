import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Define your ViT architecture (example)
# --------------------------------------------------
class ViTModified(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

# --------------------------------------------------
# 2. Evaluation function
# --------------------------------------------------
def evaluate_model(
    model_path: str,
    model_class: nn.Module,
    data_dir: str,
    class_names: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./outputs_confusion_matrices"
):
    """
    Loads a model checkpoint and evaluates it on the dataset at data_dir.
    Generates and saves a confusion matrix.

    Args:
        model_path   : Path to the .pth file
        model_class  : The class of the model (e.g., ViTModified)
        data_dir     : Path to dataset with subfolders for each class
        class_names  : List of class names
        device       : 'cuda' or 'cpu'
        output_dir   : Directory to save confusion matrix plots and CSVs
    """

    # Create output directory if not exist
    os.makedirs(output_dir, exist_ok=True)

    # 2.1. Initialize and load model weights
    model = model_class(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2.2. Data transforms and loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # We can use torchvision.datasets.ImageFolder if each subfolder is a class
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 2.3. Lists to store predictions and labels
    all_preds = []
    all_labels = []

    # 2.4. Inference loop
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # 2.5. Build confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # cm shape => [num_classes, num_classes]
    # Rows = True labels, Cols = Predicted labels

    # 2.6. Plot confusion matrix with Seaborn
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                     xticklabels=class_names,
                     yticklabels=class_names)
    plt.title(f"Confusion Matrix\nModel: {os.path.basename(model_path)}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Save the confusion matrix plot
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    plot_path = os.path.join(output_dir, f"confmat_{model_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Confusion matrix saved to {plot_path}")

    # 2.7. Calculate per-class stats (correct / total) if desired
    # cm[i, i] = number of correct predictions for class i
    per_class_correct = cm.diagonal()
    per_class_total = cm.sum(axis=1)
    per_class_acc = per_class_correct / (per_class_total + 1e-8)

    # 2.8. Save metrics to a CSV
    metrics_df = pd.DataFrame({
        "class_name": class_names,
        "correct": per_class_correct,
        "total": per_class_total,
        "accuracy": per_class_acc
    })
    csv_path = os.path.join(output_dir, f"metrics_{model_name}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[INFO] Metrics CSV saved to {csv_path}")

    return cm

# --------------------------------------------------
# 3. Main script
# --------------------------------------------------
def main():
    # 3.1. Define your data directory and class names
    data_dir = r"D:\FYP\Black BG\Black Background"  # Update to your local path
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # 3.2. (Option A) Manually list your models
    #      Or (Option B) automatically discover them with glob
    # Example: Evaluate both 'best_model_vit_20241222.pth' and 'latest_model_vit_20241222.pth'
    # from a single folder. You can repeat for other folders if needed.

    # Option A: Hardcode model paths
    model_paths = [
        r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241222\best_model_vit_20241222.pth",
        r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241222\latest_model_vit_20241222.pth"
    ]

    # Option B: If you want to automatically gather all .pth files in a directory:
    # root_model_dir = r"D:\FYP\MODELS\VIT"
    # model_paths = glob.glob(os.path.join(root_model_dir, "**", "*.pth"), recursive=True)

    # 3.3. Evaluate each model
    for mp in model_paths:
        if not os.path.isfile(mp):
            print(f"[WARNING] Model file not found: {mp}")
            continue

        print(f"\n[INFO] Evaluating model: {mp}")
        _ = evaluate_model(
            model_path=mp,
            model_class=ViTModified,  # or InterpretableViT, etc. if needed
            data_dir=data_dir,
            class_names=class_names,
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_dir=r"D:\FYP\ConfusionMatrixOutputs"  # Folder to save confusion matrices
        )

if __name__ == "__main__":
    main()
import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Define your ViT architecture (example)
# --------------------------------------------------
class ViTModified(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

# --------------------------------------------------
# 2. Evaluation function
# --------------------------------------------------
def evaluate_model(
    model_path: str,
    model_class: nn.Module,
    data_dir: str,
    class_names: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./outputs_confusion_matrices"
):
    """
    Loads a model checkpoint and evaluates it on the dataset at data_dir.
    Generates and saves a confusion matrix.

    Args:
        model_path   : Path to the .pth file
        model_class  : The class of the model (e.g., ViTModified)
        data_dir     : Path to dataset with subfolders for each class
        class_names  : List of class names
        device       : 'cuda' or 'cpu'
        output_dir   : Directory to save confusion matrix plots and CSVs
    """

    # Create output directory if not exist
    os.makedirs(output_dir, exist_ok=True)

    # 2.1. Initialize and load model weights
    model = model_class(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2.2. Data transforms and loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # We can use torchvision.datasets.ImageFolder if each subfolder is a class
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 2.3. Lists to store predictions and labels
    all_preds = []
    all_labels = []

    # 2.4. Inference loop
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # 2.5. Build confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # cm shape => [num_classes, num_classes]
    # Rows = True labels, Cols = Predicted labels

    # 2.6. Plot confusion matrix with Seaborn
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                     xticklabels=class_names,
                     yticklabels=class_names)
    plt.title(f"Confusion Matrix\nModel: {os.path.basename(model_path)}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Save the confusion matrix plot
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    plot_path = os.path.join(output_dir, f"confmat_{model_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Confusion matrix saved to {plot_path}")

    # 2.7. Calculate per-class stats (correct / total) if desired
    # cm[i, i] = number of correct predictions for class i
    per_class_correct = cm.diagonal()
    per_class_total = cm.sum(axis=1)
    per_class_acc = per_class_correct / (per_class_total + 1e-8)

    # 2.8. Save metrics to a CSV
    metrics_df = pd.DataFrame({
        "class_name": class_names,
        "correct": per_class_correct,
        "total": per_class_total,
        "accuracy": per_class_acc
    })
    csv_path = os.path.join(output_dir, f"metrics_{model_name}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[INFO] Metrics CSV saved to {csv_path}")

    return cm

# --------------------------------------------------
# 3. Main script
# --------------------------------------------------
def main():
    # 3.1. Define your data directory and class names
    data_dir = r"D:\FYP\Black BG\Black Background"  # Update to your local path
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # 3.2. (Option A) Manually list your models
    #      Or (Option B) automatically discover them with glob
    # Example: Evaluate both 'best_model_vit_20241222.pth' and 'latest_model_vit_20241222.pth'
    # from a single folder. You can repeat for other folders if needed.

    # Option A: Hardcode model paths
    model_paths = [
        r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241222\best_model_vit_20241222.pth",
        r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241222\latest_model_vit_20241222.pth"
    ]

    # Option B: If you want to automatically gather all .pth files in a directory:
    # root_model_dir = r"D:\FYP\MODELS\VIT"
    # model_paths = glob.glob(os.path.join(root_model_dir, "**", "*.pth"), recursive=True)

    # 3.3. Evaluate each model
    for mp in model_paths:
        if not os.path.isfile(mp):
            print(f"[WARNING] Model file not found: {mp}")
            continue

        print(f"\n[INFO] Evaluating model: {mp}")
        _ = evaluate_model(
            model_path=mp,
            model_class=ViTModified,  # or InterpretableViT, etc. if needed
            data_dir=data_dir,
            class_names=class_names,
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_dir=r"D:\FYP\ConfusionMatrixOutputs"  # Folder to save confusion matrices
        )

if __name__ == "__main__":
    main()
import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Define your ViT architecture (example)
# --------------------------------------------------
class ViTModified(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

# --------------------------------------------------
# 2. Evaluation function
# --------------------------------------------------
def evaluate_model(
    model_path: str,
    model_class: nn.Module,
    data_dir: str,
    class_names: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./outputs_confusion_matrices"
):
    """
    Loads a model checkpoint and evaluates it on the dataset at data_dir.
    Generates and saves a confusion matrix.

    Args:
        model_path   : Path to the .pth file
        model_class  : The class of the model (e.g., ViTModified)
        data_dir     : Path to dataset with subfolders for each class
        class_names  : List of class names
        device       : 'cuda' or 'cpu'
        output_dir   : Directory to save confusion matrix plots and CSVs
    """

    # Create output directory if not exist
    os.makedirs(output_dir, exist_ok=True)

    # 2.1. Initialize and load model weights
    model = model_class(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2.2. Data transforms and loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # We can use torchvision.datasets.ImageFolder if each subfolder is a class
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 2.3. Lists to store predictions and labels
    all_preds = []
    all_labels = []

    # 2.4. Inference loop
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # 2.5. Build confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # cm shape => [num_classes, num_classes]
    # Rows = True labels, Cols = Predicted labels

    # 2.6. Plot confusion matrix with Seaborn
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                     xticklabels=class_names,
                     yticklabels=class_names)
    plt.title(f"Confusion Matrix\nModel: {os.path.basename(model_path)}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Save the confusion matrix plot
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    plot_path = os.path.join(output_dir, f"confmat_{model_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Confusion matrix saved to {plot_path}")

    # 2.7. Calculate per-class stats (correct / total) if desired
    # cm[i, i] = number of correct predictions for class i
    per_class_correct = cm.diagonal()
    per_class_total = cm.sum(axis=1)
    per_class_acc = per_class_correct / (per_class_total + 1e-8)

    # 2.8. Save metrics to a CSV
    metrics_df = pd.DataFrame({
        "class_name": class_names,
        "correct": per_class_correct,
        "total": per_class_total,
        "accuracy": per_class_acc
    })
    csv_path = os.path.join(output_dir, f"metrics_{model_name}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[INFO] Metrics CSV saved to {csv_path}")

    return cm

# --------------------------------------------------
# 3. Main script
# --------------------------------------------------

def main():
    # 3.1. Define your data directory and class names
    data_dir = r"D:\FYP\Black BG\Black Background"  # Update to your local path
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    model_paths = [
        r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241125\final_model_vit_20241125.pth",
        r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241222\latest_model_vit_20241222.pth",
        r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241224\final_model_vit_20241224.pth",
        r"D:\FYP\MODELS\VIT\VIT2_HQ2_20250207\final_model_vit_20250207.pth",
        r"D:\FYP\MODELS\VIT\VIT2_HQ3_20250208\final_model_vit_20250208.pth",
        r"D:\FYP\MODELS\VIT\VIT3_HQ2_20250206\latest_model.pth",
        r"D:\FYP\MODELS\VIT\ModelCheckpointsHQltst_ViT_20241114\latest_model_vit_20241114_epoch17.pth",
        r"D:\FYP\MODELS\VIT\ModelCheckpointsHQltst_ViT_20241112\best_model_vit_20241112.pth"

    ]
    # # Option B: Automatically gather all .pth files in a directory:
    # root_model_dir = r"D:\FYP\MODELS\VIT"
    # model_paths = list(set(glob.glob(os.path.join(root_model_dir, "**", "*.pth"), recursive=True)))
    # processed_models = set()
    # for mp in model_paths:
    #     if mp in processed_models:
    #         print(f"[INFO] Skipping already processed model: {mp}")
    #         continue
    #     processed_models.add(mp)
    #     # Evaluate model...
    # # Debug: print out model paths
    # print("Model paths found:")
    # for mp in model_paths:
    #     print(mp)

    # 3.3. Evaluate each model
    for mp in model_paths:
        if not os.path.isfile(mp):
            print(f"[WARNING] Model file not found: {mp}")
            continue

        print(f"\n[INFO] Evaluating model: {mp}")
        _ = evaluate_model(
            model_path=mp,
            model_class=ViTModified,  # or InterpretableViT, etc. if needed
            data_dir=data_dir,
            class_names=class_names,
            device = "cuda" if torch.cuda.is_available() else "cpu",
            output_dir=r"D:\FYP\ConfusionMatrixOutputs"  # Folder to save confusion matrices
        )



if __name__ == "__main__":
    main()
