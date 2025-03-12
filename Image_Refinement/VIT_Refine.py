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
# 1. Define your ViT architectures
# --------------------------------------------------
class ViTModified(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)

# Enhanced ViT Model with Dropout (optional)
class EnhancedViT(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512, num_classes: int = 2):
        """
        Initializes the enhanced ViT model.
        """
        super(EnhancedViT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

        if hasattr(self.vit, 'embed_dim'):
            self.embed_dim = self.vit.embed_dim
        else:
            self.embed_dim = self.vit.head.in_features

        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        tokens = self.vit.forward_features(x)
        attn_scores = self.attention_layer(tokens)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_feature = torch.sum(attn_weights * tokens, dim=1)
        out = self.classifier(weighted_feature)
        return out

# Interpretable ViT Model – returns logits and attention weights
class InterpretableViT(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512, num_classes: int = 2):
        """
        Initializes the Interpretable ViT model.
        """
        super(InterpretableViT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # Remove the original classification head

        self.embed_dim = self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else 768

        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Get token embeddings from ViT; output shape: (B, N+1, embed_dim)
        tokens = self.vit.forward_features(x)
        cls_token = tokens[:, 0, :]  # First token is CLS
        patch_tokens = tokens[:, 1:, :]
        attn_scores = self.attention_layer(patch_tokens)  # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)    # (B, N, 1)
        weighted_patch = torch.sum(attn_weights * patch_tokens, dim=1)  # (B, embed_dim)
        combined = torch.cat([cls_token, weighted_patch], dim=1)  # (B, 2*embed_dim)
        logits = self.classifier(combined)
        return logits, attn_weights  # Return both for interpretability

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
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model (pass num_classes so it works with all variants)
    model = model_class(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Data transforms and loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []

    # Inference loop with extraction of logits if output is a tuple
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Compute predictions from logits
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Build confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix with Seaborn
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                     xticklabels=class_names,
                     yticklabels=class_names)
    plt.title(f"Confusion Matrix\nModel: {os.path.basename(model_path)}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    plot_path = os.path.join(output_dir, f"confmat_{model_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[INFO] Confusion matrix saved to {plot_path}")

    # Save per-class statistics
    per_class_correct = cm.diagonal()
    per_class_total = cm.sum(axis=1)
    per_class_acc = per_class_correct / (per_class_total + 1e-8)

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
    # Define your data directory and class names (update paths as needed)
    data_dir = r"D:\ALLIMAGESLATEST"
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # Specify the model file paths you wish to evaluate
    model_paths = [
         # r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241125\final_model_vit_20241125.pth",
        # r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241222\latest_model_vit_20241222.pth",
        # r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241224\final_model_vit_20241224.pth",
        # r"D:\FYP\MODELS\VIT\VIT2_HQ2_20250207\final_model_vit_20250207.pth", #adv
        # r"D:\FYP\MODELS\VIT\VIT2_HQ3_20250208\final_model_vit_20250208.pth", #adv
        # r"D:\FYP\MODELS\VIT\VIT3_HQ2_20250206\latest_model.pth",
        # r"D:\FYP\MODELS\VIT\ModelCheckpointsHQltst_ViT_20241114\latest_model_vit_20241114_epoch17.pth",
        # r"D:\FYP\MODELS\VIT\ModelCheckpointsHQltst_ViT_20241112\best_model_vit_20241112.pth"
        r"D:\FYP\MODELS\VIT\InterpretableViT_20250213\final_model_vit_20250213.pth",
    ]

    # Loop through model paths and evaluate each model.
    for mp in model_paths:
        if not os.path.isfile(mp):
            print(f"[WARNING] Model file not found: {mp}")
            continue

        print(f"\n[INFO] Evaluating model: {mp}")
        _ = evaluate_model(
            model_path=mp,
            model_class=EnhancedViT,  # Use InterpretableViT, EnhancedViT or another model class as required
            data_dir=data_dir,
            class_names=class_names,
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_dir=r"D:\FYP\ConfusionMatrixOutputs\ViTFULL"  # Folder to save confusion matrices
        )

if __name__ == "__main__":
    main()
