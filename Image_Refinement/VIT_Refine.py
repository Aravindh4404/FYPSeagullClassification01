
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

# Enhanced ViT Model with Dropout
class EnhancedViT(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512):
        """
        Initializes the enhanced ViT model.
        - Loads a pre-trained ViT backbone.
        - Removes the original classification head.
        - Adds an attention mechanism to pool patch tokens.
        - Adds a custom MLP classifier head.
        """
        super(EnhancedViT, self).__init__()
        # Load a pre-trained ViT model from timm
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        # Remove the original classification head
        self.vit.head = nn.Identity()

        # Get the embedding dimension (most timm ViT models have 'embed_dim')
        if hasattr(self.vit, 'embed_dim'):
            self.embed_dim = self.vit.embed_dim
        else:
            # Fallback: use the in_features of the original head if available
            self.embed_dim = self.vit.head.in_features

        # Attention mechanism: compute an attention score for each token (patch)
        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 1)  # Outputs a scalar score per token
        )

        # Custom classifier head: LayerNorm -> Dropout -> Linear -> ReLU -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)  # Binary classification (2 classes)
        )

    def forward(self, x):
        """
        Forward pass:
          1. Extract patch token embeddings via ViT's forward_features.
          2. Compute attention scores for each token.
          3. Aggregate tokens via a weighted sum.
          4. Classify the aggregated feature vector.
        """
        # Get patch token embeddings; expected shape: [batch, num_tokens, embed_dim]
        tokens = self.vit.forward_features(x)
        # Compute attention scores for each token; shape: [batch, num_tokens, 1]
        attn_scores = self.attention_layer(tokens)
        # Normalize attention scores using softmax along the token dimension
        attn_weights = torch.softmax(attn_scores, dim=1)
        # Compute the weighted sum of token embeddings to form a global feature vector
        weighted_feature = torch.sum(attn_weights * tokens, dim=1)  # Shape: [batch, embed_dim]
        # Pass the aggregated features through the classifier head
        out = self.classifier(weighted_feature)
        return out

class InterpretableViT(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512):
        """
        This model uses a pre-trained ViT backbone and removes its original classification head.
        It then:
          - Extracts the [CLS] token (which is already well-trained)
          - Computes a learned attention over patch tokens (the remaining tokens)
          - Aggregates the patch tokens via a weighted sum
          - Concatenates the CLS token and the weighted patch summary
          - Feeds the combined representation through a custom MLP classifier
        The attention weights are returned for later visualization.
        """
        super(InterpretableViT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # Remove the original classification head

        # Determine embedding dimension (most ViT models have an attribute 'embed_dim')
        self.embed_dim = self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else 768

        # Attention layer on patch tokens (ignoring the CLS token)
        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

        # Classifier head: use both the CLS token and a weighted average of patch tokens.
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        # Get token embeddings from ViT; output shape: (B, N+1, embed_dim)
        tokens = self.vit.forward_features(x)
        # The first token is the [CLS] token
        cls_token = tokens[:, 0, :]  # Shape: (B, embed_dim)
        # Remaining tokens are patch tokens
        patch_tokens = tokens[:, 1:, :]  # Shape: (B, N, embed_dim)
        # Compute attention scores over patch tokens
        attn_scores = self.attention_layer(patch_tokens)  # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)    # (B, N, 1)
        # Weighted average of patch tokens
        weighted_patch = torch.sum(attn_weights * patch_tokens, dim=1)  # (B, embed_dim)
        # Combine the CLS token and weighted patch representation
        combined = torch.cat([cls_token, weighted_patch], dim=1)  # (B, 2*embed_dim)
        logits = self.classifier(combined)  # (B, 2)
        return logits, attn_weights  # Return logits and attention weights for interpretability

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
    # model = model_class(num_classes=len(class_names))
    model = model_class
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
            model_class=InterpretableViT(),  # or InterpretableViT, EnhancedViT etc. if needed
            data_dir=data_dir,
            class_names=class_names,
            device = "cuda" if torch.cuda.is_available() else "cpu",
            output_dir=r"D:\FYP\ConfusionMatrixOutputs"  # Folder to save confusion matrices
        )



if __name__ == "__main__":
    main()
