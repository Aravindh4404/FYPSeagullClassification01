import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# Use Pre-trained VGG-16 model and modify it for binary classification
class VGG16Modified(nn.Module):
    def __init__(self):
        super(VGG16Modified, self).__init__()
        from torchvision.models import VGG16_Weights
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Replace the classifier with a custom binary classification layer
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.vgg(x)

########################################
# 4. Evaluation function
########################################
def evaluate_model(
    model_path: str,
    model_class: nn.Module,
    data_dir: str,
    class_names: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./outputs_confusion_matrices_vgg"
):
    """
    Loads a model checkpoint and evaluates it on the dataset at data_dir.
    Generates and saves a confusion matrix and CSV metrics.
    Returns the loaded model and the confusion matrix plot path.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Initialize and load model weights
    model = model_class()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Data transforms and DataLoader using ImageNet normalization.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
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

    # Calculate per-class stats and save as CSV
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

    return model, plot_path

########################################
# 5. Decide which model class to use
########################################
def get_model_class(model_path: str):

        return VGG16Modified

########################################
# 6. Display a sample prediction and confusion matrix
########################################
def display_results(model, cm_plot_path, data_dir, class_names, device):
    """
    Displays a random sample prediction along with the corresponding confusion matrix image.
    """
    # Display sample prediction
    # Load dataset without transforms for original image display
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))
    import random
    sample_idx = random.randint(0, len(dataset) - 1)
    img_tensor, true_label = dataset[sample_idx]
    image_path, _ = dataset.imgs[sample_idx]
    original_img = Image.open(image_path).convert("RGB")
    input_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    pred_class = class_names[pred.item()]
    true_class = class_names[true_label]

    # Display sample prediction image
    plt.figure(figsize=(6,6))
    plt.imshow(original_img)
    plt.title(f"Sample Prediction\nTrue: {true_class}\nPredicted: {pred_class} (Confidence: {confidence.item()*100:.2f}%)")
    plt.axis("off")
    plt.show()

    # Display the confusion matrix image
    cm_img = Image.open(cm_plot_path)
    plt.figure(figsize=(6,5))
    plt.imshow(cm_img)
    plt.title(f"Confusion Matrix\n{os.path.basename(cm_plot_path)}")
    plt.axis("off")
    plt.show()

########################################
# 7. Main script
########################################
def main():
    # Define your dataset directory and class names
    data_dir = r"D:\FYP\White BG"
    # data_dir = r"D:\FYP\Black BG\Black Background"
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]
    # List of all model checkpoints (example)
    model_paths = [
        r"D:\FYP\MODELS\VGGModel\HQ2ltst_20241209\best_model_vgg_20241209.pth",
        r"D:\FYP\MODELS\VGGModel\HQ2ltst_20241210\best_model_vgg_20241210.pth",
        r"D:\FYP\MODELS\VGGModel\HQ2ltst_20241123\best_model_vgg_20241123.pth",
        r"D:\FYP\MODELS\VGGModel\HQ2ltst_20241214\best_model_vgg_20241214.pth",
        r"D:\FYP\MODELS\VGGModel\HQ2ltst_20241218\final_model_vgg_20241218.pth",
        r"D:\FYP\MODELS\VGGModel\HQ3_20250218\checkpoint_model_vgg_20250218.pth",
        r"D:\FYP\MODELS\VGGModel\HQ3latst_20250210\best_model_vgg_20250210.pth",
        r"D:\FYP\MODELS\VGGModel\HQ3latst_20250216\checkpoint_model_vgg_20250216.pth",
        r"D:\FYP\MODELS\VGGModel\HQ2ltst_20241210\final_model_vgg_20241210.pth",
        r"D:\FYP\MODELS\VGGModel\HQ3_20250218\checkpoint_model_vgg_20250218.pth",
        r"D:\FYP\MODELS\VGGModel\HQ3latst_20250307\final_model_vgg_20250307.pth"
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Process each model checkpoint
    for mp in model_paths:
        if not os.path.isfile(mp):
            print(f"[WARNING] Model file not found: {mp}")
            continue

        print(f"\n[INFO] Evaluating model: {mp}")
        chosen_model_class = get_model_class(mp)
        model, cm_plot_path = evaluate_model(
            model_path=mp,
            model_class=chosen_model_class,
            data_dir=data_dir,
            class_names=class_names,
            device=device,
            output_dir=r"D:\FYP\ConfusionMatrixOutputs\VGGW"
        )
        # Display sample prediction and confusion matrix for this model
        display_results(model, cm_plot_path, data_dir, class_names, device)

if __name__ == "__main__":
    main()
