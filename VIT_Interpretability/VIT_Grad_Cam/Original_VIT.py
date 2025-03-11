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
# 1. Define the Model & Helper Classes      #
#############################################
class ViTModified(nn.Module):
    def __init__(self):
        super(ViTModified, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Linear(num_ftrs, 2)  # 2-class output

    def forward(self, x):
        return self.vit(x)


class ViTAttentionRollout:
    """
    Collects attention maps from each Transformer block in a timm ViT model
    and computes an attention rollout map for visualization.
    """

    def __init__(self, model, discard_ratio=0.0):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attentions = []

        # Register hooks for each block in the ViT model
        for blk in self.model.vit.blocks:
            blk.attn.qkv.register_forward_hook(self._hook_qkv)

    def _hook_qkv(self, module, input, output):
        qkv = output  # shape: [B, tokens, 3 * C]
        B, N, C = qkv.shape
        num_heads = self.model.vit.blocks[0].attn.num_heads
        qkv = qkv.reshape(B, N, 3, num_heads, C // (3 * num_heads))
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, tokens, C_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (q.shape[-1]) ** -0.5
        q = q * scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # shape: [B, heads, tokens, tokens]
        self.attentions.append(attn.detach().cpu())

    def _compute_rollout(self, all_attentions):
        rollout = torch.eye(all_attentions[0].size(-1), device=all_attentions[0].device)
        for attn in all_attentions:
            attn_avg = attn.mean(dim=1)  # average over heads -> [B, tokens, tokens]
            if self.discard_ratio > 0:
                flat = attn_avg.view(attn_avg.size(0), -1)
                n = flat.size(1)
                vals, _ = flat.sort(dim=1)
                threshold_idx = int(n * self.discard_ratio)
                threshold = vals[:, threshold_idx].unsqueeze(1).expand(-1, n)
                mask = (flat >= threshold).float().reshape_as(attn_avg)
                attn_avg = attn_avg * mask
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(attn_avg, rollout)
        return rollout

    def get_attention_map(self):
        if len(self.attentions) == 0:
            raise RuntimeError("No attention collected. Did you do a forward pass?")
        rollout = self._compute_rollout(self.attentions)[0]  # assume batch size 1
        # Exclude the [CLS] token itself; compute attention from CLS to each patch
        cls_attention = rollout[0, 1:]  # shape: [196] for 14x14 patches
        return cls_attention

    def clear(self):
        self.attentions = []


#############################################
# 2. Preprocessing and Visualization        #
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


def save_attention_visualization(attn_map, original_image, class_name, confidence, output_path):
    # Prepare attention map: reshape and upsample to image size
    h = w = 14  # for 14x14 patches
    attn_map_2d = attn_map.reshape(h, w).numpy()
    attn_map_2d = (attn_map_2d - attn_map_2d.min()) / (attn_map_2d.max() - attn_map_2d.min() + 1e-8)
    attn_map_2d = cv2.resize(attn_map_2d, (224, 224))

    # Create heatmap and overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_2d), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    # Plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(attn_map_2d, cmap='jet')
    axs[1].set_title("Attention Rollout Map")
    axs[1].axis("off")

    axs[2].imshow(overlay)
    axs[2].set_title(f"Overlay\nPredicted: {class_name}, Conf: {confidence:.2%}")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


#############################################
# 3. Main Driver: Process Dataset           #
#############################################
if __name__ == "__main__":
    # ----- Configuration -----
    # Path to the trained model weights
    model_weights_path = r"D:\FYP\MODELS\VIT\VIT2_HQ2_20241224\final_model_vit_20241224.pth"
    # Model name (used for naming the output folder)
    model_name = "VIT2_HQ2_20241224"
    # Dataset directory: each subfolder is assumed to be a class (e.g., "Glaucous_Winged_Gull", "Slaty_Backed_Gull")
    dataset_dir = r"D:\FYP\Black BG\Black Background"
    # Base directory to store outputs (correct predictions and confusion matrix)
    output_base_dir = os.path.join("D:/FYP/OUTPUTS", model_name)
    os.makedirs(output_base_dir, exist_ok=True)

    # Define class names (adjust these if your folder names differ)
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    # ----- Load Model and Setup Attention Rollout -----
    model = ViTModified()
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    model.eval()
    attn_rollout = ViTAttentionRollout(model, discard_ratio=0.0)

    # ----- Loop over Dataset -----
    # Assume each subdirectory in dataset_dir corresponds to one class
    true_labels = []
    pred_labels = []

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
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

            # Clear previous attentions
            attn_rollout.clear()
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
                attn_map = attn_rollout.get_attention_map()
                # Save file with the same base name as the original image
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_class_dir, base_name)
                save_attention_visualization(attn_map, original_image, predicted_label, confidence, output_path)
                print(f"Saved overlay for correct prediction: {output_path}")

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
