import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import timm
import torch.nn as nn
from torchvision import transforms, datasets
import shutil


class ViTModified(nn.Module):
    def __init__(self):
        super(ViTModified, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 2)
        )  # Modify the classifier to output 2 classes

    def forward(self, x):
        return self.vit(x)


# ------------------------------------
# Attention Rollout Implementation - Fixed Version
# ------------------------------------
class AttentionRollout:
    def __init__(self, model):
        self.model = model
        self.vit = model.vit
        self.hooks = []
        self.attention_maps = []

        # Debug information
        print(f"ViT model structure: {type(self.vit)}")
        print(f"Number of blocks: {len(self.vit.blocks)}")

        # Register hooks on attention blocks to capture attention maps
        for block_idx, block in enumerate(self.vit.blocks):
            # Print block structure to help debug
            print(f"Block {block_idx} structure: {list(block.named_children())}")

            # Try different paths to access attention
            try:
                # Option 1: Direct access to attention weights
                hook = block.attn.register_forward_hook(
                    lambda m, i, o, block_id=block_idx: self.save_attention_map(o, block_id)
                )
                self.hooks.append(hook)
                print(f"Registered hook at block.attn for block {block_idx}")
            except AttributeError:
                try:
                    # Option 2: Use attention dropout as in original code but with safer access
                    if hasattr(block, 'attn') and hasattr(block.attn, 'attn_drop'):
                        hook = block.attn.attn_drop.register_forward_hook(
                            lambda m, i, o, block_id=block_idx: self.save_attention_map(o, block_id)
                        )
                        self.hooks.append(hook)
                        print(f"Registered hook at block.attn.attn_drop for block {block_idx}")
                    else:
                        print(f"Could not find appropriate hook point for block {block_idx}")
                except Exception as e:
                    print(f"Error registering hook for block {block_idx}: {e}")

    def save_attention_map(self, output, block_idx):
        # Save attention maps during forward pass
        print(f"Saved attention map for block {block_idx}, shape: {output.shape}")
        self.attention_maps.append(output.detach())

    def __call__(self, input_tensor, discard_ratio=0.9):
        self.attention_maps = []
        _ = self.model(input_tensor)  # Forward pass to collect attention maps

        # Check if attention maps were collected
        if not self.attention_maps:
            print("No attention maps were collected! Falling back to alternative method...")
            return self.fallback_attention(input_tensor)

        # Get number of attention heads and patches
        num_tokens = self.attention_maps[0].shape[1]
        batch_size = self.attention_maps[0].shape[0]

        print(f"Collected {len(self.attention_maps)} attention maps")
        print(f"First attention map shape: {self.attention_maps[0].shape}")

        # Consider only the attention maps from the CLS token to the image patches
        attention_maps = []
        for attn_map in self.attention_maps:
            # Average attention across all heads
            attn_map = attn_map.mean(dim=1)
            attention_maps.append(attn_map)

        # Recursively multiply the attention maps
        joint_attentions = torch.zeros((batch_size, num_tokens, num_tokens),
                                       device=attention_maps[0].device)
        joint_attentions = joint_attentions + torch.eye(num_tokens, device=attention_maps[0].device)

        for attn_map in attention_maps:
            # Add identity to handle residual connections
            residual_attn = torch.eye(num_tokens, device=attn_map.device)
            aug_attn_map = attn_map + residual_attn
            aug_attn_map = aug_attn_map / aug_attn_map.sum(dim=-1, keepdim=True)
            joint_attentions = torch.bmm(aug_attn_map, joint_attentions)

        # Get attention from CLS token to patches (exclude CLS token)
        rollout = joint_attentions[:, 0, 1:]

        # Apply discard ratio
        if discard_ratio > 0:
            flat_rollout = rollout.clone()
            flat_rollout = flat_rollout.reshape(batch_size, -1)
            for i in range(batch_size):
                sorted_indices = torch.argsort(flat_rollout[i])
                indices_to_discard = sorted_indices[:int(len(sorted_indices) * discard_ratio)]
                flat_rollout[i, indices_to_discard] = 0
            rollout = flat_rollout.reshape(batch_size, rollout.shape[1])

        return rollout

    def fallback_attention(self, input_tensor):
        """Fallback method that uses gradients for attention if hooks fail"""
        # Create a baseline attention map based on patch size
        batch_size = input_tensor.shape[0]
        patch_size = 16  # ViT patch size
        img_size = 224  # Input image size
        num_patches = (img_size // patch_size) ** 2

        # Create uniform attention as fallback
        print(f"Creating fallback uniform attention map with {num_patches} patches")
        uniform_attention = torch.ones((batch_size, num_patches), device=input_tensor.device)
        uniform_attention = uniform_attention / uniform_attention.sum(dim=1, keepdim=True)

        return uniform_attention


# ------------------------------------
# Attention Last Layer - Simpler Alternative
# ------------------------------------
class AttentionLastLayer:
    """A simpler implementation that uses only the attention map from the last layer"""

    def __init__(self, model):
        self.model = model
        self.vit = model.vit
        self.attention = None
        self.hooks = []

        # Try to find the last self-attention layer through different possible paths
        try:
            # Try to access the last block's attention
            last_block = self.vit.blocks[-1]
            print(f"Last block structure: {list(last_block.named_children())}")

            # Try different paths to the attention mechanism
            try:
                # Direct access to attention
                self.hooks.append(
                    last_block.attn.register_forward_hook(
                        lambda m, i, o: self.save_attention(o)
                    )
                )
                print("Registered hook at last_block.attn")
            except AttributeError:
                # Try accessing through attention dropout
                if hasattr(last_block.attn, 'attn_drop'):
                    self.hooks.append(
                        last_block.attn.attn_drop.register_forward_hook(
                            lambda m, i, o: self.save_attention(o)
                        )
                    )
                    print("Registered hook at last_block.attn.attn_drop")
                else:
                    print("Could not register attention hooks using standard paths")
        except Exception as e:
            print(f"Error setting up AttentionLastLayer hooks: {e}")

    def save_attention(self, output):
        self.attention = output.detach()
        print(f"Saved last layer attention with shape: {self.attention.shape}")

    def __call__(self, input_tensor):
        self.attention = None
        _ = self.model(input_tensor)  # Forward pass to capture attention

        # Extract attention map from CLS token to patches
        if self.attention is not None:
            attn_map = self.attention.mean(dim=1)  # Average across heads
            cls_attention = attn_map[:, 0, 1:]  # CLS token's attention to patches
            return cls_attention
        else:
            print("No attention captured in AttentionLastLayer! Using fallback.")
            # Create a fallback attention map
            batch_size = input_tensor.shape[0]
            patch_size = 16  # Default ViT patch size
            img_size = 224  # Default image size
            num_patches = (img_size // patch_size) ** 2

            # Create uniform attention as fallback
            uniform_attention = torch.ones((batch_size, num_patches), device=input_tensor.device)
            uniform_attention = uniform_attention / uniform_attention.sum(dim=1, keepdim=True)

            return uniform_attention


# ------------------------------------
# Gradient-based Attention - Another Alternative
# ------------------------------------
class GradientAttention:
    """Calculates attention using gradients with respect to the input"""

    def __init__(self, model):
        self.model = model

    def __call__(self, input_tensor):
        # Create a copy of the input that requires gradient
        input_with_grad = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        output = self.model(input_with_grad)

        # Get the predicted class
        _, pred = torch.max(output, 1)
        pred_class = pred.item()

        # Zero the gradients
        if input_with_grad.grad is not None:
            input_with_grad.grad.zero_()

        # Backpropagate the gradient for the predicted class
        output[0, pred_class].backward(retain_graph=True)

        # Get the gradient
        gradients = input_with_grad.grad.detach()

        # Calculate the gradient attention map
        # Take absolute value and sum across channels
        grad_attention = torch.abs(gradients).sum(dim=1)

        # Reshape to patch grid (assuming 16x16 patches for a 224x224 image)
        patch_size = 16
        img_size = 224
        batch_size = input_tensor.shape[0]
        num_patches = (img_size // patch_size) ** 2

        # Average pooling to get patch-level attention
        attention_patches = torch.nn.functional.avg_pool2d(
            grad_attention, kernel_size=patch_size, stride=patch_size
        )

        # Flatten the patch attention
        attention_patches = attention_patches.view(batch_size, -1)

        # Normalize
        attention_patches = attention_patches / (attention_patches.sum(dim=1, keepdim=True) + 1e-8)

        return attention_patches


# ------------------------------------
# Utility Functions
# ------------------------------------
def visualize_attention_rollout(image_path, rollout, output_path, img_size=224, patch_size=16):
    """Visualize attention rollout from ViT on original image"""
    # Read image and resize
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Resize for visualization
        image = cv2.resize(image, (img_size, img_size))

        # Reshape attention map to patch grid
        num_patches = (img_size // patch_size) ** 2

        # Verify rollout shape matches expected patch count
        if rollout.shape[0] != num_patches:
            print(f"Warning: Rollout shape {rollout.shape[0]} doesn't match expected patch count {num_patches}")
            # Resize rollout to match expected patch count if necessary
            if rollout.numel() > 0:  # Check if rollout has elements
                rollout = torch.nn.functional.interpolate(
                    rollout.unsqueeze(0).unsqueeze(0),
                    size=(num_patches,),
                    mode='linear'
                ).squeeze()
            else:
                # Create a uniform fallback
                rollout = torch.ones(num_patches, device=rollout.device)

        attention_map = rollout.reshape(img_size // patch_size, img_size // patch_size)

        # Upsample attention map to image size
        attention_map = attention_map.cpu().numpy()
        attention_map = cv2.resize(attention_map, (img_size, img_size))

        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # Colorize attention map
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay attention map on image
        result = heatmap * 0.4 + image * 0.6
        result = result.astype(np.uint8)

        # Save result
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result)
        print(f"Saved visualization to {output_path}")
    except Exception as e:
        print(f"Error in visualize_attention_rollout: {e}")


def generate_attention_rollout_and_confusion_matrix(model_path, data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]

    print(f"Using device: {device}")
    print(f"Processing data from: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Initialize model
    model = ViTModified().to(device)

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Check if this is a checkpoint dictionary with different key formats
        if "model_state_dict" in checkpoint:
            print("Loading model from checkpoint dictionary (model_state_dict)...")
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            print("Loading model from checkpoint dictionary (state_dict)...")
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Fallback to direct loading if it's just the state dict
            print("Loading model directly...")
            model.load_state_dict(checkpoint)

        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Setup multiple interpretability methods - try them all to see what works
    print("\n=== Setting up interpretability methods ===")
    print("1. Setting up Attention Rollout")
    attention_rollout = AttentionRollout(model)

    print("\n2. Setting up Attention Last Layer")
    attention_last_layer = AttentionLastLayer(model)

    print("\n3. Setting up Gradient Attention")
    gradient_attention = GradientAttention(model)

    # Data transformations for ViT (224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Dataset and DataLoader
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        print(f"Processing {len(dataset)} images from {len(dataset.classes)} classes.")
        print(f"Class mapping: {dataset.class_to_idx}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    all_preds = []
    all_labels = []

    # Create output directories for different methods
    method_names = ["attention_rollout", "attention_last_layer", "gradient_attention"]

    for method in method_names:
        for class_name in class_names:
            os.makedirs(os.path.join(output_dir, method, "correct", class_name), exist_ok=True)
            os.makedirs(os.path.join(output_dir, method, "misclassified", class_name), exist_ok=True)

    # Process images
    for i, (inputs, labels) in enumerate(dataloader):
        try:
            print(f"\nProcessing image {i + 1}/{len(dataset)}...")
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get model prediction
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            image_path = dataset.imgs[i][0]
            file_name = os.path.basename(image_path)
            true_class = class_names[labels.item()]
            predicted_class = class_names[preds.item()]
            is_correct = preds.item() == labels.item()

            # Process with each method
            print(f"  Image: {file_name}, True: {true_class}, Predicted: {predicted_class}")

            # 1. Attention Rollout
            print("  Generating attention rollout...")
            try:
                rollout = attention_rollout(inputs, discard_ratio=0.9)
                rollout = rollout[0]  # Get first batch item

                if is_correct:
                    class_dir = os.path.join(output_dir, "attention_rollout", "correct", true_class)
                    output_path = os.path.join(class_dir, file_name)
                    visualize_attention_rollout(image_path, rollout, output_path)
                else:
                    misclassified_dir = os.path.join(output_dir, "attention_rollout", "misclassified", true_class)
                    output_path = os.path.join(misclassified_dir, file_name)
                    visualize_attention_rollout(image_path, rollout, output_path)

                    # Save original copy
                    original_copy_path = os.path.join(misclassified_dir, "original_" + file_name)
                    shutil.copy2(image_path, original_copy_path)
            except Exception as e:
                print(f"  Error in attention rollout: {e}")

            # 2. Attention Last Layer
            print("  Generating attention from last layer...")
            try:
                last_attention = attention_last_layer(inputs)

                if is_correct:
                    class_dir = os.path.join(output_dir, "attention_last_layer", "correct", true_class)
                    output_path = os.path.join(class_dir, file_name)
                    visualize_attention_rollout(image_path, last_attention, output_path)
                else:
                    misclassified_dir = os.path.join(output_dir, "attention_last_layer", "misclassified", true_class)
                    output_path = os.path.join(misclassified_dir, file_name)
                    visualize_attention_rollout(image_path, last_attention, output_path)
            except Exception as e:
                print(f"  Error in last layer attention: {e}")

            # 3. Gradient Attention
            print("  Generating gradient-based attention...")
            try:
                grad_attention = gradient_attention(inputs)

                if is_correct:
                    class_dir = os.path.join(output_dir, "gradient_attention", "correct", true_class)
                    output_path = os.path.join(class_dir, file_name)
                    visualize_attention_rollout(image_path, grad_attention, output_path)
                else:
                    misclassified_dir = os.path.join(output_dir, "gradient_attention", "misclassified", true_class)
                    output_path = os.path.join(misclassified_dir, file_name)
                    visualize_attention_rollout(image_path, grad_attention, output_path)
            except Exception as e:
                print(f"  Error in gradient attention: {e}")

        except Exception as e:
            print(f"Error processing image {i}: {e}")

    # Generate confusion matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title("Confusion Matrix - Vision Transformer")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        cm_output_path = os.path.join(output_dir, "vit_confusion_matrix.png")
        plt.savefig(cm_output_path, bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")

    # Calculate accuracy
    try:
        accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Save class-wise accuracy
        class_correct = [0, 0]
        class_total = [0, 0]

        for label, pred in zip(all_labels, all_preds):
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

        for i in range(len(class_names)):
            if class_total[i] > 0:
                class_accuracy = class_correct[i] / class_total[i] * 100
                print(f"Accuracy of {class_names[i]}: {class_accuracy:.2f}%")
    except Exception as e:
        print(f"Error calculating accuracy: {e}")

    print(f"Confusion matrix saved to: {cm_output_path}")
    print(f"Results saved to: {output_dir}")


# ------------------------------------
# Main Execution
# ------------------------------------
if __name__ == "__main__":
    MODEL_PATH = r"D:\MODELS\VIT\VITModified_HQ3_20250503\final_model_vit_20250503.pth"
    DATA_DIR = r"D:\FYPSeagullClassification01\Test_Results\Test_Data"
    OUTPUT_DIR = r"D:\FYP\GradALL\final_model_vit_20250503AR"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_attention_rollout_and_confusion_matrix(MODEL_PATH, DATA_DIR, OUTPUT_DIR)