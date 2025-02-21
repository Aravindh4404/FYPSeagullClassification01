import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import transforms

#############################################
# 1. A Standard ViT Model (Binary Classes)  #
#############################################
class ViTModified(nn.Module):
    def __init__(self):
        super(ViTModified, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        # Replace the head with a 2-class linear layer
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.vit(x)

###################################################
# 2. Attention Rollout Collector & Computation    #
###################################################
class ViTAttentionRollout:
    """
    Collects attention maps from each Transformer block in a timm ViT model
    and performs 'attention rollout' to generate a single 2D map that
    highlights how each patch influences the [CLS] token.
    """
    def __init__(self, model, discard_ratio=0.0):
        """
        Args:
            model (nn.Module): The timm ViT model
            discard_ratio (float): fraction of the lowest attention weights to discard
                                  in each layer (0.0 means keep all).
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self.attentions = []

        # Register hooks on each block's attention
        for blk in self.model.vit.blocks:
            # The timm ViT block has an 'attn' submodule with shape [B, heads, tokens, tokens].
            blk.attn.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        """
        This forward hook is called after each block's self-attention.
        The 'output' from timm's attention is a tuple: (attn_output, attn_weights),
        or sometimes just attn_output. In many timm versions, you can get
        the raw attention weights via output[1] or module.attn_drop.
        """
        # Some versions of timm return (attn_out, attn_weights), others do not.
        # We'll try to detect if the second element is attention weights:
        if isinstance(output, tuple) and len(output) == 2:
            # output[1] should be [B, num_heads, tokens, tokens]
            attn = output[1]
        else:
            # If you only get one item, or if your timm version differs,
            # you might need to access module.attn_weights or module.weights
            # (varies by timm version). Check `print(dir(module))` or timm docs.
            # If your timm version doesn't store it, you'll need a custom approach.
            raise RuntimeError(
                "Cannot retrieve attention weights from this timm version. "
                "Try printing `output` or hooking deeper in the code."
            )

        self.attentions.append(attn.detach().cpu())

    def _compute_rollout(self, all_attentions):
        """
        Attention rollout: multiply attention matrices across layers,
        removing the lowest fraction of attention weights if discard_ratio > 0.
        """
        # all_attentions: list of T tensors [B, heads, tokens, tokens]
        # We'll average heads in each layer -> [B, tokens, tokens]
        # Then combine them from first to last block to get final rollout map.
        rollout = torch.eye(all_attentions[0].size(-1))
        for attn in all_attentions:
            # attn shape: [B, heads, tokens, tokens]
            # Average across heads
            attn_avg = attn.mean(dim=1)  # [B, tokens, tokens]

            # Discard the lowest fraction of attention weights
            if self.discard_ratio > 0:
                # Flatten, find threshold, zero out below threshold
                flat = attn_avg.view(attn_avg.size(0), -1)
                n = flat.size(1)
                vals, _ = flat.sort(dim=1)
                threshold_idx = int(n * self.discard_ratio)
                threshold = vals[:, threshold_idx].unsqueeze(1).expand(-1, n)
                mask = (flat >= threshold).float().reshape_as(attn_avg)
                attn_avg = attn_avg * mask

            # Add identity to avoid losing local patch information
            # (some attention might skip patches entirely).
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)

            # Multiply into the rollout
            rollout = torch.matmul(attn_avg, rollout)

        # rollout shape: [B, tokens, tokens], we want just the last
        # but typically B=1 for a single image
        return rollout

    def get_attention_map(self):
        """
        After a forward pass, compute the attention rollout map.
        Returns a 2D array [tokens], describing each token's
        overall contribution to [CLS].
        """
        if len(self.attentions) == 0:
            raise RuntimeError("No attention collected. Did you do a forward pass?")
        # Stack up: shape => (#layers, B, heads, tokens, tokens)
        all_attentions = torch.stack(self.attentions, dim=0)
        # We assume B=1 for a single image. If B>1, you'd need to handle that.
        all_attentions = all_attentions[:, 0, :, :, :]  # keep only the first item in batch
        rollout = self._compute_rollout(all_attentions.unsqueeze(0))  # re-insert B=1
        # rollout shape => [1, tokens, tokens], final is [tokens, tokens]
        rollout = rollout[0]

        # The last row/column correspond to how each token contributes to [CLS].
        # We'll take the row corresponding to [CLS] token (index 0).
        # If you want a map for each patch => CLS, you can take rollout[:, 0] or rollout[0, :].
        # Usually, for "attention rollout," we look at the 'CLS' row (index=0) to see
        # how [CLS] is connected to each patch.
        cls_attention = rollout[0, 1:]  # skip the [CLS] token itself, index=1..N are patches
        return cls_attention  # shape: [196]

    def clear(self):
        """Clear stored attentions (useful if processing multiple images)."""
        self.attentions = []

##########################################
# 3. Preprocessing & Visualization Code  #
##########################################
def preprocess_image(image_path):
    # Standard ImageNet normalization recommended for timm ViT
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

def visualize_attention_map(attn_map, original_image, class_name, confidence):
    """
    attn_map: shape [196], representing attention for each patch (14x14).
    We'll reshape it to (14, 14), then upsample to 224x224, then overlay.
    """
    h = w = 14  # 14x14 = 196 patches
    attn_map_2d = attn_map.reshape(h, w).numpy()
    # Normalize
    attn_map_2d = (attn_map_2d - attn_map_2d.min()) / (attn_map_2d.max() - attn_map_2d.min() + 1e-8)
    # Resize to 224x224
    attn_map_2d = cv2.resize(attn_map_2d, (224, 224))

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_2d), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    # Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn_map_2d, cmap='jet')
    plt.title("Attention Rollout Map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay\nPredicted: {class_name}, Conf: {confidence:.2%}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

##########################
# 4. Main driver code    #
##########################
if __name__ == "__main__":
    # 1. Load your model
    model = ViTModified()
    model.load_state_dict(
        torch.load(
            "D:/FYP/MODELS/VIT/VIT2_HQ2_20241224/final_model_vit_20241224.pth",
            map_location="cpu"
        )
    )
    model.eval()

    # 2. Instantiate the Attention Rollout tool
    # discard_ratio=0.0 => keep all attention
    # you can try discard_ratio=0.9 to remove the lowest 90% of attention if you want sharper maps
    attn_rollout = ViTAttentionRollout(model, discard_ratio=0.0)

    # 3. Preprocess your image
    image_path = "D:/FYP/FYP DATASETS USED/Dataset HQ/slaty only/3O4A1180.JPG"
    input_tensor, original_image = preprocess_image(image_path)

    # 4. Make a forward pass to collect attention
    with torch.no_grad():
        output = model(input_tensor)
    # output shape: [1, 2]. Let's pick predicted class
    pred_class_idx = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, pred_class_idx].item()

    class_names = ["Glaucous_Winged_Gull", "Slaty_Backed_Gull"]
    predicted_label = class_names[pred_class_idx]

    # 5. Compute the final attention map
    # attn_rollout.attentions now has a list of [B, heads, tokens, tokens] for each block
    attn_map = attn_rollout.get_attention_map()  # shape [196]

    # 6. Visualize
    visualize_attention_map(attn_map, original_image, predicted_label, confidence)

    # 7. (Optional) Clear the stored attentions if you want to process more images
    attn_rollout.clear()
