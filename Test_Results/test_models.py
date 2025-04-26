import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import timm
from torchvision import models, transforms, datasets
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader
import json

# Create directories for results
os.makedirs('test_results/interpretability', exist_ok=True)
os.makedirs('test_results/wrong_predictions', exist_ok=True)

# Define SE Block for CNN
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        se = torch.mean(x, (2, 3))
        se = torch.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        se = se.view(batch, channels, 1, 1)
        return x * se

# Define all model architectures
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBlock(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class EnhancedViT(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512):
        super(EnhancedViT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()
        self.embed_dim = self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else self.vit.head.in_features

        self.attention_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        tokens = self.vit.forward_features(x)
        # Save gradients for Grad-CAM
        tokens.register_hook(self.save_tokens_grad)
        self.tokens = tokens

        attn_scores = self.attention_layer(tokens)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_feature = torch.sum(attn_weights * tokens, dim=1)
        out = self.classifier(weighted_feature)
        return out

    def save_tokens_grad(self, grad):
        self.tokens_grad = grad

class InterpretableViT(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_dim=512):
        super(InterpretableViT, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()
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
            nn.Linear(hidden_dim, 2)
        )

        self.tokens = None
        self.tokens_grad = None

    def forward(self, x):
        tokens = self.vit.forward_features(x)
        tokens.register_hook(self._save_tokens_grad)
        self.tokens = tokens

        cls_token = tokens[:, 0, :]
        patch_tokens = tokens[:, 1:, :]

        attn_scores = self.attention_layer(patch_tokens)
        attn_weights = torch.softmax(attn_scores, dim=1)

        weighted_patch = torch.sum(attn_weights * patch_tokens, dim=1)
        combined = torch.cat([cls_token, weighted_patch], dim=1)

        logits = self.classifier(combined)
        return logits, attn_weights

    def _save_tokens_grad(self, grad):
        self.tokens_grad = grad

class InceptionV3Modified(nn.Module):
    def __init__(self):
        super(InceptionV3Modified, self).__init__()
        from torchvision.models import Inception_V3_Weights
        self.inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        if self.training:
            aux, x = self.inception(x)
            return aux, x
        else:
            return self.inception(x)

class ResNet50Modified(nn.Module):
    def __init__(self):
        super(ResNet50Modified, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.resnet(x)

class VGG16Modified(nn.Module):
    def __init__(self):
        super(VGG16Modified, self).__init__()
        from torchvision.models import VGG16_Weights
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.vgg(x)

class VGG16Custom(nn.Module):
    def __init__(self):
        super(VGG16Custom, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.vgg(x)

class ViTModified(nn.Module):
    def __init__(self):
        super(ViTModified, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.vit(x)

# Grad-CAM implementation for CNN models
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        model_output = self.model(input_tensor)
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        self.model.zero_grad()
        model_output[0, target_class].backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(pooled_gradients.shape[0]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

# Attention Rollout for ViT models
class AttentionRollout:
    def __init__(self, model, discard_ratio=0.0):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attentions = []
        
        for blk in self.model.vit.blocks:
            blk.attn.qkv.register_forward_hook(self._hook_qkv)
    
    def _hook_qkv(self, module, input, output):
        qkv = output
        B, N, C = qkv.shape
        num_heads = self.model.vit.blocks[0].attn.num_heads
        qkv = qkv.reshape(B, N, 3, num_heads, C // (3 * num_heads))
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (q.shape[-1]) ** -0.5
        q = q * scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        self.attentions.append(attn.detach().cpu())
    
    def get_attention_map(self):
        if len(self.attentions) == 0:
            raise RuntimeError("No attention collected. Did you do a forward pass?")
        
        rollout = torch.eye(self.attentions[0].size(-1))
        for attn in self.attentions:
            attn_avg = attn.mean(dim=1)
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
        
        cls_attention = rollout[0, 1:]
        return cls_attention

def compute_gradcam_vit(model, input_tensor, target_class=None):
    model.eval()
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    model.zero_grad()
    score = output[0, target_class]
    score.backward()
    
    tokens = model.tokens[:, 1:, :]
    tokens_grad = model.tokens_grad[:, 1:, :]
    
    batch_size, num_tokens, embed_dim = tokens.shape
    h = w = int(np.sqrt(num_tokens))
    
    tokens_reshaped = tokens.reshape(batch_size, h, w, embed_dim).permute(0, 3, 1, 2)
    tokens_grad_reshaped = tokens_grad.reshape(batch_size, h, w, embed_dim).permute(0, 3, 1, 2)
    
    weights = tokens_grad_reshaped.mean(dim=(2, 3), keepdim=True)
    cam = (weights * tokens_reshaped).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))
    
    confidence = torch.softmax(output, dim=1)[0, target_class].item()
    return cam, target_class, confidence

def visualize_and_save_results(model, input_tensor, original_image, model_name, class_names, target_class=None, save_path=None):
    if 'ViT' in model_name:
        if model_name in ['EnhancedViT', 'InterpretableViT']:
            cam, pred_class, confidence = compute_gradcam_vit(model, input_tensor, target_class)
        else:
            rollout = AttentionRollout(model)
            with torch.no_grad():
                output = model(input_tensor)
            attn_map = rollout.get_attention_map()
            
            h = w = int(np.sqrt(attn_map.shape[0]))
            cam = attn_map.reshape(h, w).numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cv2.resize(cam, (224, 224))
            pred_class = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    else:
        if 'VGG' in model_name:
            target_layer = model.vgg.features[-1]
        elif 'ResNet' in model_name:
            target_layer = model.resnet.layer4[-1]
        elif 'Inception' in model_name:
            target_layer = model.inception.Mixed_7c
        else:  # Custom CNN
            target_layer = model.conv_layers[-2]
        
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam.generate_cam(input_tensor, target_class)
        pred_class = target_class
        confidence = torch.softmax(model(input_tensor), dim=1)[0, pred_class].item()
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("Grad-CAM Map" if 'ViT' in model_name and model_name in ['EnhancedViT', 'InterpretableViT'] else "Attention Map")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay\nPredicted: {class_names[pred_class]}, Conf: {confidence:.2%}")
    plt.axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_model(model, test_loader, device, model_name, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    wrong_predictions = []
    
    # Create directory for model's interpretability results
    interpretability_dir = f'test_results/interpretability/{model_name}'
    os.makedirs(interpretability_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if model_name in ['InterpretableViT']:
                outputs, _ = model(inputs)
            elif model_name in ['InceptionV3Modified']:
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[1]
            else:
                outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Track wrong predictions and save interpretability visualizations
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    # Save wrong prediction info
                    wrong_predictions.append({
                        'image_path': test_loader.dataset.samples[batch_idx * test_loader.batch_size + i][0],
                        'true_label': class_names[labels[i].item()],
                        'predicted_label': class_names[preds[i].item()]
                    })
                    
                    # Save interpretability visualization
                    original_image = np.array(Image.open(wrong_predictions[-1]['image_path']).resize((224, 224)))
                    save_path = os.path.join(interpretability_dir, f'wrong_pred_{len(wrong_predictions)}.png')
                    visualize_and_save_results(
                        model, inputs[i:i+1], original_image, model_name,
                        class_names, preds[i].item(), save_path
                    )
    
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'test_results/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Save wrong predictions to JSON
    with open(f'test_results/wrong_predictions/{model_name}_wrong_predictions.json', 'w') as f:
        json.dump(wrong_predictions, f, indent=4)
    
    return accuracy, cm, wrong_predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['Glaucous_Winged_Gull', 'Slaty_Backed_Gull']
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(root='D:/FYPSeagullClassification01/Test_Results/Test_Data', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model configurations
    model_configs = [
        {
            'name': 'ImprovedCNN',
            'model_class': ImprovedCNN,
            'path': 'D:/FYP/MODELS/CustomModel/HQ2_20241121_173047/final_model_20241121_173047.pth'
        },
        {
            'name': 'EnhancedViT',
            'model_class': EnhancedViT,
            'path': 'D:/FYP/MODELS/VIT/EnhancedViT_20250213/final_model_vit_20250213.pth'
        },
        {
            'name': 'InterpretableViT',
            'model_class': InterpretableViT,
            'path': 'D:/FYP/MODELS/VIT/InterpretableViT_20250213/final_model_vit_20250213.pth'
        },
        {
            'name': 'InceptionV3Modified',
            'model_class': InceptionV3Modified,
            'path': 'D:/FYP/MODELS/InceptionModel/HQ2ltst_20241121/final_model_inception_20241121.pth'
        },
        {
            'name': 'ResNet50Modified',
            'model_class': ResNet50Modified,
            'path': 'D:/FYP/MODELS/ResNet/ResNetHQ2/20241208/final_model.pth'
        },
        {
            'name': 'VGG16Modified',
            'model_class': VGG16Modified,
            'path': 'D:/FYP/MODELS/VGGModel/HQ3latst_20250210/best_model_vgg_20250210.pth'
        }
        # {
        #     'name': 'VGG16Custom',
        #     'model_class': VGG16Custom,
        #     'path': 'D:/FYP/MODELS/VGGModel/your_vgg_custom_model.pth'
        # },
        # {
        #     'name': 'ViTModified',
        #     'model_class': ViTModified,
        #     'path': 'D:/FYP/MODELS/VIT/VIT2_HQ2_20241224/final_model_vit_20241224.pth'
        # }
    ]
    
    results = []
    for config in model_configs:
        print(f"\nEvaluating {config['name']}...")
        
        try:
            model = config['model_class']()
            model.load_state_dict(torch.load(config['path'], map_location=device))
            model = model.to(device)
            
            accuracy, cm, wrong_predictions = evaluate_model(
                model, test_loader, device, config['name'], class_names
            )
            
            results.append({
                'model': config['name'],
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'wrong_predictions_count': len(wrong_predictions)
            })
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Number of wrong predictions: {len(wrong_predictions)}")
            
        except Exception as e:
            print(f"Error evaluating {config['name']}: {str(e)}")
    
    # Save overall results
    with open('test_results/test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main() 