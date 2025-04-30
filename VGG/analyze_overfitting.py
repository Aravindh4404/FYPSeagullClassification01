# WORKS ONLY FOR CHECKPOINTS

# Use
# python  VGG\analyze_overfitting.py --model_dir "D:\MODELS\VGGModel" --models "HQ3latest_20250429\_valloss\early_stopping_model_vgg_20250429.pth" "HQ3latest_20250426\checkpoint_model_vgg_20250426.pth" --test_data "D:\FYPSeagullClassification01\Test_Results\Test_Data" --output_dir "D:\FYP\OverfitCheck\comparison"
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import csv
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets

# Use Pre-trained VGG-16 model and modify it for binary classification
class VGG16Modified(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(VGG16Modified, self).__init__()
        from torchvision.models import vgg16, VGG16_Weights

        # Load pre-trained VGG16 model
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Replace the classifier with a custom classification layer
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.vgg(x)


def extract_metrics_from_checkpoint(checkpoint_path):
    """Extract training metrics from a model checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract epoch number from filename if not in checkpoint
        if 'epoch' not in checkpoint:
            match = re.search(r'_e(\d+)_', os.path.basename(checkpoint_path))
            epoch = int(match.group(1)) if match else -1
        else:
            epoch = checkpoint.get('epoch', -1)
            
        metrics = {
            'model_name': os.path.basename(checkpoint_path),
            'epoch': epoch,
            'train_loss': checkpoint.get('train_loss', None),
            'val_loss': checkpoint.get('val_loss', None),
            'val_acc': checkpoint.get('val_acc', None),
        }
        return metrics
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def evaluate_model_on_test_data(model_path, test_data_dir):
    """Evaluate a model on test data and return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VGG16Modified().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Data transformations - ensure they match your training transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluation
    correct = 0
    total = 0
    test_loss = 0
    
    true_labels = []
    predicted_labels = []
    confidences = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            test_loss += loss.item()
            
            probabilities = F.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
    
    test_acc = correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    return {
        "test_acc": test_acc,
        "test_loss": avg_test_loss,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "confidences": confidences
    }


def analyze_models(model_paths, test_data_dir, output_dir):
    """Analyze multiple models for overfitting."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_list = []
    test_metrics_list = []
    
    # Extract metrics from checkpoints and evaluate on test data
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(f"Processing {model_name}...")
        
        # Extract saved metrics
        checkpoint_metrics = extract_metrics_from_checkpoint(model_path)
        if checkpoint_metrics:
            metrics_list.append(checkpoint_metrics)
        
        # Evaluate on test data
        test_metrics = evaluate_model_on_test_data(model_path, test_data_dir)
        test_metrics['model_name'] = model_name
        test_metrics_list.append(test_metrics)
    
    # Create DataFrame and save to CSV
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_csv_path = os.path.join(output_dir, "model_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved metrics to {metrics_csv_path}")
        
        # Plot metrics
        create_overfitting_plots(metrics_df, output_dir)
    
    # Create test metrics DataFrame
    test_summary = []
    for metrics in test_metrics_list:
        test_summary.append({
            'model_name': metrics['model_name'],
            'test_acc': metrics['test_acc'],
            'test_loss': metrics['test_loss']
        })
    
    if test_summary:
        test_df = pd.DataFrame(test_summary)
        test_csv_path = os.path.join(output_dir, "test_metrics.csv")
        test_df.to_csv(test_csv_path, index=False)
        print(f"Saved test metrics to {test_csv_path}")
        
        # Plot test metrics
        plot_test_comparison(test_df, output_dir)
    
    # If we have both training and test data
    if metrics_list and test_summary:
        # Create combined metrics
        combined_metrics = []
        for train_metrics in metrics_list:
            model_name = train_metrics['model_name']
            # Find matching test metrics
            test_entry = next((t for t in test_summary if t['model_name'] == model_name), None)
            if test_entry:
                entry = {
                    'model_name': model_name,
                    'epoch': train_metrics['epoch'],
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': train_metrics['val_loss'], 
                    'val_acc': train_metrics['val_acc'],
                    'test_loss': test_entry['test_loss'],
                    'test_acc': test_entry['test_acc']
                }
                combined_metrics.append(entry)
        
        if combined_metrics:
            combined_df = pd.DataFrame(combined_metrics)
            combined_csv_path = os.path.join(output_dir, "combined_metrics.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"Saved combined metrics to {combined_csv_path}")
            
            # Plot train/val/test comparison
            plot_complete_comparison(combined_df, output_dir)


def create_overfitting_plots(metrics_df, output_dir):
    """Create plots to visualize potential overfitting."""
    plt.figure(figsize=(12, 8))
    
    # Sort by epoch for proper progression
    metrics_df = metrics_df.sort_values('epoch')
    
    # Plot loss curves
    plt.subplot(2, 1, 1)
    for model_name, group in metrics_df.groupby('model_name'):
        plt.plot(group['epoch'], group['train_loss'], 'o-', label=f'{model_name} - Train Loss')
        plt.plot(group['epoch'], group['val_loss'], 's--', label=f'{model_name} - Val Loss')
    
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(2, 1, 2)
    for model_name, group in metrics_df.groupby('model_name'):
        plt.plot(group['epoch'], group['val_acc'], 'o-', label=f'{model_name} - Val Accuracy')
    
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300)
    plt.close()


def plot_test_comparison(test_df, output_dir):
    """Plot test metrics comparison between models."""
    plt.figure(figsize=(10, 12))
    
    # Sort by test accuracy for better visualization
    test_df = test_df.sort_values('test_acc')
    
    # Plot test accuracy
    plt.subplot(2, 1, 1)
    sns.barplot(x='model_name', y='test_acc', data=test_df)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Plot test loss
    plt.subplot(2, 1, 2)
    sns.barplot(x='model_name', y='test_loss', data=test_df)
    plt.title('Test Loss Comparison')
    plt.xlabel('Model')
    plt.ylabel('Loss')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_metrics_comparison.png"), dpi=300)
    plt.close()


def plot_complete_comparison(combined_df, output_dir):
    """Plot train/val/test comparison for detecting overfitting."""
    plt.figure(figsize=(12, 10))
    
    # Plot loss comparison
    plt.subplot(2, 1, 1)
    model_names = combined_df['model_name'].unique()
    x = np.arange(len(model_names))
    width = 0.25
    
    train_losses = [combined_df[combined_df['model_name'] == name]['train_loss'].values[0] for name in model_names]
    val_losses = [combined_df[combined_df['model_name'] == name]['val_loss'].values[0] for name in model_names]
    test_losses = [combined_df[combined_df['model_name'] == name]['test_loss'].values[0] for name in model_names]
    
    plt.bar(x - width, train_losses, width, label='Train Loss')
    plt.bar(x, val_losses, width, label='Validation Loss')
    plt.bar(x + width, test_losses, width, label='Test Loss')
    
    plt.xlabel('Model')
    plt.ylabel('Loss')
    plt.title('Loss Comparison Across Datasets')
    plt.xticks(x, [name.split('.')[0] for name in model_names], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Plot accuracy comparison
    plt.subplot(2, 1, 2)
    val_accs = [combined_df[combined_df['model_name'] == name]['val_acc'].values[0] for name in model_names]
    test_accs = [combined_df[combined_df['model_name'] == name]['test_acc'].values[0] for name in model_names]
    
    plt.bar(x - width/2, val_accs, width, label='Validation Accuracy')
    plt.bar(x + width/2, test_accs, width, label='Test Accuracy')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks(x, [name.split('.')[0] for name in model_names], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complete_comparison.png"), dpi=300)
    plt.close()
    
    # Create overfitting visualization
    plt.figure(figsize=(10, 6))
    
    # Calculate overfitting metrics
    combined_df['train_val_diff'] = combined_df['train_loss'] - combined_df['val_loss']
    combined_df['val_test_diff'] = abs(combined_df['val_acc'] - combined_df['test_acc'])
    
    sns.scatterplot(
        x='train_val_diff', 
        y='val_test_diff', 
        data=combined_df,
        s=100,
        alpha=0.7
    )
    
    # Add labels for each point
    for i, row in combined_df.iterrows():
        plt.text(
            row['train_val_diff'], 
            row['val_test_diff'], 
            row['model_name'].split('.')[0],
            fontsize=9
        )
    
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.title('Overfitting Analysis')
    plt.xlabel('Train Loss - Validation Loss (negative = possible underfitting)')
    plt.ylabel('|Validation Acc - Test Acc| (higher = possible overfitting)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overfitting_analysis.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze models for overfitting')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model checkpoints')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='overfitting_analysis', help='Output directory for results')
    parser.add_argument('--models', type=str, nargs='+', help='Specific model filenames to analyze (optional)')
    
    args = parser.parse_args()
    
    # Get model paths
    if args.models:
        model_paths = [os.path.join(args.model_dir, model) for model in args.models]
    else:
        model_paths = [os.path.join(args.model_dir, f) for f in os.listdir(args.model_dir) 
                      if f.endswith('.pth') or f.endswith('.pt')]
    
    # Run analysis
    analyze_models(model_paths, args.test_data, args.output_dir)
    print("Analysis complete!")