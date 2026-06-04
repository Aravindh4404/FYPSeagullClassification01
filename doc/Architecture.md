# System Architecture

## Overview

The system has three parallel analysis pipelines, all operating on the same two-species seagull image dataset:

```
Raw Images + Segmentation Masks
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ▼                                 ▼
Deep Learning                  Feature Analysis
Pipeline                       Pipeline
    │                                 │
    │  ┌──────────────┐         ┌─────┴──────────┐
    │  │ Model Train  │         │ Intensity       │
    │  │ (Notebooks)  │         │ Analysis        │
    │  └──────┬───────┘         └────────┬────────┘
    │         │                          │
    │  ┌──────▼───────┐         ┌────────▼────────┐
    │  │ test_models  │         │ LBP Texture     │
    │  │ .py          │         │ Analysis        │
    │  └──────┬───────┘         └────────┬────────┘
    │         │                          │
    │  Accuracy, CM,                     │
    │  Grad-CAM                 ┌────────▼────────┐
    │                           │ Clustering      │
    │                           │ Analysis        │
    │                           └────────┬────────┘
    │                                    │
    └─────────────────────────────── Results ──────
```

---

## Main Modules

### 1. Deep Learning Models (`MODEL_NOTEBOOKS/`, `Test_Results/`)

Seven model architectures, all PyTorch, all binary classification (2 outputs):

| Model | Base | Custom Head | Notes |
|-------|------|-------------|-------|
| `ImprovedCNN` | Custom | 3×Conv+SEBlock → FC512 → FC2 | SE (Squeeze-Excite) attention |
| `VGG16Modified` | VGG16 (ImageNet) | Dropout(0.4)+Linear(4096→2) | Replaces classifier[6] |
| `VGG16Custom` | VGG16 (ImageNet) | Dropout(0.5)+Linear → 2 | Variant with higher dropout |
| `ResNet50Modified` | ResNet50 (ImageNet) | Dropout(0.5)+Linear → 2 | Replaces fc layer |
| `InceptionV3Modified` | InceptionV3 (ImageNet) | Dropout(0.5)+Linear → 2 | Handles aux output in train |
| `EnhancedViT` | ViT-B/16 (timm, ImageNet) | Attention pool → LayerNorm+Dropout+Linear | Soft attention over tokens |
| `InterpretableViT` | ViT-B/16 (timm, ImageNet) | CLS+weighted-patch concat → Linear | Dual-stream for interpretability |

Training is done in Google Colab (Jupyter notebooks), models saved as `.pth` files at `D:/FYP/MODELS/`.

Input transform for all models:
- Resize to 224×224
- ToTensor
- Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ← ImageNet stats

### 2. Features Analysis (`Features_Analysis/`)

#### 2a. Region Segmentation (`config.py`)
- Segmentation masks use exact BGR colours to identify four body regions:
  - **Wingtip**: `(0,255,0)` (green in BGR)
  - **Wing**: `(0,0,255)` (red)
  - **Head**: `(255,255,0)` (yellow)
  - **Body**: `(0,255,255)` (sky-blue)
- `extract_region_mask(seg_img, region_name)` → binary mask
- `extract_region(orig_img, seg_img, region_name)` → masked pixel values
- `get_image_paths(species)` → list of `(image_path, segmentation_path)` pairs

#### 2b. Intensity Analysis (`Intensity/`)
- Loads wing pixel intensity values from CSV
- Computes mean, std, skewness, kurtosis per species
- Runs Welch t-test to confirm statistical significance
- `areawingtipdarkerthanwing*.py` scripts directly test the wingtip-darker-than-wing hypothesis on raw images

#### 2c. LBP Texture Analysis (`LBP/LBP.py`)
- Radius=1, Points=8, Method='default'
- Extracts LBP histograms (256 bins) per region mask
- Compares species using chi-square distance
- Trains SVM (RBF kernel) per region as baseline classifier
- Outputs: histogram comparison charts, texture metrics (uniformity, entropy, contrast, energy, smoothness)

### 3. Clustering (`Clustering/`)

- **Input**: CSV with per-image features: `mean_wing_intensity`, `mean_wingtip_intensity`, `percentage_dark_pixels_wingtip`
- **Pipeline**: StandardScaler → PCA(2) → KMeans / AgglomerativeClustering / GMM
- **Evaluation**: Silhouette score, Adjusted Rand Index, confusion matrix
- **Outputs**: PCA biplot with feature loading vectors, overlap-zone bird identification, cluster-to-species mapping

---

## Data Flow (High Level)

```
Bird Images ──→ Segmentation Masks
     │                  │
     └──────────────────┘
              │
        config.py
        extract_region_mask()
              │
     ┌────────┴─────────┐
     │                  │
  Intensity           LBP
  Analysis         Analysis
     │                  │
  per-image          per-region
  intensity CSV      histogram
     │                  │
     └────────┬─────────┘
              │
         Clustering
         (CSV input)
              │
         KMeans/Hier/GMM
              │
         Results CSV/PNG
```

---

## Training Pipeline

1. Load dataset with `torchvision.datasets.ImageFolder`
2. Apply transforms (resize, normalize to ImageNet stats)
3. Train in Colab with cross-entropy loss + Adam optimizer
4. Early stopping on validation loss / accuracy
5. Save best model to `.pth`

## Inference Pipeline

1. Load `.pth` with `torch.load(..., map_location=device)`
2. `model.eval()` + `torch.no_grad()`
3. Apply same transforms as training
4. `model(input_tensor)` → logits → `argmax` → class index
5. Map index 0→`Glaucous_Winged_Gull`, 1→`Slaty_Backed_Gull`

## Interpretability Pipeline

- **CNN models** (VGG, ResNet, Inception, Custom): GradCAM on last conv layer
- **EnhancedViT / InterpretableViT**: Grad-CAM on token activations via backward hook
- **ViTModified**: Attention Rollout (average multi-head attention across blocks)
- Output: heatmap overlaid on original image, saved to `test_results/interpretability/<model>/`
