# FYP Seagull Classification — Project Documentation

## What This Project Does

This project classifies two visually similar seagull species — **Slaty-Backed Gull (SBG)** and **Glaucous-Winged Gull (GWG)** — using a combination of deep learning and traditional computer-vision feature analysis.

The core scientific question is whether morphological differences in wing and wingtip pigmentation can be detected reliably by machine learning models, providing a tool for ornithological field identification.

---

## Main Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Deep Learning Models | `MODEL_NOTEBOOKS/`, `Colab Notebooks/` | Train VGG16, ResNet50, InceptionV3, ViT, Custom CNN |
| Model Testing | `Test_Results/test_models.py` | Evaluate all models on held-out test set |
| Intensity Analysis | `Features_Analysis/Intensity/` | Measure wing/wingtip pixel intensity difference |
| LBP Texture Analysis | `Features_Analysis/LBP/` | Extract LBP texture features per body region |
| Clustering | `Clustering/clustering_latest.py` | K-means, Hierarchical, GMM on extracted features |
| Image Utilities | `checkimage.py`, `checksame.py` | Dataset validation tools |
| Segmentation Config | `Features_Analysis/config.py` | Region colours, mask extraction helpers |
| Interpretability | `Interpretability/`, `Test_Results/test_models.py` | Grad-CAM and Attention Rollout visualisations |

---

## How to Run

### Requirements
```bash
pip install -r requirements.txt
```
Key dependencies: PyTorch 2.6, torchvision 0.21, timm 1.0.14, scikit-learn 1.6, OpenCV 4.11, Pandas 2.2, Matplotlib 3.10, Seaborn 0.13, scikit-image 0.25.

### Dataset Structure Expected
```
D:/Dataset_FYP/Interpretable_Seagull_Classification/Model_Training_Dataset/
    train/
        Glaucous_Winged_Gull/   ← JPEG images
        Slaty_Backed_Gull/
    test/
        Glaucous_Winged_Gull/
        Slaty_Backed_Gull/

Features_Analysis/Dataset/
    Original_Images/
        Glaucous_Winged_Gull/
        Slaty_Backed_Gull/
    Colored_Images/             ← colour-coded segmentation masks
        Glaucous_Winged_Gull/
        Slaty_Backed_Gull/
```

### Test All Deep-Learning Models
```bash
cd Test_Results
python test_models.py
```
Outputs: `test_results/test_results.json`, per-model confusion matrix PNGs, Grad-CAM overlays for wrong predictions.

### Run Clustering Analysis
```bash
cd Clustering
python clustering_latest.py
```
Input: `Clustering/dark_pixel_results_all_images.csv`
Outputs: `clustering_results/` — PNGs, CSVs of cluster assignments and PCA biplots.

### Run LBP Feature Analysis
```bash
cd Features_Analysis
python LBP/LBP.py
```
Outputs: `Outputs/LBP_Analysis/<run_folder>/` — histograms, classification accuracy per region.

### Run Intensity Analysis
```bash
cd Features_Analysis/Intensity
python Wing_Intensity_Analysis.py
```
Input: `Intensity_Results/wing_intensity_analysis.csv`
Outputs: `Wing_Analysis_Results/` — summary CSVs and t-test results.

### Validate Dataset Consistency
```bash
python checksame.py      # checks folder images vs CSV entries match
python checkimage.py     # finds images in folder A absent from folder B
```

---

## Inputs and Outputs

### Inputs
- Bird photographs (JPEG/PNG), one folder per species
- Colour-coded segmentation masks aligned 1:1 with originals
  - Green pixels = wingtip, Red = wing, Yellow = head, Sky-blue = body
- Pre-trained `.pth` PyTorch model weights (stored outside repo at `D:/FYP/MODELS/`)
- Feature CSVs (`dark_pixel_results_all_images.csv`, `wing_intensity_analysis.csv`)

### Outputs
- Classification accuracy per model (JSON + console)
- Confusion matrices (PNG)
- Grad-CAM / Attention-Rollout overlays for misclassified images
- PCA biplots of feature space (PNG)
- LBP histogram comparisons per body region (PNG)
- Statistical test results (CSV): t-test p-values for intensity differences
- Overlap-zone bird lists (CSV): birds with intermediate feature profiles

---

## Key Finding
Slaty-Backed Gulls have significantly darker wingtips relative to their wing body than Glaucous-Winged Gulls. This single feature — `percentage_dark_pixels_wingtip` — drives most clustering separation and is the primary discriminating signal for all trained models.
