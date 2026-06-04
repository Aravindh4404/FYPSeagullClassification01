# Data Flow

## 1. Source Data

### 1a. Raw Images
- **Location**: `D:/Dataset_FYP/Interpretable_Seagull_Classification/Model_Training_Dataset/` (deep learning) and `Features_Analysis/Dataset/Original_Images/` (feature analysis)
- **Format**: JPEG / PNG, variable resolution
- **Classes**: `Glaucous_Winged_Gull/` and `Slaty_Backed_Gull/` subdirectories

### 1b. Segmentation Masks
- **Location**: `Features_Analysis/Dataset/Colored_Images/`
- **Format**: Same resolution as originals; each pixel is one of four exact BGR colours or background black
- **Regions encoded**:
  - `(0,255,0)` → wingtip
  - `(0,0,255)` → wing
  - `(255,255,0)` → head
  - `(0,255,255)` → body
- **Pairing**: Matched by filename stem (e.g. `bird001.jpg` ↔ `bird001.png`)

---

## 2. Deep Learning Data Flow

```
Dataset folder (train/test split)
        │
        ▼
torchvision.datasets.ImageFolder
        │  class_to_idx: {Glaucous_Winged_Gull:0, Slaty_Backed_Gull:1}
        │
        ▼
transforms.Compose([
    Resize(224,224),
    ToTensor(),          # HxWxC uint8 → CxHxW float [0,1]
    Normalize(mean=[0.485,0.456,0.406],
              std= [0.229,0.224,0.225])  # ImageNet stats
])
        │
        ▼
DataLoader(batch_size=32, shuffle=True/False)
        │  Yields: (images: [B,3,224,224], labels: [B])
        │
        ▼
model.forward(images)
        │  Returns: logits [B,2]
        │
        ▼
CrossEntropyLoss(logits, labels)   ← training
torch.argmax(logits, dim=1)        ← inference
        │
        ▼
Predictions mapped:  0 → Glaucous_Winged_Gull
                     1 → Slaty_Backed_Gull
```

---

## 3. Feature Analysis Data Flow

### 3a. From Images to Feature CSV

```
Original Image + Segmentation Mask
        │
        ▼
config.extract_region_mask(seg_img, 'wingtip')
        │  Returns: binary mask (0/255)
        │
        ▼
cv2.bitwise_and(orig, orig, mask=mask)
        │  Extracts wingtip pixels
        │
        ▼
Pixel intensity calculation (grayscale)
  - mean_wingtip_intensity
  - mean_wing_intensity
  - percentage_dark_pixels_wingtip  (pixels < threshold / total)
        │
        ▼
Row added to dark_pixel_results_all_images.csv
  columns: image_name, species, mean_wing_intensity,
           mean_wingtip_intensity, method3_percentage
```

### 3b. LBP Pipeline

```
Original Image + Segmentation Mask
        │
        ▼
config.extract_region_mask(seg, region)  for each region in
        │  {wingtip, wing, head, body}
        │
        ▼
cv2.cvtColor(img, BGR→GRAY)
        │
        ▼
skimage.feature.local_binary_pattern(gray, P=8, R=1, method='default')
        │  Returns: LBP image same shape as gray
        │
        ▼
np.histogram(lbp[mask>0], bins=256, range=(0,256), density=True)
        │  Returns: normalised histogram (256,)
        │
        ▼
Feature vector per region per image
        │
        ▼
SVM(kernel='rbf').fit(X_train, y_train)
        │
        ▼
Accuracy score + chi-square distance between species histograms
```

---

## 4. Clustering Data Flow

```
dark_pixel_results_all_images.csv
  columns: image_name, species, mean_wing_intensity,
           mean_wingtip_intensity, percentage_dark_pixels_wingtip
        │
        ▼
load_data()  →  features DataFrame (3 columns), true_labels Series
        │
        ▼
StandardScaler.fit_transform(features)  →  X_scaled [N,3]
        │
        ▼
PCA(n_components=2).fit_transform(X_scaled)  →  X_pca [N,2]
        │
        ├── visualize_pca_biplot()  →  PNG with feature loading arrows
        │
        ├── identify_overlap_zone_birds()  →  CSV of boundary birds
        │
        ▼
KMeans(k=2)          →  cluster labels, centroids
AgglomerativeClustering(k=2, linkage='ward')  →  cluster labels
GaussianMixture(k=2) →  cluster labels, means
        │
        ▼
evaluate_clustering()  →  silhouette_score, adjusted_rand_index
        │
        ▼
map_clusters_to_species()  →  majority-class mapping
        │
        ▼
confusion matrix PNG + misclassified CSVs
```

---

## 5. Output Summary

| Pipeline | Output Location | Content |
|----------|----------------|---------|
| DL Training | `D:/FYP/MODELS/<model>/` | `.pth` weight files |
| DL Testing | `test_results/` | `test_results.json`, confusion PNGs, Grad-CAM overlays |
| Intensity | `Features_Analysis/Intensity/Wing_Analysis_Results/` | `wing_intensity_summary.csv`, t-test CSV |
| LBP | `Features_Analysis/Outputs/LBP_Analysis/<run>/` | histogram PNGs, `analysis_summary.txt` |
| Clustering | `Clustering/clustering_results/` | PCA plots, cluster CSVs, `console_output.txt` |
| Dataset validation | root | `dataset_validation_report.txt` (when run) |
