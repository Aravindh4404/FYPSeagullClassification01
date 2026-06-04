# API Reference — All Python Files

## Root Level

### `checkimage.py`
**Purpose**: Detects images present in folder A but absent from folder B using perceptual hashing.

| Symbol | Type | Input | Output | Notes |
|--------|------|-------|--------|-------|
| `get_image_hashes(folder_path)` | function | `str` folder path | `dict` {filename: (hash, path)} | Uses `imagehash.average_hash` |
| `find_absent_images(folder_A, folder_B, similarity_threshold=0)` | function | two folder paths, int threshold | `list[str]` of absent image names | Threshold 0 = exact match only |

---

### `checksame.py`
**Purpose**: Validates that images in `Glaucous_Winged_Gull/` and `Slaty_Backed_Gull/` folders exactly match CSV entries.

| Symbol | Type | Input | Output | Notes |
|--------|------|-------|--------|-------|
| `validate_dataset_csv_match(dataset_path, csv_path)` | function | two paths | `dict` with counts and discrepancy lists | CSV must have `image_name` and `species` columns |
| `print_validation_results(results)` | function | results dict | None (prints) | — |
| `save_validation_report(results)` | function | results dict | `str` output filename | Saves to `dataset_validation_report.txt` |
| `main()` | function | — | None | Interactive: asks to save report |

---

### `visualize_normalization.py`
**Purpose**: Visualises ImageNet normalisation effect on a single image.

| Symbol | Type | Input | Output | Notes |
|--------|------|-------|--------|-------|
| `normalize_image(image, mean, std)` | function | `np.ndarray` RGB, mean list, std list | `np.ndarray` normalised | Scales to [0,1] first |
| `visualize_normalization(image_path, mean, std)` | function | path string, lists | None (shows plot) | Side-by-side original vs normalised |

---

### `deleteextra.py`
**Purpose**: Utility script for removing extra/duplicate files from dataset folders.

---

### `noextrasinfolders.py`
**Purpose**: Checks that no extra (unexpected) files exist in dataset folders.

---

### `removefromcsv.py`
**Purpose**: Removes specific rows from analysis CSV files.

---

## `Features_Analysis/config.py`
**Purpose**: Central configuration — species directories, region colour codes, mask extraction helpers. Imported by LBP.py and other analysis scripts.

| Symbol | Type | Description |
|--------|------|-------------|
| `BASE_DIR` | `Path` | Directory of config.py |
| `SLATY_BACKED_IMG_DIR` | `Path` | `dataset/Original_Images/Slaty_Backed_Gull` |
| `SLATY_BACKED_SEG_DIR` | `Path` | `dataset/Colored_Images/Slaty_Backed_Gull` |
| `GLAUCOUS_WINGED_IMG_DIR` | `Path` | Same structure for GWG |
| `GLAUCOUS_WINGED_SEG_DIR` | `Path` | — |
| `S` | `int` | 100 — default image limit per species |
| `REGION_COLORS` | `dict` | `{wingtip:(0,255,0), wing:(0,0,255), head:(255,255,0), body:(0,255,255)}` in BGR |
| `ENTIRE_BIRD_COLORS` | `list` | All region BGR values combined |
| `SPECIES` | `dict` | Maps species name → `{img_dir, seg_dir}` |
| `get_image_paths(species)` | function | species name str | `list[(img_path, seg_path)]` | Pairs by matching filename stem |
| `get_region_masks(segmentation)` | function | BGR segmentation image | `(masks_dict, stats_dict)` | stats include center pixel and pixel count |
| `get_region_color(region_name)` | function | region name str | BGR tuple | Raises ValueError if unknown |
| `extract_region_mask(segmentation_img, region_name)` | function | BGR seg image, region name | binary mask `uint8` | `cv2.inRange` on exact colour; supports `'entire_bird'` |
| `extract_region(original_img, segmentation_img, region_name)` | function | two BGR images, region name | `(masked_img, binary_mask)` | Masks original with region colour |

---

## `Features_Analysis/LBP/LBP.py`
**Purpose**: Full LBP texture analysis pipeline — extract, compare, classify.

| Symbol | Type | Input → Output | Notes |
|--------|------|----------------|-------|
| `RADIUS` | int | — | 1 |
| `N_POINTS` | int | — | 8 |
| `METHOD` | str | — | `'default'` |
| `get_image_paths(species)` | function | species name | `list[(img_path, seg_path)]` | Local override of config version |
| `get_region_masks(segmentation, debug=False)` | function | BGR seg image | `(masks_dict, stats_dict[, debug_img])` | — |
| `extract_lbp_features(image, mask, region_stats, region_name, debug)` | function | BGR image + optional mask | `(lbp, masked_lbp, hist[, debug_imgs])` | Returns 256-bin normalised histogram |
| `visualize_region_lbp_details(image_path, seg_path, region_name, output_dir)` | function | paths + region name | histogram array; saves PNGs | 6-panel debug figure |
| `analyze_species_texture(species_name, limit, debug)` | function | species name, int limit | `(region_features_dict, debug_outputs_dict)` | Collects histograms across all images |
| `visualize_lbp_comparison(species_data, output_filename)` | function | dict, str | `dict` chi-sq distances | Saves comparison bar charts |
| `chi_square_distance(hist1, hist2)` | function | two np arrays | float | 0.5 * sum((h1-h2)²/(h1+h2)) |
| `analyze_texture_properties(species_data)` | function | species data dict | `dict` texture metrics | Metrics: uniformity, entropy, contrast, energy, smoothness |
| `build_feature_dataset(species_data)` | function | species data dict | `(X, y, region_info)` arrays | Flattens all histograms into feature matrix |
| `evaluate_classification(X, y, region_info)` | function | arrays | `dict` per-region accuracy | SVM(RBF), train/test split or CV |
| `visualize_lbp_patterns(species_paths)` | function | paths dict | None; saves PNGs | Side-by-side: original, seg, LBP |
| `visualize_region_lbp(species_paths)` | function | paths dict | None; saves PNGs | Per-region LBP centre patch |
| `experiment_with_parameters(img_path, seg_path, region_name, species_name, output_dir)` | function | paths + strings | None; saves grid PNG | Tests R∈{1,2,3,5,8}, P∈{8,16,24} |
| `main()` | function | — | None | Entry point: full analysis + SVM eval |

---

## `Features_Analysis/Intensity/Wing_Intensity_Analysis.py`
**Purpose**: Statistical comparison of wing mean intensity between species.

| Symbol | Type | Description |
|--------|------|-------------|
| `(script-level)` | — | Loads CSV, groups by species, runs Welch t-test, saves two CSVs |

**Key outputs**: `wing_intensity_summary.csv`, `wing_intensity_ttest.csv`

---

## `Features_Analysis/utils.py`
**Purpose**: Shared utility helpers for the features analysis module.

---

## `Clustering/clustering_latest.py`
**Purpose**: Full unsupervised clustering pipeline on the three-feature intensity dataset.

| Symbol | Type | Input → Output | Notes |
|--------|------|----------------|-------|
| `load_data(file_path)` | function | CSV path | `(df, features, true_labels, species_names)` | Renames `method3_percentage` column |
| `preprocess_data(features)` | function | DataFrame | `(X_scaled, scaler)` | StandardScaler |
| `apply_pca(X_scaled, n_components=2)` | function | scaled array | `(X_pca, pca, explained_variance)` | 2-component PCA |
| `visualize_pca_biplot(...)` | function | PCA results + labels | `loadings_df`; saves PNG | Feature loading arrows on scatter |
| `identify_overlap_zone_birds(X_pca, df, ...)` | function | PCA coords + data | `(overlap_df, overlap_birds)` | Bottom 15% by distance to boundary |
| `highlight_population_on_pca(...)` | function | PCA + population column | None; saves PNG | Marks specific population with stars |
| `evaluate_clustering(X_scaled, clusters, true_labels)` | function | arrays | `dict` with silhouette + ARI | — |
| `visualize_clusters(...)` | function | PCA + cluster labels | None; saves PNG | Scatter coloured by cluster |
| `visualize_pca_by_species(...)` | function | PCA + species labels | None; saves PNG | — |
| `visualize_feature_pairs(...)` | function | df + features | None; saves PNG | N×N scatter matrix |
| `visualize_feature_distributions(...)` | function | df + features | None; saves PNG | Histogram per feature |
| `visualize_feature_boxplots(...)` | function | df + features | None; saves PNG | Box plot per feature |
| `map_clusters_to_species(df, clusters, ...)` | function | cluster labels + true labels | `(analysis_df, majority_mapping)` | Majority-vote labelling |
| `visualize_confusion_matrix_with_mapping(...)` | function | labels + mapping | `confusion_df`; saves PNG | — |
| `visualize_misclassifications(...)` | function | PCA + analysis df | None; saves PNG | X markers on misclassified |
| `visualize_feature_importance(...)` | function | scaler + centers | None; saves PNG | Abs diff between cluster centers |
| `calculate_feature_importance(...)` | function | scaler + centers | `dict` feature→importance | — |
| `calculate_hierarchical_centers(...)` | function | X_scaled + clusters | `ndarray` centers | Mean per cluster |
| `create_feature_importance_table(...)` | function | three importance dicts | `DataFrame` ranked table | Avg importance across models |
| `apply_kmeans(X_scaled, n_clusters=2)` | function | scaled array | `(clusters, centers)` | `KMeans(n_init=10, random_state=42)` |
| `apply_hierarchical(X_scaled, n_clusters=2)` | function | scaled array | `(clusters, None)` | Ward linkage |
| `apply_gmm(X_scaled, n_components=2)` | function | scaled array | `(clusters, means)` | `GaussianMixture` |
| `run_clustering_analysis(file_path, output_dir, ...)` | function | CSV path + output dir | `dict` with all three results | Master pipeline function |
| `export_misclassified_points(analysis_df, algorithm_name, output_path)` | function | analysis df | None; saves CSV | — |

---

## `Test_Results/test_models.py`
**Purpose**: Loads all trained models, runs them on the test set, generates accuracy/confusion/Grad-CAM results.

### Classes

| Class | Base | Purpose |
|-------|------|---------|
| `SEBlock` | `nn.Module` | Squeeze-and-Excite channel attention: FC→ReLU→FC→Sigmoid→channel-wise multiply |
| `ImprovedCNN` | `nn.Module` | Custom CNN: 3×(Conv+BN+ReLU+SEBlock+MaxPool) → Flatten → FC512+Dropout → FC2 |
| `EnhancedViT` | `nn.Module` | ViT-B/16 + soft attention pool over all tokens → LayerNorm+Dropout → FC2 |
| `InterpretableViT` | `nn.Module` | ViT-B/16 → concat(CLS token, attention-weighted patches) → FC2; returns `(logits, attn_weights)` |
| `InceptionV3Modified` | `nn.Module` | InceptionV3 (ImageNet) + Dropout+Linear head; returns `(aux, out)` in train mode |
| `ResNet50Modified` | `nn.Module` | ResNet50 (ImageNet) + Dropout(0.5)+Linear head |
| `VGG16Modified` | `nn.Module` | VGG16 (ImageNet) + Dropout(0.4)+Linear replacing classifier[6] |
| `VGG16Custom` | `nn.Module` | VGG16 (ImageNet) + Dropout(0.5)+Linear |
| `ViTModified` | `nn.Module` | ViT-B/16 (timm) + Linear head on existing head |
| `GradCAM` | — | Registers forward/backward hooks on target layer; `generate_cam(input_tensor)` returns heatmap |
| `AttentionRollout` | — | Registers QKV hooks on ViT blocks; `get_attention_map()` returns CLS attention over patches |

### Functions

| Function | Input → Output | Notes |
|----------|----------------|-------|
| `compute_gradcam_vit(model, input_tensor, target_class)` | model + tensor | `(cam_np, pred_class, confidence)` | Grad-CAM via token grad hooks |
| `visualize_and_save_results(model, input_tensor, original_image, model_name, ...)` | — | None; saves 3-panel PNG | Dispatches Grad-CAM vs Attention-Rollout |
| `evaluate_model(model, test_loader, device, model_name, class_names)` | — | `(accuracy, cm, wrong_predictions)` | Saves JSON + confusion PNG |
| `main()` | — | None | Loads all 6 model configs, evaluates, saves `test_results.json` |

---

## `Features_Analysis/Intensity/areawingtipdarkerthanwing*.py` (family)

These scripts directly test the hypothesis that wingtip pixels are darker than wing pixels for a given species:

- `areawingtipdarkerthanwing.py` — basic pixel count threshold comparison
- `areawingtipdarkerthanwing2.py` — improved version with percentage calculation
- `areawingtipdarkerthanwing3.py` — version 3 with additional filtering
- `areawingtipdarkerthanwingwhitespotsremovedclusters.py` — removes white spot pixels before comparison using clustering
- `areawingtipdarkerthanwingwhitespotsremovedignorehigherintensity.py` — additional variant ignoring high-intensity outliers

---

## `Final Report - Copy/Python script/genlateximages.py`
**Purpose**: Generates LaTeX `\includegraphics` commands for report figures.

---

## `copypastedel/comparenames.py`
**Purpose**: Compares filenames between two folders to find matches/mismatches.

### `copypastedel/copysamenamefromdifferentfolder.py`
**Purpose**: Copies files that share the same name across two different folder structures.
