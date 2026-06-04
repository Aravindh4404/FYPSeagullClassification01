# Call Graph

## Main Entry Points

### 1. `Test_Results/test_models.py` — Deep Learning Evaluation

```
main()
├── torch.device("cuda" if torch.cuda.is_available() else "cpu")
├── transforms.Compose([Resize, ToTensor, Normalize])
├── datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)
├── DataLoader(test_dataset, batch_size=32, shuffle=False)
├── for config in model_configs:
│   ├── config['model_class']()                    → instantiates model
│   ├── torch.load(config['path'], map_location)   → loads .pth weights
│   ├── model.to(device)
│   └── evaluate_model(model, test_loader, device, model_name, class_names)
│       ├── model.eval()
│       ├── for batch in test_loader:
│       │   ├── model(inputs)                      → forward pass
│       │   ├── torch.max(outputs, 1)              → predictions
│       │   └── [if wrong pred]:
│       │       ├── Image.open(path).resize(224,224)
│       │       └── visualize_and_save_results(...)
│       │           ├── [if ViT Enhanced/Interpretable]:
│       │           │   └── compute_gradcam_vit(model, input_tensor)
│       │           │       ├── model(input_tensor)        → forward
│       │           │       ├── score.backward()           → compute grads
│       │           │       ├── model.tokens[:, 1:, :]     → patch tokens
│       │           │       └── (weights * tokens).sum()   → CAM
│       │           ├── [if ViTModified]:
│       │           │   └── AttentionRollout(model).get_attention_map()
│       │           │       └── hooks on blk.attn.qkv for each block
│       │           └── [else CNN/VGG/ResNet/Inception]:
│       │               └── GradCAM(model, target_layer).generate_cam()
│       │                   ├── forward hook: save_activation()
│       │                   ├── backward hook: save_gradient()
│       │                   └── pooled_gradients * activations → heatmap
│       ├── accuracy_score(all_labels, all_preds)
│       ├── confusion_matrix(all_labels, all_preds)
│       └── json.dump(wrong_predictions, f)
└── json.dump(results, 'test_results.json')
```

---

### 2. `Clustering/clustering_latest.py` — Unsupervised Analysis

```
main() [if __name__ == "__main__"]
└── run_clustering_analysis(file_path, output_dir, population_column, population_value)
    ├── load_data(file_path)
    │   └── pd.read_csv() → df, features[3 cols], true_labels, species_names
    ├── preprocess_data(features)
    │   └── StandardScaler().fit_transform() → X_scaled
    ├── apply_pca(X_scaled, n_components=2)
    │   └── PCA(2).fit_transform() → X_pca, pca, explained_variance
    ├── visualize_pca_biplot(X_pca, true_labels, species_names, pca, ...)
    │   └── pca.components_.T * sqrt(pca.explained_variance_) → loadings
    ├── identify_overlap_zone_birds(X_pca, df, true_labels, ...)
    │   └── project points onto centroid-connecting line → boundary distances
    ├── visualize_pca_by_species(X_pca, ...)
    ├── visualize_feature_distributions(df, features, ...)
    ├── visualize_feature_boxplots(df, features, ...)
    ├── visualize_feature_pairs(df, features, ...)
    │
    ├── apply_kmeans(X_scaled)
    │   └── KMeans(n_clusters=2, n_init=10).fit_predict() → clusters, centers
    ├── evaluate_clustering(X_scaled, clusters, true_labels)
    │   └── silhouette_score() + adjusted_rand_score()
    ├── map_clusters_to_species(df, clusters, true_labels, species_names)
    │   └── pd.crosstab() → majority_mapping
    ├── visualize_confusion_matrix_with_mapping(...)
    ├── visualize_misclassifications(X_pca, cluster_analysis, ...)
    ├── visualize_feature_importance(scaler, kmeans_centers, ...)
    │   └── scaler.inverse_transform() → center_diffs
    │
    ├── apply_hierarchical(X_scaled)
    │   └── AgglomerativeClustering(n_clusters=2, linkage='ward').fit_predict()
    │
    ├── apply_gmm(X_scaled)
    │   └── GaussianMixture(n_components=2).fit_predict() → clusters, means
    │
    ├── create_feature_importance_table(kmeans_imp, hierarchical_imp, gmm_imp, ...)
    │
    └── export_misclassified_points(analysis_df, algo, output_path)
        └── analysis_df[~Correctly_Clustered].to_csv()
```

---

### 3. `Features_Analysis/LBP/LBP.py` — Texture Analysis

```
main()
├── RESULT_DIR / DEBUG_DIR / REGION_DIR / COMPARISON_DIR setup
├── for species in SPECIES:
│   └── get_image_paths(species)
│       └── os.listdir(img_dir + seg_dir) → paired by filename stem
│
├── visualize_lbp_patterns(species_paths)
│   └── local_binary_pattern(gray, P=8, R=1, method='default') on full image
│
├── visualize_region_lbp(species_paths)
│   └── for region in REGION_COLORS:
│       └── visualize_region_lbp_details(img_path, seg_path, region, dir)
│           ├── get_region_masks(seg, debug=True)
│           │   └── extract_region_mask(seg, region_name)
│           │       └── cv2.inRange(seg, color, color) → binary mask
│           └── extract_lbp_features(img, mask, region_stats, region_name, debug=True)
│               └── local_binary_pattern() + np.histogram(lbp[mask>0], bins=256)
│
├── experiment_with_parameters(img, seg, 'wingtip', species, RESULT_DIR)
│   └── for R in [1,2,3,5,8], P in [8,16,24]:
│       └── local_binary_pattern(gray_patch, P, R, 'uniform')
│
├── for species in SPECIES:
│   └── analyze_species_texture(species, limit=100, debug=True)
│       └── for (img_path, seg_path):
│           ├── get_region_masks(seg)
│           └── for region in REGION_COLORS:
│               └── extract_lbp_features(img, mask) → hist[256]
│
├── visualize_lbp_comparison(species_data, 'lbp_histogram_comparison.png')
│   └── chi_square_distance(avg_hist_sp1, avg_hist_sp2)
│
├── analyze_texture_properties(species_data)
│   └── uniformity, entropy, contrast, energy, smoothness per region
│
├── build_feature_dataset(species_data)
│   └── flatten all histograms → (X, y, region_info) arrays
│
└── evaluate_classification(X, y, region_info)
    └── SVC(kernel='rbf').fit() + .score() or cross_val_score()
```

---

### 4. `Features_Analysis/Intensity/areawingtipdarkerthanwing.py` — Darkness Hypothesis

```
[script level]
├── config.get_image_paths(species) for each species
├── for (img_path, seg_path):
│   └── analyze_wingtip_dark_pixels(img_path, seg_path, species, ...)
│       ├── cv2.imread(img_path), cv2.imread(seg_path)
│       ├── cv2.cvtColor(orig, BGR→GRAY)
│       ├── cv2.normalize(gray, 0, 255, NORM_MINMAX)
│       ├── config.extract_region(gray, seg, "wingtip") → wingtip_pixels
│       └── count pixels < mean_wing_intensity → percentage_dark
└── collect results → CSV + plots
```

---

## Key Function → Caller Relationships

| Function | Called By | Notes |
|----------|-----------|-------|
| `config.extract_region_mask()` | All Intensity scripts, LBP.py, featureextraction1.py | Most-called internal function |
| `config.extract_region()` | areawingtipdarkerthanwing*.py | Returns masked region + binary mask |
| `config.get_image_paths()` | LBP.py, all areawingtipdarkerthanwing*.py | Pairs images with segmentation masks |
| `clustering_latest.run_clustering_analysis()` | `__main__` block | Master pipeline for clustering |
| `LBP.extract_lbp_features()` | `analyze_species_texture()`, `visualize_region_lbp_details()` | Core LBP computation |
| `test_models.evaluate_model()` | `main()` | Runs inference + saves results |
| `GradCAM.generate_cam()` | `visualize_and_save_results()` | CNN interpretability |
| `AttentionRollout.get_attention_map()` | `visualize_and_save_results()` | ViT interpretability |
| `compute_gradcam_vit()` | `visualize_and_save_results()` | Enhanced/Interpretable ViT |
