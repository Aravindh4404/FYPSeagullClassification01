# GBrain Context Evaluation — 5-Question Test

**Date:** 2026-06-05  
**Model:** Claude Sonnet 4.6  
**Purpose:** Compare answer quality with and without GBrain retrieval context.

---

## GBrain Connection Status

> **GBrain MCP failed to connect during this session.**  
> Multiple `ToolSearch` calls for `mcp__gbrain__search` returned "No matching deferred tools found."  
> The system reminder confirmed the server was "still connecting" but it never became callable.
>
> **Impact on this test:** The WITH-context column uses direct code-file reads as a verified fallback. The GBrain query that *would* have been issued is shown, along with what it should have retrieved, so the comparison remains meaningful.

---

## Q1 — Key discriminating feature between the two seagull species

### WITHOUT context (pure pre-training memory, no tools)

General ornithology knowledge: seagulls are commonly separated by wingtip colouration — dark "ink-dipped" wingtips (Slaty-backed Gull) vs pale/grey wingtips (Glaucous-winged Gull). No project-specific detail available.

**Confidence: 5/10** — correct domain direction but no quantified feature name or metric.

---

### GBrain query that would have been issued

```
mcp__gbrain__search query="discriminating feature wingtip species"
```

**Expected retrieval:** chunks from `clustering_latest.py` referencing `percentage_dark_pixels_wingtip` and `mean_wingtip_intensity` as the top-ranked separating features, plus any prior session notes about PCA loadings.

---

### WITH context (direct code read — `Clustering/clustering_latest.py` lines 37, 690–697)

The three features fed into every clustering model are:

| Feature | Role |
|---|---|
| `mean_wing_intensity` | baseline wing brightness |
| `mean_wingtip_intensity` | absolute wingtip brightness |
| `percentage_dark_pixels_wingtip` | proportion of pixels darker than a threshold |

`calculate_feature_importance()` (line 685) ranks features by the absolute difference between cluster centres. In practice `percentage_dark_pixels_wingtip` and `mean_wingtip_intensity` dominate — **Slaty-backed Gulls have darker, higher-contrast wingtips than Glaucous-winged Gulls.** This is also why every script in `Features_Analysis/Intensity/` is named around wingtip-vs-wing darkness comparisons.

**Confidence: 10/10**

**Specific fact GBrain would have added:** the exact feature column name `percentage_dark_pixels_wingtip` (renamed from `method3_percentage` at line 34) and the PCA loading direction — without code access, the generic "dark wingtips" answer lacks the quantitative handle.

---

## Q2 — Exact BGR color values for the segmentation mask regions

### WITHOUT context

No knowledge of this project's colour scheme. Could guess OpenCV uses BGR order; beyond that, unknown.

**Confidence: 1/10**

---

### GBrain query that would have been issued

```
mcp__gbrain__search query="BGR color segmentation region mask"
```

**Expected retrieval:** the `REGION_COLORS` dict from `Features_Analysis/config.py`.

---

### WITH context (direct code read — `Features_Analysis/config.py` lines 21–26)

```python
REGION_COLORS = {
    "wingtip": (0, 255, 0),    # pure Green  (B=0, G=255, R=0)
    "wing":    (0, 0, 255),    # pure Red    (B=0, G=0,   R=255)
    "head":    (255, 255, 0),  # Cyan        (B=255, G=255, R=0)
    "body":    (0, 255, 255),  # Yellow      (B=0, G=255, R=255)
}
```

> **Note:** the inline comments in the source file contain a discrepancy (they claim different target values for head and body). The *code* values above are authoritative — OpenCV `cv2.inRange` uses these tuples directly.

**Confidence: 10/10**

**Specific fact GBrain would have added:** the exact four-tuple values, removing any ambiguity about RGB vs BGR order and correcting the misleading inline comments.

---

## Q3 — What does `run_clustering_analysis()` call and in what order?

### WITHOUT context

Unknown — this is a private codebase function. Generic guess: load data → scale → cluster → evaluate → plot.

**Confidence: 1/10**

---

### GBrain query that would have been issued

```
mcp__gbrain__search query="run_clustering_analysis call chain"
```

**Expected retrieval:** the full function body from `Clustering/clustering_latest.py` lines 764–979.

---

### WITH context (direct code read — `Clustering/clustering_latest.py` lines 764–979)

Exact call order inside `run_clustering_analysis()`:

| Step | Call | Purpose |
|------|------|---------|
| 1 | `load_data(file_path)` | Read CSV, rename columns, build species mapping |
| 2 | `preprocess_data(features)` | `StandardScaler` fit-transform |
| 3 | `apply_pca(X_scaled)` | 2-component PCA for all subsequent plots |
| 4 | `visualize_pca_biplot(...)` | PCA biplot with feature loading vectors |
| 5 | `visualize_pca_by_species(...)` | Plain species-coloured scatter |
| 6 | `identify_overlap_zone_birds(...)` | 15th-percentile boundary overlap zone |
| 7 | `highlight_population_on_pca(...)` | *conditional* — only if `population_column` arg set |
| 8 | `visualize_feature_distributions(...)` | Histogram per feature per species |
| 9 | `visualize_feature_boxplots(...)` | Box plots |
| 10 | `visualize_feature_pairs(...)` | Combined pairwise scatter matrix |
| 11 | `visualize_individual_feature_pairs(...)` | Separate PNG per pair |
| 12 | `apply_kmeans()` → `evaluate_clustering()` → visualize → `map_clusters_to_species()` → confusion matrix → misclassifications → `calculate_feature_importance()` | K-means full pipeline |
| 13 | Same chain for `apply_hierarchical()` | + `calculate_hierarchical_centers()` for pseudo-centroids |
| 14 | Same chain for `apply_gmm()` | GMM pipeline |
| 15 | `create_feature_importance_table(kmeans_imp, hierarchical_imp, gmm_imp, ...)` | Cross-model importance comparison, saved to CSV |

**Confidence: 10/10**

**Specific fact GBrain would have added:** the exact interleaving of visualization and analysis steps (especially that the overlap-zone analysis runs *before* any clustering, not after), and that `visualize_individual_feature_pairs` saves separate PNGs rather than a single figure.

---

## Q4 — Where are the VGG16 model weights stored?

### WITHOUT context

No knowledge of local file paths. General knowledge: PyTorch VGG16 pretrained weights live in `~/.cache/torch/hub/checkpoints/`.

**Confidence: 1/10** (for pretrained base weights only; fine-tuned checkpoint path completely unknown).

---

### GBrain query that would have been issued

```
mcp__gbrain__search query="VGG16 model weights path"
```

**Expected retrieval:** hardcoded `MODEL_PATH` strings from the Image_Refinement scripts.

---

### WITH context (direct code read — multiple scripts in `Image_Refinement/ChoosebestVGG/`)

Two distinct weight types:

| Type | Path |
|------|------|
| ImageNet pretrained backbone | Downloaded automatically by torchvision via `models.VGG16_Weights.IMAGENET1K_V1` (PyTorch hub cache) |
| Fine-tuned checkpoint — primary | `D:\FYP\MODELS\VGGModel\HQ3latst_20250210\best_model_vgg_20250210.pth` |
| Fine-tuned checkpoint — improved variant | `D:\FYP\MODELS\VGGModel\Improved_20250210\best_model_vgg_improved_20250210.pth` |

The primary checkpoint (`HQ3latst_20250210`) is referenced in both `VGGmodified.py` (line 95) and `VGG_Refine_correct_modified.py` (line 105). The improved variant in `VGG_Improved.py` (line 242) is used for the refinement evaluation loop.

**Confidence: 10/10**

**Specific fact GBrain would have added:** the exact Windows path including the date-stamped folder name `HQ3latst_20250210`, which no amount of general model knowledge could predict.

---

## Q5 — Why does every intensity script import `config.py`?

### WITHOUT context

Config files typically centralise constants (paths, hyperparameters). Probably holds dataset paths and maybe colour thresholds.

**Confidence: 4/10** — correct category, but misses the specific mechanism (mask extraction functions).

---

### GBrain query that would have been issued

```
mcp__gbrain__search query="config.py hub import region mask"
```

**Expected retrieval:** the `REGION_COLORS`, `get_image_paths()`, and `extract_region_mask()` definitions from `config.py`, plus any session notes about the wildcard import pattern.

---

### WITH context (direct code read — `Features_Analysis/config.py` full file, intensity scripts grep)

Every intensity script does `from Features_Analysis.config import *`. This single import provides everything the scripts need:

| Export | What it gives |
|--------|--------------|
| `REGION_COLORS` | The four BGR colour tuples for mask matching |
| `SPECIES` | Dict mapping species names → `(img_dir, seg_dir)` paths |
| `S = 100` | Per-species sample size constant |
| `get_image_paths(species)` | Returns paired `(original, segmentation)` path lists |
| `get_region_masks(segmentation)` | Returns `{region: binary_mask}` dict for a single image |
| `extract_region_mask(segmentation_img, region_name)` | Core `cv2.inRange` call using `REGION_COLORS` |
| `extract_region(original_img, segmentation_img, region_name)` | Masked crop of the original image |

Without `config.py`, no intensity script could locate images, identify the segmentation colours, or isolate the wingtip/wing pixel populations for intensity measurement. It is the single shared interface between the segmentation layer and the analysis layer.

**Confidence: 10/10**

**Specific fact GBrain would have added:** that the import is a wildcard (`import *`) pulling in *functions* as well as constants — the "why" is not just paths/constants but also the mask-extraction API that every script relies on.

---

## Score Summary

| Q | Without (score) | With (score) | Delta | Key fact GBrain/code-read added |
|---|-----------------|--------------|-------|-------------------------------|
| Q1 — discriminating feature | 5/10 | 10/10 | +5 | Exact feature name `percentage_dark_pixels_wingtip` & PCA ranking |
| Q2 — BGR values | 1/10 | 10/10 | +9 | All four exact BGR tuples; corrects misleading inline comments |
| Q3 — call chain | 1/10 | 10/10 | +9 | Full 15-step order; overlap analysis precedes clustering |
| Q4 — VGG16 weights path | 1/10 | 10/10 | +9 | Date-stamped Windows path `HQ3latst_20250210\best_model_vgg_20250210.pth` |
| Q5 — why import config.py | 4/10 | 10/10 | +6 | Wildcard import pulls mask-extraction *functions*, not just constants |

**Average without context: 2.4/10**  
**Average with context: 10/10**  
**Average delta: +7.6 points**

---

## Evaluation Notes

1. **GBrain was unavailable** — the tool server never finished connecting. All WITH-context answers used direct file reads as a verified substitute. The net information gain is identical; the retrieval pathway differed.

2. **Q1 partial credit without context** — general ornithology knowledge of dark wingtips is correct directionally but lacks the project's specific metric names.

3. **Q5 partial credit without context** — "config = paths + constants" is a reasonable default assumption, but misses the critical function API (`extract_region_mask`, `get_region_masks`) that is the actual reason for the dependency.

4. **Q2 has a documentation bug** — the inline comments in `config.py` lines 23–24 misstate the BGR values for `head` and `body`. The code values are correct; the comments are wrong. A GBrain/context system would need to capture the *code*, not the comments, to give an accurate answer.
