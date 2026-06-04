# Module Dependency Graph

## Mermaid Flowchart

```mermaid
flowchart TD
    CONFIG[Features_Analysis/config.py]
    UTILS[Features_Analysis/utils.py]
    IMAGE_NORM[Features_Analysis/image_normalization.py]

    LBP[Features_Analysis/LBP/LBP.py]
    FE1[Features_Analysis/LBP/featureextraction1.py]
    NGTEX[Features_Analysis/NG_Texture.py]
    NORM_COMP[Features_Analysis/normalization_comparison.py]
    NORM_IND[Features_Analysis/normalization_ind_image.py]
    CONFIGTEST[Features_Analysis/configtest.py]

    WING1[Intensity/areawingtipdarkerthanwing.py]
    WING2[Intensity/areawingtipdarkerthanwing2.py]
    WING3[Intensity/areawingdarkerthanwing3.py]
    WING4[Intensity/areawingtipdarkerthanwingwhitespotsremovedclusters.py]
    WING5[Intensity/areawingtipdarkerthanwingwhitespotsremovedignorehigherintensity.py]
    RDWPA[Intensity/Real_Dark_Wingtip_Pixel_Analysis.py]
    SWCR[Intensity/Save_Wingtip_Comparison_Results.py]
    SWR[Intensity/Save_Wing_Results.py]
    WDAN[Intensity/wingtipdarkeranalyser.py]
    PPTY[Intensity/ppty.py]
    DAPU1[Intensity/Data_Analysis/publishimage1.py]
    DAPU2[Intensity/Data_Analysis/Real_Dark_Wingtip_Pixel_Analysis.py]

    CLUST[Clustering/clustering_latest.py]
    COMPCLUST[Clustering/Compare_Clustering.py]
    CORRPCA[Clustering/correctpcadirection.py]
    INTERACT[Clustering/interactive.py]
    PROFSPEC[Clustering/profspecific.py]

    TEST_MOD[Test_Results/test_models.py]

    CONFIG --> LBP
    CONFIG --> FE1
    CONFIG --> NGTEX
    CONFIG --> WING1
    CONFIG --> WING2
    CONFIG --> WING3
    CONFIG --> WING4
    CONFIG --> WING5
    CONFIG --> RDWPA
    CONFIG --> SWCR
    CONFIG --> SWR
    CONFIG --> WDAN
    CONFIG --> PPTY
    CONFIG --> DAPU1
    CONFIG --> DAPU2
    CONFIG --> CONFIGTEST
    CONFIG --> NORM_COMP
    CONFIG --> NORM_IND
    UTILS --> NGTEX
    IMAGE_NORM --> NORM_COMP
    IMAGE_NORM --> NORM_IND
```

---

## Module Dependency Table

| Module | Imports From (internal) | Imports From (external key libs) |
|--------|------------------------|----------------------------------|
| `Features_Analysis/config.py` | — | `os`, `pathlib`, `numpy`, `cv2` |
| `Features_Analysis/utils.py` | — | various |
| `Features_Analysis/image_normalization.py` | `config` (DIRS) | `numpy`, `cv2`, `PIL` |
| `Features_Analysis/configtest.py` | `config` (DIRS, extract fns) | `cv2`, `matplotlib` |
| `Features_Analysis/NG_Texture.py` | `config.*`, `utils.*` | `cv2`, `numpy`, `skimage` |
| `Features_Analysis/normalization_comparison.py` | `config`, `image_normalization` | `numpy`, `matplotlib` |
| `Features_Analysis/normalization_ind_image.py` | `config`, `image_normalization` | `numpy`, `matplotlib` |
| `Features_Analysis/LBP/LBP.py` | `config.*` | `cv2`, `numpy`, `skimage`, `sklearn`, `pandas` |
| `Features_Analysis/LBP/featureextraction1.py` | `config.*` | `cv2`, `skimage`, `numpy` |
| `Features_Analysis/Intensity/areawingtipdarkerthanwing.py` | `config.*` | `cv2`, `numpy`, `pandas`, `matplotlib`, `seaborn` |
| `Features_Analysis/Intensity/areawingtipdarkerthanwing2.py` | `config.*` | same |
| `Features_Analysis/Intensity/areawingdarkerthanwing3.py` | `config.*` | same |
| `Features_Analysis/Intensity/areawingtipdarkerthanwingwhitespotsremovedclusters.py` | `config.*` | + `sklearn` |
| `Features_Analysis/Intensity/areawingtipdarkerthanwingwhitespotsremovedignorehigherintensity.py` | `config.*` | same |
| `Features_Analysis/Intensity/Real_Dark_Wingtip_Pixel_Analysis.py` | `config.*` | `cv2`, `numpy`, `pandas` |
| `Features_Analysis/Intensity/Save_Wing_Results.py` | `config.*` | `pandas`, `cv2` |
| `Features_Analysis/Intensity/Save_Wingtip_Comparison_Results.py` | `config.*` | `pandas`, `cv2` |
| `Features_Analysis/Intensity/wingtipdarkeranalyser.py` | `config.*` | `cv2`, `pandas`, `matplotlib` |
| `Features_Analysis/Intensity/ppty.py` | `config.*` | `cv2`, `numpy` |
| `Features_Analysis/Intensity/Wing_Intensity_Analysis.py` | — | `pandas`, `scipy`, `numpy` |
| `Features_Analysis/Intensity/Data_Analysis/publishimage1.py` | `config.*` | `cv2`, `numpy` |
| `Features_Analysis/Intensity/Data_Analysis/Real_Dark_Wingtip_Pixel_Analysis.py` | `config.*` | `cv2`, `numpy`, `pandas` |
| `Clustering/clustering_latest.py` | — | `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn` |
| `Clustering/Compare_Clustering.py` | — | same as above |
| `Clustering/correctpcadirection.py` | — | `numpy`, `matplotlib`, `pandas`, `sklearn`, `scipy` |
| `Clustering/interactive.py` | — | `pandas`, `numpy`, `matplotlib`, `sklearn` |
| `Clustering/profspecific.py` | — | `pandas`, `numpy`, `sklearn`, `scipy`, `seaborn` |
| `Test_Results/test_models.py` | — | `torch`, `torchvision`, `timm`, `cv2`, `sklearn`, `numpy`, `PIL` |
| `checkimage.py` | — | `PIL`, `imagehash` |
| `checksame.py` | — | `os`, `pandas`, `pathlib` |
| `visualize_normalization.py` | — | `matplotlib`, `numpy`, `cv2` |

---

## Key Observations

1. **`Features_Analysis/config.py` is the hub**: 16+ scripts import from it, making it the most critical file in the project. It defines all region colours, species directories, and mask extraction functions.

2. **The Clustering module is self-contained**: None of the `Clustering/` scripts import internal project modules — they only read CSVs generated by the Intensity analysis pipeline.

3. **The deep learning `Test_Results/test_models.py` is also self-contained**: All model architecture classes are defined inline; no imports from other project files.

4. **The LBP pipeline depends only on `config.py`**: `LBP/LBP.py` uses `config.py` for region colours and `extract_region_mask`.

5. **`normalization_comparison.py` and `normalization_ind_image.py` depend on two internal modules**: Both import from `config` and `image_normalization`, making them a small two-layer sub-dependency chain.

6. **No circular imports**: The dependency graph is strictly a DAG — `config.py` never imports from any other project module.

---

## Entry Points (Scripts with `if __name__ == "__main__"`)

| Script | What it does when run directly |
|--------|-------------------------------|
| `Clustering/clustering_latest.py` | Runs full clustering pipeline on `dark_pixel_results_all_images.csv` |
| `Features_Analysis/LBP/LBP.py` | Full LBP texture analysis + SVM classification |
| `Features_Analysis/Intensity/Wing_Intensity_Analysis.py` | Stats + t-test on wing intensity CSV |
| `Features_Analysis/Intensity/areawingtipdarkerthanwing*.py` | Wingtip darkness hypothesis test on raw images |
| `Test_Results/test_models.py` | Multi-model evaluation on test dataset |
| `checksame.py` | Dataset/CSV consistency validation |
| `checkimage.py` | Cross-folder image duplicate detection |
| `visualize_normalization.py` | ImageNet normalisation visualiser |
