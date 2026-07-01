# Task 1: Fair Four-Model Comparison — Summary

**Source of truth:** `Test_Results/Test_Results/<model>/metrics/classification_metrics.txt`, all four
evaluated on the identical current test set (`Test_Results/Test_Data`, 86 images: 43 Glaucous-winged
Gull / 43 Slaty-backed Gull). No re-inference was run — all numbers and GradCAM/attention images
already existed from a single consistent 2025-08-20 batch. See `task1_model_comparison_table.csv`
and `task1_gradcam_comparison.png`.

## Corrected model identity (important)

Your two ViT labels were swapped in the draft. Verified directly from the model class definitions
and checkpoint files (`Test_Results/architectures/test_vitoriginal.py` and
`test_interpretable_vit.py`):

- **ViT-Base/16 (plain)** = `ViTModified` class (linear head only, no attention layer) →
  checkpoint `final_model_vit_20241224` → **94.19%** accuracy.
- **ViT+Attention** = `InterpretableViTGradCAM` class (attention-weighted patch pooling + concatenated
  cls/patch tokens + deeper classifier) → checkpoint `final_model_vit_20250213` → **96.51%** accuracy.

This is the reverse of the previously stated 96.51%/94.19% split. Also confirmed by CRC checksum:
the checkpoint duplicated under the `EnhancedViT_20250213` folder is byte-identical to the
`InterpretableViT_20250213` one — only one attention-variant checkpoint was ever evaluated on the
current test set; `EnhancedViT` (a structurally different, simpler attention head) has its own
separate, later checkpoint (`VIT2_HQ3_20250208`) that was never run against `Test_Data`.

## Comparison table

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| VGG-16 | 98.84% | 98.86% | 98.84% | 98.84% |
| Inception v3 | 70.93% | 74.81% | 70.93% | 69.75% |
| ViT-Base/16 | 94.19% | 94.40% | 94.19% | 94.18% |
| ViT+Attention | 96.51% | 96.54% | 96.51% | 96.51% |

Validation accuracy is **not available** from `Test_Results/Test_Results` — only test-set confusion
matrices are stored there. Recovering validation accuracy would require pulling epoch-level metrics
from the original Colab training notebooks, which I have not done (flagging so it isn't silently
left out of the paper without explanation).

## GradCAM region comparison (see `task1_gradcam_comparison.png`)

Four birds, shown across all four models:

1. **Inception-specific failure** (`0Z2A1082`, GWG, correct except Inception): VGG and Inception both
   spread heat diffusely across both wings and the body; Inception's attention doesn't concentrate
   sharply on any single discriminating region, and it flips to Slaty-backed Gull at only 60%
   confidence. Both ViT variants place a tighter, more localized peak on the wingtip/covert boundary
   and classify correctly.
2. **CNN-specific failure** (`49995412233`, GWG, VGG & Inception wrong, both ViT variants correct):
   with birds and background clutter in frame, VGG and Inception's heat drifts onto the body/background
   transition rather than the wingtip, driving a confident wrong answer. Both ViT models keep a tight
   attention spot on the wing region and get it right — evidence that ViT's patch-token attention is
   more robust to background clutter than CNN conv-feature GradCAM here.
3. **ViT-specific failure** (`mod-0R8A5217`, SBG, both ViT variants wrong, VGG & Inception correct):
   here it reverses — VGG and Inception concentrate heat on the head/neck, and classify correctly;
   both ViT variants' attention drifts toward the head and slightly off-bird, landing on
   Glaucous-winged Gull incorrectly. This is the clearest case where head/bill-region reliance from
   the CNNs actually helped, while the ViTs' typically wingtip-focused attention misfired.
4. **Consensus case** (`0Z2A0606`, GWG, all four correct): all four models place at least part of
   their heat on the wingtip/outer-wing region, though VGG and Inception's heat maps are visibly more
   diffuse (spread across the whole wing and part of the body) compared to the tighter, more localized
   ViT attention peaks.

**Overall pattern:** the two strongest models (VGG-16, ViT+Attention) most consistently localize to
the wingtip/outer-primaries region described in your pipeline as the discriminating feature. Inception
v3's heat is the most diffuse and least localized of the four, consistent with its far lower accuracy
and its strong bias toward misclassifying Glaucous-winged Gulls as Slaty-backed (21 of its 25 errors,
confirmed from its own `misclassified_images.txt`). ViT-Base/16 (plain) sits in between — generally
wingtip-focused but occasionally drawn toward the head, corresponding to its higher error rate than
the attention variant.
