"""
Task 1: side-by-side GradCAM/attention comparison figure across the four models,
built entirely from existing artifacts in Test_Results/Test_Results (no re-inference).
"""
import os
from PIL import Image
import matplotlib.pyplot as plt

BASE = r"D:\FYPSeagullClassification01\Test_Results"
R = os.path.join(BASE, "Test_Results")
DATA = os.path.join(BASE, "Test_Data")
OUT = r"D:\FYPSeagullClassification01\outputs"

# (filename, species_folder, short_note)
BIRDS = [
    ("0Z2A2122.JPG", "Glaucous_Winged_Gull", "Inception misclassifies; others correct"),
    ("mod-0R8A5109.JPG", "Slaty_Backed_Gull", "ViT-Base/16 misclassifies; others correct"),
    ("mod-0R8A5217.JPG", "Slaty_Backed_Gull", "Both ViT variants misclassify; VGG & Inception correct"),
    ("0Z2A0606.JPG", "Glaucous_Winged_Gull", "All four models correct"),
]

COLS = ["Original", "VGG-16", "Inception v3", "ViT-Base/16", "ViT+Attention"]


def crop_third_panel(path):
    """The ViT scripts save a 3-panel matplotlib figure (original | map | overlay).
    Crop out the rightmost third (the overlay panel)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    third = w // 3
    return im.crop((2 * third, 0, w, h))


fig, axes = plt.subplots(len(BIRDS), len(COLS), figsize=(4 * len(COLS), 4 * len(BIRDS)))

for row, (fname, species, note) in enumerate(BIRDS):
    stem = os.path.splitext(fname)[0]

    orig = Image.open(os.path.join(DATA, species, fname)).convert("RGB")
    vgg = Image.open(os.path.join(R, "best_model_vgg_20250210", "gradcam_all", fname)).convert("RGB")
    inc = Image.open(os.path.join(R, "best_model_inception", "gradcam_all", fname)).convert("RGB")
    vit_base = crop_third_panel(os.path.join(R, "final_model_vit_20241224", "attention_all", f"{species}_{fname}"))
    vit_attn = crop_third_panel(os.path.join(R, "final_model_vit_20250213", "gradcam_all", f"{species}_{stem}.png"))

    imgs = [orig, vgg, inc, vit_base, vit_attn]
    for col, (img, title) in enumerate(zip(imgs, COLS)):
        ax = axes[row, col]
        ax.imshow(img)
        ax.axis("off")
        if row == 0:
            ax.set_title(title, fontsize=14, fontweight="bold")
        if col == 0:
            ax.text(-0.15, 0.5, f"{species.replace('_', ' ')}\n{stem}\n({note})",
                     transform=ax.transAxes, fontsize=9, va="center", ha="right", wrap=True)

plt.tight_layout()
out_path = os.path.join(OUT, "task1_gradcam_comparison.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
