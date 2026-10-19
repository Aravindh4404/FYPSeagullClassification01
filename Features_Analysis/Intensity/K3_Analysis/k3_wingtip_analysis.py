"""
k=3 vs k=2 (done twice) wingtip pixel clustering.

Prof. Gibbins' request (6 Jul 2026): instead of the two sequential k-means passes
  pass 1: k=2 on wingtip pixels  -> drop the white cluster
  pass 2: k=2 on what is left    -> dark vs intermediate
apply ONE k-means with k=3 to the whole wingtip -> white / intermediate / dark grey,
then compute stats for each subset.

Uses exactly the same preprocessing as Data_Analysis/Real_Dark_Wingtip_Pixel_Analysis.py
(grayscale -> min-max normalise to 0-255 -> wing / wingtip masks from the colour-coded
segmentation) and the same 180 images as Clustering/dark_pixel_results_all_images.csv.
Nothing existing is overwritten; all output goes to this folder.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))
from Features_Analysis.config import SPECIES, extract_region, get_image_paths  # noqa: E402

K2_CSV = ROOT / "Clustering" / "dark_pixel_results_all_images.csv"
OUT_DIR = Path(__file__).resolve().parent
OVERLAY_DIR = OUT_DIR / "overlays"
RESULTS_CSV = OUT_DIR / "k3_results_all_images.csv"

SEED = 42
COL_DARK = np.array([30, 60, 200])       # blue
COL_MID = np.array([245, 150, 20])       # orange
COL_WHITE = np.array([250, 235, 90])     # yellow


def kmeans_1d(values, k):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(values.reshape(-1, 1))
    return km.labels_, km.cluster_centers_.flatten()


def k2_twice(wingtip_pixels):
    """Existing pipeline. Returns per-pixel class: 0 dark, 1 intermediate, 2 white(removed)."""
    lab1, c1 = kmeans_1d(wingtip_pixels, 2)
    keep = lab1 == np.argmin(c1)
    cls = np.full(len(wingtip_pixels), 2, dtype=np.uint8)
    kept_vals = wingtip_pixels[keep]
    lab2, c2 = kmeans_1d(kept_vals, 2)
    sub = np.where(lab2 == np.argmin(c2), 0, 1).astype(np.uint8)
    cls[keep] = sub
    return cls, {"c1": np.sort(c1), "c2": np.sort(c2), "kept_mean": float(kept_vals.mean())}


def k3_once(wingtip_pixels):
    """Single k=3. Per-pixel class: 0 dark, 1 intermediate, 2 white (ordered by centre)."""
    lab, c = kmeans_1d(wingtip_pixels, 3)
    order = np.argsort(c)                    # ascending: dark, mid, white
    remap = np.empty(3, dtype=np.uint8)
    remap[order] = np.arange(3)
    return remap[lab], c[order]


def overlay_rgb(rgb, mask2d, cls, bbox):
    """Paint classes onto the wingtip pixels of an RGB image copy, crop to bbox."""
    out = rgb.copy()
    palette = np.stack([COL_DARK, COL_MID, COL_WHITE]).astype(np.uint8)
    blended = (0.35 * out[mask2d] + 0.65 * palette[cls]).astype(np.uint8)
    out[mask2d] = blended
    y0, y1, x0, x1 = bbox
    return out[y0:y1, x0:x1]


def process(img_path, seg_path, species, name):
    img = cv2.imread(img_path)
    seg = cv2.imread(seg_path)
    if img is None or seg is None:
        return None, None
    gray = cv2.normalize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, 0, 255, cv2.NORM_MINMAX)
    wing_reg, wing_mask = extract_region(gray, seg, "wing")
    tip_reg, tip_mask = extract_region(gray, seg, "wingtip")
    wing_px = wing_reg[wing_mask > 0]
    tip_px = tip_reg[tip_mask > 0]
    if len(wing_px) == 0 or len(tip_px) == 0:
        return None, None

    cls2, info2 = k2_twice(tip_px)
    cls3, centers3 = k3_once(tip_px)

    wing_mean = float(wing_px.mean())
    row = {"species": species, "image_name": name, "n_wingtip_pixels": len(tip_px),
           "mean_wing_intensity": wing_mean}
    for tag, cls in (("k3", cls3), ("k2x2", cls2)):
        for ci, cname in enumerate(("dark", "mid", "white")):
            v = tip_px[cls == ci]
            row[f"{tag}_{cname}_n"] = int(len(v))
            row[f"{tag}_{cname}_pct"] = 100.0 * len(v) / len(tip_px)
            row[f"{tag}_{cname}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{tag}_{cname}_std"] = float(v.std()) if len(v) else np.nan
    row["k3_centres"] = ";".join(f"{c:.1f}" for c in centers3)

    # bbox + overlay panels
    ys, xs = np.nonzero(tip_mask)
    pad = 25
    H, W = gray.shape
    bbox = (max(ys.min() - pad, 0), min(ys.max() + pad, H), max(xs.min() - pad, 0), min(xs.max() + pad, W))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    m = tip_mask > 0
    panels = {
        "orig": rgb[bbox[0]:bbox[1], bbox[2]:bbox[3]],
        "k2": overlay_rgb(rgb, m, cls2, bbox),
        "k3": overlay_rgb(rgb, m, cls3, bbox),
    }
    return row, {"panels": panels, "c3": centers3, "c2": info2, "row": row}


def save_overlay(species, name, d):
    r = d["row"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    ax[0].imshow(d["panels"]["orig"]); ax[0].set_title("Original (wingtip crop)")
    ax[1].imshow(d["panels"]["k2"])
    ax[1].set_title("k=2 applied twice (current)\n"
                    f"dark {r['k2x2_dark_pct']:.0f}% (mean {r['k2x2_dark_mean']:.0f}) | "
                    f"mid {r['k2x2_mid_pct']:.0f}% ({r['k2x2_mid_mean']:.0f}) | "
                    f"white {r['k2x2_white_pct']:.0f}% ({r['k2x2_white_mean']:.0f})", fontsize=9)
    ax[2].imshow(d["panels"]["k3"])
    ax[2].set_title("k=3 single pass (proposed)\n"
                    f"dark {r['k3_dark_pct']:.0f}% (mean {r['k3_dark_mean']:.0f}) | "
                    f"mid {r['k3_mid_pct']:.0f}% ({r['k3_mid_mean']:.0f}) | "
                    f"white {r['k3_white_pct']:.0f}% ({r['k3_white_mean']:.0f})", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.legend(handles=[Patch(color=COL_DARK / 255, label="dark grey"),
                        Patch(color=COL_MID / 255, label="intermediate"),
                        Patch(color=COL_WHITE / 255, label="white")],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{species.replace('_', ' ')} - {name}   (upperpart mean = {r['mean_wing_intensity']:.0f})",
                 fontweight="bold")
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = OVERLAY_DIR / species
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{Path(name).stem}.png", dpi=110)
    plt.close(fig)


def main():
    OVERLAY_DIR.mkdir(exist_ok=True)
    ref = pd.read_csv(K2_CSV)
    wanted = {(r.species, r.image_name) for r in ref.itertuples()}
    rows = []
    for species in SPECIES:
        for img_path, seg_path in get_image_paths(species):
            name = os.path.basename(img_path)
            if (species, name) not in wanted:
                continue
            row, d = process(img_path, seg_path, species, name)
            if row is None:
                print("SKIP", species, name)
                continue
            rows.append(row)
            save_overlay(species, name, d)
            print(f"{len(rows):3d}/{len(wanted)} {species} {name}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print("saved", RESULTS_CSV, len(df))


if __name__ == "__main__":
    main()
