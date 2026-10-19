"""
Compare k=2 (applied twice) with single-pass k=3 using k3_results_all_images.csv.

1. Scatter: upperpart mean (x) vs wingtip measure (y) with 1:1 line and fitted line;
   counts birds ABOVE the line (wingtip lighter than upperparts = biologically odd).
2. Does species separation (unsupervised K-means / Hierarchical / GMM, k=2 species)
   change when the k=3 features replace the k=2x2 features?
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
k3 = pd.read_csv(HERE / "k3_results_all_images.csv")
old = pd.read_csv(ROOT / "Clustering" / "dark_pixel_results_all_images.csv")
df = k3.merge(old[["species", "image_name", "mean_wingtip_intensity", "method3_percentage"]],
              on=["species", "image_name"])
assert len(df) == len(k3) == 180

COL = {"Glaucous_Winged_Gull": "#3274A1", "Slaty_Backed_Gull": "#E1812C"}
LAB = {"Glaucous_Winged_Gull": "Glaucous-winged", "Slaty_Backed_Gull": "Slaty-backed"}

measures = [
    ("Current: mean of wingtip after white removal\n(mean_wingtip_intensity)", "mean_wingtip_intensity"),
    ("Current: dark cluster of 2nd k=2 pass", "k2x2_dark_mean"),
    ("Proposed: k=3 dark cluster", "k3_dark_mean"),
    ("Proposed: k=3 dark + intermediate pooled", None),
]
# pooled dark+mid mean for k=3
df["k3_darkmid_mean"] = ((df.k3_dark_n * df.k3_dark_mean + df.k3_mid_n * df.k3_mid_mean)
                         / (df.k3_dark_n + df.k3_mid_n))
measures[3] = (measures[3][0], "k3_darkmid_mean")

x = df.mean_wing_intensity.values
fig, axes = plt.subplots(1, 4, figsize=(24, 6.4))
summary = []
for ax, (title, col) in zip(axes, measures):
    y = df[col].values
    slope, icpt, r, p, _ = stats.linregress(x, y)
    above11 = y > x
    abovefit = y > slope * x + icpt
    for sp, c in COL.items():
        m = (df.species == sp).values
        ax.scatter(x[m & ~above11], y[m & ~above11], color=c, alpha=.7, s=36, edgecolor="k", lw=.3,
                   label=f"{LAB[sp]}")
        ax.scatter(x[m & above11], y[m & above11], color=c, alpha=.9, s=60, edgecolor="red", lw=1.6, marker="D")
    lim = [0, max(x.max(), y.max()) + 5]
    ax.plot(lim, lim, "--", color="gray", label="1:1 (wingtip = upperpart)")
    xs = np.array(lim)
    ax.plot(xs, slope * xs + icpt, "k", lw=1.6, label=f"fit R²={r ** 2:.2f}")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Mean upperpart intensity"); ax.set_ylabel("Wingtip intensity")
    ax.set_title(f"{title}\nabove 1:1 line (red diamonds): {above11.sum()}/180", fontsize=10)
    ax.grid(alpha=.3)
    summary.append({"measure": col, "above_1to1": int(above11.sum()),
                    "above_1to1_SBG": int((above11 & (df.species == "Slaty_Backed_Gull").values).sum()),
                    "above_1to1_GWG": int((above11 & (df.species == "Glaucous_Winged_Gull").values).sum()),
                    "above_fit": int(abovefit.sum()), "slope": slope, "intercept": icpt, "R2": r ** 2})
axes[0].legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(HERE / "scatter_k2_vs_k3.png", dpi=200)
plt.close()
pd.DataFrame(summary).to_csv(HERE / "above_line_counts.csv", index=False)
print(pd.DataFrame(summary).round(3).to_string(index=False))

# ---- species separation -------------------------------------------------------------------
true = (df.species == "Glaucous_Winged_Gull").astype(int).values


def acc_after_mapping(labels):
    a = (labels == true).mean()
    return max(a, 1 - a)


def evaluate(X, name):
    Xs = StandardScaler().fit_transform(X)
    out = []
    for algo, lab in (("KMeans", KMeans(2, random_state=42, n_init=10).fit_predict(Xs)),
                      ("Hierarchical", AgglomerativeClustering(2, linkage="ward").fit_predict(Xs)),
                      ("GMM", GaussianMixture(2, random_state=42).fit_predict(Xs))):
        out.append({"feature_set": name, "algorithm": algo,
                    "accuracy": acc_after_mapping(lab), "ARI": adjusted_rand_score(true, lab),
                    "silhouette": silhouette_score(Xs, lab)})
    return out


sets = {
    "k=2x2 (current): wing, wingtip, dark%": df[["mean_wing_intensity", "mean_wingtip_intensity", "method3_percentage"]],
    "k=3: wing, dark mean, dark%": df[["mean_wing_intensity", "k3_dark_mean", "k3_dark_pct"]],
    "k=3: wing, dark mean, mid mean, dark%, white%": df[["mean_wing_intensity", "k3_dark_mean", "k3_mid_mean",
                                                         "k3_dark_pct", "k3_white_pct"]],
}
res = []
for n, X in sets.items():
    res += evaluate(X.values, n)
res = pd.DataFrame(res)
res.to_csv(HERE / "species_separation_k2_vs_k3.csv", index=False)
print("\n", res.round(3).to_string(index=False))

# per-species means of the new stats
cols = [c for c in df.columns if c.startswith("k3_") and (c.endswith("_mean") or c.endswith("_pct"))]
df.groupby("species")[cols + ["mean_wing_intensity"]].mean().round(2).T.to_csv(HERE / "k3_species_means.csv")
print("\n", df.groupby("species")[cols].mean().round(1).T.to_string())
df.to_csv(HERE / "k3_merged_with_k2.csv", index=False)
