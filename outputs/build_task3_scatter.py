"""
Task 3: upperpart (mean_wing_intensity) vs wingtip/primaries (mean_wingtip_intensity)
scatterplot, one point per bird, colored by species, with a fitted regression line
and the 1:1 reference line.

Data source: Clustering/dark_pixel_results_all_images.csv (180 rows, 90 GWG / 90 SBG).
Confirmed byte-identical to Features_Analysis/Intensity/Data_Analysis/
Wingtip_Dark_Pixel_Visualizations/dark_pixel_results_all_images.csv via CRC/diff,
and matches the n=180 (90/90) sample size stated in the methods section.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATH = r"D:\FYPSeagullClassification01\Clustering\dark_pixel_results_all_images.csv"
OUT_PATH = r"D:\FYPSeagullClassification01\outputs\task3_upperpart_vs_wingtip_scatter.png"

SPECIES_COLORS = {
    "Glaucous_Winged_Gull": "#3274A1",
    "Slaty_Backed_Gull": "#E1812C",
}
SPECIES_LABELS = {
    "Glaucous_Winged_Gull": "Glaucous-winged Gull",
    "Slaty_Backed_Gull": "Slaty-backed Gull",
}

df = pd.read_csv(CSV_PATH)
x = df["mean_wing_intensity"].values       # upperpart
y = df["mean_wingtip_intensity"].values    # wingtip / primaries

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

fig, ax = plt.subplots(figsize=(8, 8))

for species, color in SPECIES_COLORS.items():
    mask = df["species"] == species
    ax.scatter(df.loc[mask, "mean_wing_intensity"], df.loc[mask, "mean_wingtip_intensity"],
               label=f"{SPECIES_LABELS[species]} (n={mask.sum()})",
               color=color, alpha=0.7, s=50, edgecolors="black", linewidth=0.4)

lims = [min(x.min(), y.min()) - 5, max(x.max(), y.max()) + 5]

# 1:1 reference line
ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1.5, label="1:1 reference")

# Fitted regression line
xs = np.linspace(lims[0], lims[1], 100)
ax.plot(xs, slope * xs + intercept, color="black", linewidth=2,
         label=f"Fit: y = {slope:.2f}x + {intercept:.1f} (R²={r_value**2:.2f})")

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.set_xlabel("Mean upperpart (wing) grey intensity", fontsize=12)
ax.set_ylabel("Mean wingtip (primaries) grey intensity", fontsize=12)
ax.set_title("Upperpart vs. Wingtip Intensity per Bird (n=180)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
print(f"n={len(df)}, slope={slope:.4f}, intercept={intercept:.4f}, R2={r_value**2:.4f}, p={p_value:.2e}")
