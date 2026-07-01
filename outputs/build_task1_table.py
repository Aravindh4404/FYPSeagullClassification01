"""
Task 1: comparison table (accuracy, precision, recall, F1) computed directly from the
confusion matrices already saved in Test_Results/Test_Results/<model>/metrics/classification_metrics.txt
No re-inference. Source of truth restricted to Test_Results/Test_Results only.
"""
import csv

# (model, result_folder, confusion matrix as [[TN_GWG_row...]], class order = [GWG, SBG])
MODELS = [
    ("VGG-16", "best_model_vgg_20250210", [[42, 1], [0, 43]]),
    ("Inception v3", "best_model_inception", [[22, 21], [4, 39]]),
    ("ViT-Base/16", "final_model_vit_20241224", [[42, 1], [4, 39]]),
    ("ViT+Attention", "final_model_vit_20250213", [[42, 1], [2, 41]]),
]

rows = []
for name, folder, cm in MODELS:
    (tp_g, fn_g), (fp_g_as_fn_s, tp_s) = cm
    # cm = [[GWG->GWG, GWG->SBG], [SBG->GWG, SBG->SBG]]
    gwg_tp, gwg_fn = cm[0]
    sbg_fn, sbg_tp = cm[1]
    gwg_fp = sbg_fn   # predicted GWG but true SBG
    sbg_fp = gwg_fn   # predicted SBG but true GWG

    total = sum(sum(r) for r in cm)
    correct = cm[0][0] + cm[1][1]
    accuracy = correct / total

    def prec_rec_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    p_g, r_g, f1_g = prec_rec_f1(gwg_tp, gwg_fp, gwg_fn)
    p_s, r_s, f1_s = prec_rec_f1(sbg_tp, sbg_fp, sbg_fn)

    macro_p = (p_g + p_s) / 2
    macro_r = (r_g + r_s) / 2
    macro_f1 = (f1_g + f1_s) / 2

    rows.append({
        "model": name,
        "result_folder": folder,
        "n_test_images": total,
        "test_accuracy": round(accuracy, 4),
        "validation_accuracy": "N/A (not present in Test_Results/Test_Results; only test-set confusion matrices exist there)",
        "precision_macro": round(macro_p, 4),
        "recall_macro": round(macro_r, 4),
        "f1_macro": round(macro_f1, 4),
        "precision_GWG": round(p_g, 4),
        "recall_GWG": round(r_g, 4),
        "f1_GWG": round(f1_g, 4),
        "precision_SBG": round(p_s, 4),
        "recall_SBG": round(r_s, 4),
        "f1_SBG": round(f1_s, 4),
    })

out_path = r"D:\FYPSeagullClassification01\outputs\task1_model_comparison_table.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {out_path}")
for r in rows:
    print(r["model"], "-> acc:", r["test_accuracy"], "macroP:", r["precision_macro"],
          "macroR:", r["recall_macro"], "macroF1:", r["f1_macro"])
