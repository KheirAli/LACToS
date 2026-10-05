#!/usr/bin/env python3
"""Print the reproduced numbers next to the reference values.

  python scripts/summarize_results.py            # reads ./results_eval_reproduction
  OUT=/other/dir python summarize_results.py

Section A: px ROC-AUC at eval sigma 5 from the evaluate.py JSONs (the REPRODUCE_RESULTS.md
metric: mean over defect subcategories, excluding good/combined).
Section B: the full metric table from score_metrics.sh (image-averaged and pooled modes),
compared with the reference main-table row (results_eval_main_sp30_full, Sep 2026).
"""
import glob, json, os, statistics as st
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_eval_reproduction"))
REF_AUC = {"bottle": .9771, "cable": .9569, "capsule": .9553, "carpet": .9259, "grid": .9126, "hazelnut": .9688,
           "leather": .9842, "metal_nut": .9580, "pill": .9193, "screw": .9231, "tile": .9007, "toothbrush": .9477,
           "transistor": .9432, "wood": .8912, "zipper": .9666, "CT": .9378, "xray_b1": .9528, "faces_random22": .9105}
CATS = [c for c in REF_AUC if c not in ("CT", "xray_b1", "faces_random22")]

def subcat_mean(summary):
    d = json.load(open(summary))
    rows = [r for r in d["rows"] if r["eval_sigma"] == "sigma_5.0" and r["subcategory"] not in ("good", "combined")]
    return st.mean(r["px_roc_auc"] for r in rows) if rows else None

def single(jdir):
    vals = []
    for f in glob.glob(os.path.join(jdir, "*.json")):
        d = json.load(open(f))
        if isinstance(d, dict) and d.get("averaged"):
            vals.append(d["averaged"]["sigma_5.0"]["px_roc_auc"])
        elif isinstance(d, dict) and "rows" in d:
            v = subcat_mean(f); vals.append(v) if v is not None else None
    return vals[0] if vals else None

print("A. px ROC-AUC (sigma 5, subcat mean)      reproduced   reference   diff")
got = {}
for c in CATS:
    s = glob.glob(f"{OUT}/json/mvtec/{c}/*summary.json")
    got[c] = subcat_mean(s[0]) if s else None
got["CT"] = single(f"{OUT}/json/CT/ct100"); got["xray_b1"] = single(f"{OUT}/json/xray/b1"); got["faces_random22"] = single(f"{OUT}/json/faces/random22")
for k, ref in REF_AUC.items():
    v = got.get(k)
    print(f"  {k:16s} {'  n/a  ' if v is None else f'{v:.4f} ':>12} {ref:10.4f} {'' if v is None else f'{v-ref:+.4f}':>9}")
mv = [got[c] for c in CATS if got[c] is not None]
if len(mv) == 15: print(f"  {'MVTec mean':16s} {st.mean(mv):12.4f} {0.9420:10.4f} {st.mean(mv)-0.9420:+9.4f}")

# ── B. full metric table ─────────────────────────────────────────────────────
R = f"{OUT}/metrics"
REF = {  # reference "Ours" rows, image-averaged: AUROC SNR AP MSE F1img F1pooled AUPRO  (main_all_both_modes.md)
    "MVTec (15 cats)": (94.13, 4.81, 49.25, 0.0491, 51.52, 46.11, 79.41),
    "CT (100)":        (93.77, 1.82, 55.38, 0.2375, 54.99, 50.28, 80.52),
    "faces (50)":      (84.09, 2.83, 29.83, 0.0665, 34.27, 28.15, 45.63),
    "xray (72)":       (94.22, 2.79, 11.64, 0.0253, 18.23, 10.37, 72.08)}
KI = ["per_image_auroc", "per_image_snr", "per_image_ap", "per_image_mask_mse", "per_image_f1max", "pooled_f1max", "aupro"]
KP = ["pooled_auroc", "pooled_snr", "pooled_aupr", "pooled_mask_mse", "per_image_f1max", "pooled_f1max", "aupro"]
def fmt(v): return " ".join(f"{x:6.4f}" if i == 3 else f"{x:6.2f}" if i == 1 else f"{100*x:6.2f}" for i, x in enumerate(v))  # i=1 SNR (ratio), i=3 MSE
def load(p): return json.load(open(p)) if os.path.exists(p) else None
def row(d, keys): return [d[k] for k in keys]
mv = [load(f"{R}/mvtec__{c}.json") for c in CATS]
tab = {"MVTec (15 cats)": ([st.mean(r) for r in zip(*[row(d, KI) for d in mv])] if all(mv) else None,
                           [st.mean(r) for r in zip(*[row(d, KP) for d in mv])] if all(mv) else None)}
for name, f in [("CT (100)", "CT__ct100"), ("faces (50)", "faces__all50"), ("xray (72)", "xray__full72")]:
    d = load(f"{R}/{f}.json"); tab[name] = (row(d, KI), row(d, KP)) if d else (None, None)
print("\nB. full metrics (AUROC SNR AP MSE F1img F1pooled AUPRO; % except SNR and MSE)")
print("   dataset            mode        " + "  AUROC    SNR     AP    MSE   F1img F1pool  AUPRO")
for name, (im, po) in tab.items():
    if im is None: print(f"   {name:18s} (not scored yet)"); continue
    print(f"   {name:18s} image-avg   {fmt(im)}\n   {'':18s} reference   {fmt([x if i in (1, 3) else x/100 for i, x in enumerate(REF[name])])}\n   {'':18s} pooled      {fmt(po)}")
