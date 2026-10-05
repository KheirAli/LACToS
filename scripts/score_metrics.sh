#!/bin/bash
# Score the delta-map dumps written by run_reproduce.sh with the unified metric code
# (tools/metrics/full_metric_table.py): pooled AUROC/AUPR/F1max/SNR/MSE, AUPRO (FPR<=0.3),
# per-image AUROC/AP/F1max/SNR/mask-MSE/PSNR, all at eval sigma 5.
set -uo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PKG"
OUT="${OUT:-$PKG/results_eval_reproduction}"; M="$OUT/maps"; R="$OUT/metrics"; mkdir -p "$R"
S="python tools/metrics/full_metric_table.py"
for c in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
  [ -d "$M/mvtec/$c" ] && $S --maps_dir "$M/mvtec/$c" --out "$R/mvtec__$c.json" > /dev/null && echo "scored mvtec/$c"
done
[ -d "$M/CT/ct100" ] && $S --maps_dir "$M/CT/ct100" --out "$R/CT__ct100.json" > /dev/null && echo "scored CT"
for t in b1 b2; do [ -d "$M/xray/$t" ] && $S --maps_dir "$M/xray/$t" --out "$R/xray__$t.json" > /dev/null && echo "scored xray/$t"; done
for t in random22 moreface28; do [ -d "$M/faces/$t" ] && $S --maps_dir "$M/faces/$t" --out "$R/faces__$t.json" > /dev/null && echo "scored faces/$t"; done
# pooled full sets = union of the npz dumps (symlinks), scored as one dataset
pool() { local name="$1"; shift; local d="$M/$name"; rm -rf "$d"; mkdir -p "$d"; for src in "$@"; do [ -d "$src" ] && for f in "$src"/*.npz; do ln -sf "$f" "$d/$(basename "$(dirname "$f")")__$(basename "$f")"; done; done; }
if [ -d "$M/xray/b1" ] && [ -d "$M/xray/b2" ]; then pool xray/full72 "$M/xray/b1" "$M/xray/b2"; $S --maps_dir "$M/xray/full72" --out "$R/xray__full72.json" > /dev/null && echo "scored xray/full72 (72 patients)"; fi
if [ -d "$M/faces/random22" ] && [ -d "$M/faces/moreface28" ]; then pool faces/all50 "$M/faces/random22" "$M/faces/moreface28"; $S --maps_dir "$M/faces/all50" --out "$R/faces__all50.json" > /dev/null && echo "scored faces/all50"; fi
echo "metrics -> $R ; now: python scripts/summarize_results.py"
