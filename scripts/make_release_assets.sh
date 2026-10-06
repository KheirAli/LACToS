#!/bin/bash
# MAINTAINERS ONLY: pack weights / data / reconstructions into release zip files (each < 2 GB)
# and write SHA256SUMS. Upload the contents of $DEST (Google Drive, GitHub Release, Zenodo, ...),
# then set BASE_URL in scripts/download_assets.sh.
#
#   SRC=/path/to/dir/with/{weights,data,recons} DEST=/big/disk/lactos_assets bash scripts/make_release_assets.sh
#
# PNGs are already compressed, so data / reconstruction zips are stored without recompression (-0).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SRC:-$ROOT}"; DEST="${DEST:?set DEST to an output directory with ~10 GB free}"
mkdir -p "$DEST"; DEST="$(cd "$DEST" && pwd)"; cd "$SRC"
# write NAME.partial.zip first, then rename (no half-written zip is left behind on failure)
Z() { local name="$1" opt="$2"; shift 2; rm -f "$DEST/$name"; echo "  $name"
      zip -q -r $opt "$DEST/${name%.zip}.partial.zip" "$@"; mv "$DEST/${name%.zip}.partial.zip" "$DEST/$name"; }

Z lactos_weights.zip   ""  weights/ae weights/backbones weights/torch
[ -d weights/diffusion ] && Z lactos_diffusion.zip "" weights/diffusion   # only needed to re-sample
Z lactos_data.zip      -0  data

R=recons/results_patches_ddad_native
for c in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
  Z lactos_recons_$c.zip -0 ${R}_${c}_*
done
Z lactos_recons_CT.zip             -0 ${R}_CT_scan
Z lactos_recons_xray.zip           -0 ${R}_xray_scissors_1 ${R}_xray_scissors_b2_500k
Z lactos_recons_faces.zip          -0 ${R}_faces_random_1
Z lactos_recons_faces_moreface.zip -0 recons/results_patches_dps_guided_faces_random

cd "$DEST"
big=$(find . -maxdepth 1 -name 'lactos_*.zip' -size +2000M)
[ -n "$big" ] && echo "WARNING: over 2 GB (split before uploading to GitHub Releases): $big"
sha256sum lactos_*.zip > SHA256SUMS
ls -lh lactos_*.zip SHA256SUMS
