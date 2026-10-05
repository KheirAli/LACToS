#!/bin/bash
# MAINTAINERS ONLY: pack weights / data / reconstructions into release tarballs (each < 2 GB,
# the GitHub-Releases per-file limit) and write SHA256SUMS. Upload the contents of $DEST as
# release assets (GitHub Release, Zenodo, or Hugging Face), then set BASE_URL in
# scripts/download_assets.sh.
#
#   SRC=/path/to/dir/with/{weights,data,recons} DEST=/big/disk/lactos_assets bash scripts/make_release_assets.sh
#
# PNG reconstructions are already compressed, so tarballs are uncompressed (.tar) except weights.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SRC:-$ROOT}"; DEST="${DEST:?set DEST to an output directory with ~10 GB free}"
mkdir -p "$DEST"; cd "$SRC"

tar czhf "$DEST/lactos_weights.tar.gz" weights/ae weights/backbones weights/torch
[ -d weights/diffusion ] && tar chf "$DEST/lactos_diffusion.tar" weights/diffusion   # only needed to re-sample
tar chf  "$DEST/lactos_data.tar" data

R=recons/results_patches_ddad_native
for c in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
  tar chf "$DEST/lactos_recons_$c.tar" ${R}_${c}_*
done
tar chf "$DEST/lactos_recons_CT.tar"             ${R}_CT_scan
tar chf "$DEST/lactos_recons_xray.tar"           ${R}_xray_scissors_1 ${R}_xray_scissors_b2_500k
tar chf "$DEST/lactos_recons_faces.tar"          ${R}_faces_random_1
tar chf "$DEST/lactos_recons_faces_moreface.tar" recons/results_patches_dps_guided_faces_random

cd "$DEST"
big=$(find . -maxdepth 1 -name 'lactos_*' -size +2000M)
[ -n "$big" ] && echo "WARNING: over 2 GB (split before uploading to GitHub Releases): $big"
sha256sum lactos_* > SHA256SUMS
ls -lh lactos_* SHA256SUMS
