#!/bin/bash
# Download + verify + extract the release assets into the repository root.
#
#   bash scripts/download_assets.sh                  # everything (~9.5 GB)
#   bash scripts/download_assets.sh weights data CT  # only what you need
#
# Groups: weights data  bottle cable capsule carpet grid hazelnut leather metal_nut pill screw
#         tile toothbrush transistor wood zipper  CT xray faces faces_moreface
# `weights` and `data` are needed for every target.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
BASE_URL="${BASE_URL:-https://REPLACE-ME/lactos-assets}"   # TODO(maintainers): set after uploading the release
DL="${DL:-$ROOT/.assets_cache}"; mkdir -p "$DL"

ALL=(weights data bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper CT xray faces faces_moreface)
GROUPS_=("$@"); [ "${#GROUPS_[@]}" -eq 0 ] && GROUPS_=("${ALL[@]}")

fetch() { local f="$1"; [ -f "$DL/$f" ] || curl -fL --retry 3 -o "$DL/$f" "$BASE_URL/$f"; }
fetch SHA256SUMS
for g in "${GROUPS_[@]}"; do
  case "$g" in weights) f=lactos_weights.tar.gz ;; data) f=lactos_data.tar ;; *) f=lactos_recons_$g.tar ;; esac
  echo "== $g ($f)"; fetch "$f"
  ( cd "$DL" && grep " $f\$" SHA256SUMS | sha256sum -c - )
  tar xf "$DL/$f" -C "$ROOT"
done
echo "Assets ready. Verify with: bash scripts/check_env.sh"
