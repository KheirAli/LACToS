#!/bin/bash
# Download + verify (SHA-256) + extract the release assets into the repository root.
#
#   bash scripts/download_assets.sh weights            # 18 autoencoders + ResNet-101 backbones (Step 3)
#   bash scripts/download_assets.sh diffusion          # diffusion priors: CT, X-ray, FFHQ (Step 1, sampling only)
#   bash scripts/download_assets.sh weights data CT    # several groups
#   bash scripts/download_assets.sh                    # everything that is available
#
# Groups: weights diffusion data  bottle cable capsule carpet grid hazelnut leather metal_nut pill screw
#         tile toothbrush transistor wood zipper  CT xray faces faces_moreface
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
DL="${DL:-$ROOT/.assets_cache}"; mkdir -p "$DL"

# Google Drive files: group -> "<file id> <file name> <sha256>"
declare -A GDRIVE=(
  [weights]="1N6PpLGLLnyDuDG8ecb0trPyS_rbsLM-x lactos_weights.tar.gz 54c0e293f625b9e3ff1dd1ecc9606210b44a65b79dcf9d1a9e8230cf2e7f090d"
  [diffusion]="10jF91HOyhw6VAEEBWKM7GoLMwdcBisaB lactos_diffusion.tar 25d8b0e9aae115081497812a18c049bd4ada93b9984cf2e8251a3bee1e55c2ce"
)
# Not uploaded yet: data and reconstructions (lactos_data.zip, lactos_recons_<target>.zip). Set BASE_URL to a
# folder that serves them together with SHA256SUMS, or add their Google Drive ids to GDRIVE above.
BASE_URL="${BASE_URL:-}"

ALL=(weights diffusion data bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper CT xray faces faces_moreface)
GROUPS_=("$@"); [ "${#GROUPS_[@]}" -eq 0 ] && GROUPS_=("${ALL[@]}")

gdrive_get() {  # id out
  curl -fL --retry 3 -o "$2" "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"
}
extract() {     # file
  case "$1" in
    *.tar.gz|*.tgz) tar xzf "$1" -C "$ROOT" ;;
    *.tar)          tar xf  "$1" -C "$ROOT" ;;
    *.zip)          unzip -q -o "$1" -d "$ROOT" ;;
  esac
}

for g in "${GROUPS_[@]}"; do
  if [ -n "${GDRIVE[$g]:-}" ]; then
    read -r id f sha <<< "${GDRIVE[$g]}"
    echo "== $g ($f, Google Drive)"
    [ -f "$DL/$f" ] || gdrive_get "$id" "$DL/$f"
    echo "$sha  $DL/$f" | sha256sum -c --quiet - || { echo "checksum mismatch for $f; delete $DL/$f and retry"; exit 1; }
  elif [ -n "$BASE_URL" ]; then
    case "$g" in data) f=lactos_data.zip ;; *) f=lactos_recons_$g.zip ;; esac
    echo "== $g ($f)"
    [ -f "$DL/SHA256SUMS" ] || curl -fL --retry 3 -o "$DL/SHA256SUMS" "$BASE_URL/SHA256SUMS"
    [ -f "$DL/$f" ] || curl -fL --retry 3 -o "$DL/$f" "$BASE_URL/$f"
    ( cd "$DL" && grep " $f\$" SHA256SUMS | sha256sum -c --quiet - )
  else
    echo "== $g: not available for download yet (see docs/DATA.md); skipping"; continue
  fi
  extract "$DL/$f"
done
echo "Done. Verify with: bash scripts/check_env.sh"
