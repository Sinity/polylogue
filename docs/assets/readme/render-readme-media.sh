#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

render_gif() {
  local name="$1"
  local trim="$2"
  local input="docs/assets/readme/${name}.gif"
  local palette=".local/${name}-palette.png"
  local output=".local/${name}-trimmed.gif"

  vhs "docs/assets/readme/${name}.tape"
  ffmpeg -y -ss "$trim" -i "$input" \
    -vf "fps=12,scale=960:-1:flags=lanczos,palettegen=stats_mode=diff" \
    "$palette" >/dev/null 2>&1
  ffmpeg -y -ss "$trim" -i "$input" -i "$palette" \
    -lavfi "fps=12,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
    -loop 0 "$output" >/dev/null 2>&1
  mv "$output" "$input"
}

vhs validate \
  docs/assets/readme/polylogue-query.tape \
  docs/assets/readme/polylogue-products.tape \
  docs/assets/readme/polylogue-verification.tape

render_gif polylogue-query 8.2
render_gif polylogue-products 8.2
render_gif polylogue-verification 2.8
