#!/usr/bin/env bash
set -euo pipefail

readme_caption() {
  local row="${2:-22}"
  printf '\033[s\033[%s;1H\033[48;5;236m\033[38;5;231m %-105s \033[0m\033[u' "$row" "$1"
}

if [[ "${1:-}" != "--no-seed" ]]; then
  demo_root="${1:-.local/vhs-readme-demo}"
  count="${2:-8}"
  rm -rf "$demo_root"
  eval "$(devtools lab-corpus seed --count "$count" --output-dir "$demo_root" --env-only 2>/dev/null | sed -n '/^export /p')"
fi

export POLYLOGUE_FORCE_PLAIN=1
export PS1='$ '
clear
