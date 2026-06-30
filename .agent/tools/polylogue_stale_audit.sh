#!/usr/bin/env bash
set -euo pipefail
root=${1:-.}
rg -n --glob '!**/.git/**' --glob '!**/.venv/**' --glob '!**/__pycache__/**' \
  '\b(provider|conversation|content_blocks|proof)\b|\bQA\b|showcase|exercise|bulk-export|context-pack|resume-candidates|\brecent\b' \
  "$root"
