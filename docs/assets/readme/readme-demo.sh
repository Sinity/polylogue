#!/usr/bin/env bash
set -euo pipefail

scene="${1:?usage: readme-demo.sh query|products|verification}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

caption() {
  local text="$1"
  local row="${2:-22}"
  printf '\033[s\033[%s;1H\033[48;5;236m\033[38;5;231m %-94s \033[0m\033[u' "$row" "$text"
}

seed_env() {
  local demo_root="$1"
  local count="${2:-8}"
  rm -rf "$demo_root"
  eval "$(devtools lab-corpus seed --count "$count" --output-dir "$demo_root" --env-only 2>/dev/null | sed -n '/^export /p')"
  export POLYLOGUE_FORCE_PLAIN=1
}

run_scene() {
  local note="$1"
  local display_command="$2"
  local command="${3:-$display_command}"
  local hold="${4:-3}"
  if [[ "$command" =~ ^[0-9]+$ ]]; then
    hold="$command"
    command="$display_command"
  fi
  clear
  caption "$note"
  printf '$ %s\n' "$display_command"
  bash -lc "$command"
  sleep "$hold"
}

case "$scene" in
  query)
    seed_env ".local/vhs-readme-query" 8
    run_scene "Synthetic archive stats: providers, messages, attachments, and date coverage." \
      "polylogue --plain stats" \
      4
    run_scene "Query-first CLI: results keep provider, title, timestamp, and conversation id together." \
      "polylogue --plain error list --limit 3" \
      5
    ;;
  products)
    seed_env ".local/vhs-readme-products" 8
    run_scene "Products turn raw sessions into rollups: profiles, phases, threads, summaries, tags, and costs." \
      "polylogue --plain products tags --limit 5" \
      5
    run_scene "Publication is explicit; catch-up imports can run without rebuilding the site each time." \
      "polylogue --plain run site -o .local/vhs-readme-products/site-preview" \
      5
    ;;
  verification)
    export POLYLOGUE_FORCE_PLAIN=1
    run_scene "Changed files route to proof obligations before escalating to the full gate." \
      "devtools affected-obligations --path polylogue/cli/query.py" \
      "devtools affected-obligations --json --path polylogue/cli/query.py | jq -r '\"changed: \" + (.changed_paths | join(\", \")), \"affected obligations: \" + ((.affected_obligations | length) | tostring), \"\", \"inner loop:\", (.inner_loop_checks[] | \"- \" + (.command | join(\" \")) + \" — \" + .reason), \"\", \"PR gates:\", (.pr_gates[] | \"- \" + (.command | join(\" \")) + \" — \" + .reason)'" \
      6
    ;;
  *)
    printf 'unknown README demo scene: %s\n' "$scene" >&2
    exit 2
    ;;
esac
