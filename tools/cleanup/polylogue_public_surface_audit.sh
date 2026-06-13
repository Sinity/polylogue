#!/usr/bin/env bash
# Polylogue public-surface stale-vocabulary audit (#1850 / #1810 / #1835).
#
# Polylogue's public schema-v1 vocabulary is `session` and `origin`. The terms
# `conversation` / `conversation_id` and the field `provider_meta`, plus the
# legacy reader route `/c/{conversation_id}`, are stale and must not reappear on
# any *public surface*: the README, hand-authored (non-historical) docs, the
# daemon web reader (routes + HTML), the JSON-RPC/HTTP API layer, and the CLI.
#
# This audit is path-aware. It does NOT police:
#   - provider-wire parsers that read external vendor payloads verbatim
#     (`polylogue/sources/**`) — those field names mirror the vendor's own JSON;
#   - the internal storage/core/schema layer where the `provider`/`provider_meta`
#     identifiers are deferred internal renames behind origin<->provider boundary
#     translation (tracked debt; public API already speaks `origin`);
#   - tests, fixtures, and synthetic wire builders (incl. stale-term rejection
#     tests, which must mention the stale term on purpose);
#   - historical/tombstone docs (audits, retros, design canvases, plans).
#
# It deliberately does NOT police the bare word `provider`: the deep
# `provider` -> `origin` internal rename is schema-migration-scale work tracked
# separately (Ref #1835). Only the unambiguous stale tokens below are failed.
#
# Failing output names the exact file:line and the replacement term. A single
# line inside a policed path may opt out with a trailing
# `# polylogue-audit: allow <reason>` marker when it legitimately documents an
# external vendor format.
#
# Exit 0 = clean; exit 1 = stale terms found; exit 2 = misuse.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if ! command -v rg >/dev/null 2>&1; then
  echo "polylogue-public-surface-audit: ripgrep (rg) is required" >&2
  exit 2
fi

# Public surfaces the audit scans. A leading slash anchors the glob to the repo
# root (the root README only; nested component READMEs are their own surface).
PUBLIC_GLOBS=(
  "/README.md"
  "docs/**/*.md"
  "polylogue/daemon/**/*.py"
  "polylogue/api/**/*.py"
  "polylogue/cli/**/*.py"
)

# Path prefixes excluded from scanning even when matched by a public glob:
#   - historical/tombstone docs that intentionally preserve old vocabulary;
#   - docs/providers/** documents external vendor wire formats and the deferred
#     internal `provider_meta` field (Ref #1790), so it is provider-wire, not
#     stale Polylogue vocabulary.
EXCLUDE_GLOBS=(
  "docs/audits/**"
  "docs/retro/**"
  "docs/design/**"
  "docs/plans/**"
  "docs/providers/**"
)

# Stale token => human-readable replacement guidance.
# Each entry is "REGEX|||REPLACEMENT".
declare -a RULES=(
  '\bconversation_id\b|||session_id (schema-v1 identifier)'
  '\bconversationId\b|||sessionId (schema-v1 identifier)'
  '\bconversation\b|||session (schema-v1 vocabulary)'
  '\bprovider_meta\b|||origin-typed fields (Ref #1790; provider_meta is stale)'
  "['\"\`]/c/|||/s/{session_id} reader route"
)

rg_args=()
for g in "${PUBLIC_GLOBS[@]}"; do rg_args+=(--glob "$g"); done
for g in "${EXCLUDE_GLOBS[@]}"; do rg_args+=(--glob "!$g"); done

status=0
for rule in "${RULES[@]}"; do
  pattern="${rule%%|||*}"
  replacement="${rule##*|||}"
  # --no-heading line-oriented output: path:line:text
  while IFS= read -r hit; do
    # Honor inline opt-out marker for legitimate external-vendor-format docs.
    # Comment syntax is irrelevant (``#``, ``//``, ``<!-- -->`` all work).
    case "$hit" in
      *"polylogue-audit: allow"*) continue ;;
    esac
    if [ "$status" -eq 0 ]; then
      echo "Stale schema-v1 vocabulary found on public surfaces:" >&2
      echo >&2
      status=1
    fi
    echo "  $hit" >&2
    echo "    -> replace with: $replacement" >&2
  done < <(rg --no-heading --line-number --color never "${rg_args[@]}" -e "$pattern" 2>/dev/null || true)
done

if [ "$status" -ne 0 ]; then
  echo >&2
  echo "Public Polylogue vocabulary is 'session' and 'origin'. Stale terms are" >&2
  echo "allowed only in provider-wire parsers, internal storage/core, tests," >&2
  echo "and historical/tombstone docs. See tools/cleanup/$(basename "${BASH_SOURCE[0]}")." >&2
  exit 1
fi

echo "polylogue-public-surface-audit: clean (no stale schema-v1 vocabulary on public surfaces)"
exit 0
