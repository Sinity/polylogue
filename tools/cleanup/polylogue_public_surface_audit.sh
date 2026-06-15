#!/usr/bin/env bash
# Polylogue public-surface stale-vocabulary audit (#1850 / #1810 / #1835 / #1848).
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
# #1848 adds a narrower product-docs audit for stale proof/QA/showcase/static
# product language. Verification-lab, generated-test, devtools, internals, and
# architecture docs may still use those words for internal machinery; the public
# getting-started/README surface should talk in terms of evidence,
# verification, demos, and read/export outputs instead.
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

PRODUCT_DOC_GLOBS=(
  "/README.md"
  "docs/README.md"
  "docs/getting-started.md"
  "docs/onboarding.md"
  "docs/installation.md"
  "docs/configuration.md"
  "docs/release.md"
  "docs/cloud-agents.md"
  "docs/export.md"
  "docs/generate.md"
  "docs/mcp-integration.md"
  "docs/library-api.md"
  "docs/search.md"
  "docs/browser-capture.md"
  "docs/security.md"
  "docs/maintenance.md"
  "docs/archive-backup.md"
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

declare -a PRODUCT_RULES=(
  '\bproof\b|||evidence or verification result (#1848)'
  '\bwitness\b|||fixture, evidence, or contract test (#1848)'
  '\bQA\b|||verification or test workflow (#1848)'
  '\bshowcase\b|||demo fixture or verification-lab baseline (#1848)'
  '\bstatic[- ]site\b|\bstatic site\b|||read/export output or documentation site (#1848)'
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

product_rg_args=()
for g in "${PRODUCT_DOC_GLOBS[@]}"; do product_rg_args+=(--glob "$g"); done
for rule in "${PRODUCT_RULES[@]}"; do
  pattern="${rule%%|||*}"
  replacement="${rule##*|||}"
  while IFS= read -r hit; do
    case "$hit" in
      *"polylogue-audit: allow"*) continue ;;
    esac
    if [ "$status" -eq 0 ]; then
      echo "Stale public product-surface wording found:" >&2
      echo >&2
      status=1
    fi
    echo "  $hit" >&2
    echo "    -> replace with: $replacement" >&2
  done < <(rg --no-heading --line-number --color never "${product_rg_args[@]}" -e "$pattern" 2>/dev/null || true)
done

if [ "$status" -ne 0 ]; then
  echo >&2
  echo "Public Polylogue vocabulary is 'session' and 'origin'; public product" >&2
  echo "docs should describe evidence/verification/demo/read-export surfaces," >&2
  echo "not stale proof/witness/QA/showcase/static-site product layers." >&2
  echo "Provider-wire parsers, internal storage/core, tests, generated quality" >&2
  echo "references, and historical/tombstone docs are outside this public audit." >&2
  echo "See tools/cleanup/$(basename "${BASH_SOURCE[0]}")." >&2
  exit 1
fi

echo "polylogue-public-surface-audit: clean (no stale public vocabulary/product wording)"
exit 0
