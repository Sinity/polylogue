#!/usr/bin/env bash
set -euo pipefail

cd /tmp/rewrite-arc20

# Omission function
strip_omissions() {
  git rm -rf --ignore-unmatch \
    mutmut.toml alembic.ini \
    tests/benchmarks/ tests/integration/test_mcp_mutations.py \
    docs/programs/ docs/plans/ \
    polylogue/alembic/ \
    .gitconfig .githooks/ .github/CONTRIBUTING.md .github/GIT_WORKFLOW.md .github/SETUP.md \
    examples/ demos/ \
    polylogue/schemas/providers/*.schema.json.gz \
    REMAINING_TASKS.md AGENTS.meta.md ASYNC-ARCHITECTURE.md \
    COMPLETION_SUMMARY.md FINALIZATION_STEPS.md IMPLEMENTATION_SUMMARY.md \
    MANUAL_TESTING_PROTOCOL.md \
    docs/archive/ docs/roadmap/ docs/releases/ \
    docs/CLAUDE.md docs/CLI-FEATURE-MAP.md docs/COVERAGE_VERIFICATION.md \
    docs/DB_PIVOT_PLAN.md docs/FUTURE_WORK_PLAN.md docs/IMPROVEMENTS.md \
    docs/STATUS.md docs/architecture-roadmap.md docs/automation.md \
    docs/backlog.md docs/beta_rollout.md docs/cli_tips.md docs/demo.md \
    docs/design-view-command.md docs/impl_plan.md docs/import_pipeline.md \
    docs/integrations/ docs/live_capture.md docs/observability.md \
    docs/performance.md docs/plan.md docs/report.md \
    docs/storage-backend-abstraction.md docs/ui-ux-survey.md \
    docs/ux-improvements-roadmap.md \
    scripts/ qa_outputs/ \
    2>/dev/null || true

  # Glob-based removals
  git ls-files 'docs/*2026-*' 'docs/*remaining*' 'docs/*session-recovery*' \
    'docs/*mutation*' 'docs/*workload*' 'docs/*triage*' 'docs/*test-ideas*' \
    'docs/*testing-gaps*' 'docs/*tasklist*' 'docs/*task[0-9]*' \
    'docs/TEST_*' 'docs/PARAMETRIZATION_*' 'docs/FINAL_TEST_*' \
    'docs/*-summary.md' 'docs/*architecture*' 'docs/*refactoring*' \
    'docs/*idempotency*' 'docs/*optimization*' 'docs/*polylogue.config*' \
    'docs/*phase-cohort*' 'docs/*canonical-archive*' 'docs/*intentional*' \
    'docs/*artifact-and-semantic*' 'docs/*refactoring-first*' \
    'docs/*testing-reliability*' \
    2>/dev/null | xargs -r git rm --ignore-unmatch 2>/dev/null || true
}

# Restore rewrite-only files
restore_rewrite_files() {
  git checkout history/rewrite -- .gitattributes .github/pull_request_template.md LICENSE 2>/dev/null || true
}

# Process one commit
# Args: SHA DATE SUBJECT BODY
do_commit() {
  local sha="$1" date="$2" subject="$3" body="$4"
  
  echo "=== Processing $sha: $subject ==="
  
  # Clean tree, checkout target state
  git rm -rf . 2>/dev/null || true
  git checkout "$sha" -- .
  
  # Restore rewrite-only files
  restore_rewrite_files
  
  # Strip omissions
  strip_omissions
  
  # Stage everything
  git add -A
  
  # Commit with original date
  GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
    git commit --allow-empty -m "$subject

$body

Original: $sha
Assisted-By: Codex" 2>&1
  
  echo "=== Done $sha ==="
  echo
}

# Commit #208
do_commit "8ee92f1" "2025-12-12T06:19:37+01:00" \
  "docs: remove implementation summary" \
  "- Remove IMPLEMENTATION_SUMMARY.md (consolidated into remaining tasks)"

# Commit #209
do_commit "7745314" "2025-12-12T06:20:00+01:00" \
  "feat: add sanitize-html redaction" \
  "- Add --sanitize-html to render/import/sync flows
- Sanitize persisted Markdown before hashing
- Record redacted runs with core redaction patterns
- Add redaction tests"

# SKIP #210 (ed1f85d) — DELETE: remaining tasks doc

# Commit #211
do_commit "705a2a9" "2025-12-12T12:41:41+01:00" \
  "feat: add config edit and multi-root prefs" \
  "- Add interactive 'polylogue config edit' command
- Include labeled roots in default scans
- Expand prefs defaults (root/sanitize)"

# Commit #212
do_commit "ba89c70" "2025-12-12T12:42:03+01:00" \
  "refactor: streamline render summaries" \
  "- Add summarize_render helper
- Apply diff/sanitize prefs consistently in render CLI"

# Commit #213
do_commit "16bf559" "2025-12-12T12:42:26+01:00" \
  "docs: mark P1 multi-root and UI done" \
  "- Update remaining tasks to reflect completed multi-root/prefs polish and UI streamlining"

# Commit #214
do_commit "6e7206e" "2025-12-12T12:47:59+01:00" \
  "perf: use content hashes for local session freshness" \
  "- Skip re-imports when raw hashes match, even if mtimes drift
- Still re-run for dirty outputs
- Add local sync hash tests"

# Commit #215
do_commit "9d4ea4f" "2025-12-12T12:48:34+01:00" \
  "docs: note content-hash freshness progress" \
  "- Mark local session hash-based freshness complete
- Leave export-level freshness pending"

# Commit #216
do_commit "ae1df14" "2025-12-12T12:55:02+01:00" \
  "perf: skip unchanged export bundles" \
  "- Record bundle hashes in local export sync
- Skip reprocessing ChatGPT/Claude exports when unchanged
- Allow override when pruning"

# Commit #217
do_commit "a476a25" "2025-12-12T12:55:21+01:00" \
  "docs: mark export freshness complete" \
  "- Update content-hash freshness note now that export bundles are hash-skipped"

# Commit #218
do_commit "12e71a6" "2025-12-12T15:13:08+01:00" \
  "feat: complete Click-first CLI and completions" \
  "- Route app/main/help/completions through Click
- Add optional-value Click option for --html
- Switch completion engine to introspect Click commands
- Fix search dispatcher import and Click naming collisions
- Update CLI tests to use Click runner"

# Commit #219
do_commit "3221b3e" "2025-12-12T15:13:18+01:00" \
  "docs: update CLI command module README" \
  "- Document Click-first entrypoint
- Treat argparse setup_parser as legacy"

# Commit #220
do_commit "af96000" "2025-12-12T20:51:26+01:00" \
  "refactor(cli): remove legacy argparse CLI" \
  "- Switch CLI helpers to SimpleNamespace-based args
- Extract completions, search/compare/branches/prune helpers
- Delete legacy argparse entrypoint and commands modules
- Update tests to target Click CLI"

echo "ALL DONE"
