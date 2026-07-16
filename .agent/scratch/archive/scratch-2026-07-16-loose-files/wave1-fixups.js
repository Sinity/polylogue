export const meta = {
  name: 'polylogue-wave1-fixups',
  description: 'Fix the 3 Wave-1 clusters held back by adversarial review (excision blocker, raw-identity majors, hermes-bridge majors), continuing their existing branches, then re-review and report merge-readiness.',
  phases: [
    { title: 'Excision fix' },
    { title: 'Raw-identity fix' },
    { title: 'Hermes-bridge fix' },
  ],
}

const REPO = '/realm/project/polylogue'

const CONVENTIONS = `
You are working in the Polylogue repo (${REPO}). Standing conventions (condensed from CLAUDE.md):
- Conventional commit prefixes. PR body needs Summary/Problem/Solution/Verification with exact commands + output.
- Stage by path, never git add -A. No GitHub resolver keywords next to bead numbers; use "Ref polylogue-<id>." only.
- devtools verify --quick + devtools test <files/-k expr> (never raw pytest, never blanket directories) for verification.
- GitHub Actions CI is disabled repo-wide (billing-locked) -- do not wait for CI checks.
- Do NOT close beads yourself; leave bd update <id> --append-notes "..." with what you did + verification + PR link.
`

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    pr_url: { type: 'string' },
    findings_addressed: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          finding: { type: 'string' },
          resolution: { type: 'string', enum: ['fixed', 'disagreed', 'deferred'] },
          evidence: { type: 'string' },
        },
        required: ['finding', 'resolution', 'evidence'],
      },
    },
    verification_summary: { type: 'string' },
    ready_to_merge: { type: 'boolean' },
  },
  required: ['branch', 'pr_url', 'findings_addressed', 'verification_summary', 'ready_to_merge'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    approved: { type: 'boolean' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor', 'nit'] },
          description: { type: 'string' },
          file: { type: 'string' },
        },
        required: ['severity', 'description'],
      },
    },
    summary: { type: 'string' },
  },
  required: ['approved', 'findings', 'summary'],
}

const CLUSTERS = [
  {
    key: 'excision-fix',
    title: 'Excision fix',
    branch: 'feature/security/excision-secret-hygiene-27m',
    worktreePath: '/realm/project/polylogue/.claude/worktrees/wf_5be33c21-b3d-5',
    prUrl: 'https://github.com/Sinity/polylogue/pull/2875',
    beadIds: ['polylogue-27m', 'polylogue-layg'],
    findings: `BLOCKER: The excision non-resurrection guarantee is bypassable through a second, real production write chokepoint. write_source_raw_session (polylogue/storage/sqlite/archive_tiers/source_write.py, around line 434) checks is_blob_hash_excised before insert, but a sibling write path does not — find it and either route it through the same check or extract a shared chokepoint helper both paths call.
MAJOR: resolve_session_excision_target/apply_session_excision (polylogue/security/excision.py) resolve only rows keyed directly to the target session_id (sessions/messages/blocks WHERE session_id = ?) and are unaware of related rows that should also be covered by the same excision (check what "related" means here — forked/resumed sessions sharing content, attachments referenced from elsewhere, etc. — read the bead's own AC for the exact intended scope).
MAJOR: docs/plans/security-privacy-coverage.yaml marks captured_content_secret_detection implemented:true and removes it from coverage_gaps entirely (previously severity major) — verify the secret scanner's actual coverage genuinely supports this claim before keeping it; downgrade the claim if it doesn't.
MINOR: raw sqlite3.connect() calls in apply_session_excision/plan_session_excision omit the PRAGMA busy_timeout every other write chokepoint in this codebase sets via connection_prof (or equivalent) — add it for consistency.`,
  },
  {
    key: 'raw-identity-fix',
    title: 'Raw-identity fix',
    branch: 'feature/fix/raw-identity-repair-cluster',
    worktreePath: '/realm/project/polylogue/.claude/worktrees/wf_5be33c21-b3d-1',
    prUrl: 'https://github.com/Sinity/polylogue/pull/2877',
    beadIds: ['polylogue-t0dy', 'polylogue-lkrc.3', 'polylogue-hleq'],
    findings: `MAJOR: repair_duplicate_raw_identity's apply-mode receipt (polylogue/storage/repair.py) is a single unlocked receipt_path.write_text(...) call issued after the transaction commits, gated only by a TOCTOU-racy "if receipt_path.exists(): raise" check. Two concurrent apply calls can both pass the exists-check before either writes, corrupting the receipt guarantee. Fix using the same locked/atomic write pattern this codebase's other actuators use for receipt files (find an existing example and match it — do not invent a new pattern).
MAJOR: the byte-frontier competing-head branch in _browser_canonical_authority_conflict_witness (the "elif competing_frontier_kind == 'byte' and competing_raw_id != raw_id:" block) re-reads the competing raw mid-function without re-proving it hasn't changed since the outer proof was taken — this is a re-check-of-a-stale-read bug, not just style. Either re-prove atomically within the same transaction/snapshot, or explicitly document why the race is acceptable here (it probably is not, given this touches identity-conflict evidence).
MINOR: record_browser_canonical_authority_conflict_blockers mutates the durable, irreplaceable user.db tier unconditionally — no apply flag, no proof-digest gate, no receipt file, unlike every other actuator in this codebase's established dry-run/apply/CAS/fail-closed pattern (which this cluster's own implementer was explicitly told to follow). Bring it in line with that pattern, or if it's genuinely exempt (e.g. it only ever writes a low-stakes candidate-status assertion, not a durable judgment), say so explicitly in the PR body with reasoning.
NIT: none of the three new functions has CLI wiring, unlike the older browser-origin/quarantined-raw actuators which each get a dedicated maintenance_group.command(...) subcommand — add if time allows, not blocking.`,
  },
  {
    key: 'hermes-bridge-fix',
    title: 'Hermes-bridge fix',
    branch: 'feature/hermes/lifecycle-spool-and-bridge',
    worktreePath: '/realm/worktrees/polylogue-hermes-bridge',
    prUrl: 'https://github.com/Sinity/polylogue/pull/2876',
    beadIds: ['polylogue-fs1.2', 'polylogue-fs1.7', 'polylogue-segf'],
    findings: `MAJOR: the ATIF/ATOF observer-trace detector (looks_like_atif_payload in polylogue/sources/parsers/hermes_spans.py) only ever matches a synthetic polylogue_artifact: "hermes_atif_trace" marker key invented by this repo's own marker_payload() fixture helper — it does not detect the real Hermes wire format, since no Hermes source access was available when it was written. This is already honestly disclosed in the bead's own notes (fidelity capped at "inferred"). You likely cannot fully fix this without real Hermes source access — if that's still unavailable, do NOT fabricate a fix; instead (a) make the disclosure more prominent in user-facing docs (not just internal bead notes) so nobody mistakes "inferred" for "verified", and (b) file the fs1.2.1 follow-up bead the implementer's notes said should exist but was never created (bd create, don't just mention it in prose again).
MAJOR: fs1.7's reconciliation mechanism (reconcile_lifecycle_events / HermesLifecycleReconciliation, claimed to satisfy "an incomplete event stream is reconciled visibly against the session snapshot") is unit-tested only against synthetic fixtures your own code generates, not proven against anything resembling real Hermes output. Same constraint as above — if real data isn't available, strengthen the fixtures to at least cover adversarial/malformed shapes a real stream might produce (truncated JSON, out-of-order events, duplicate lifecycle markers) rather than only the happy path, and be explicit in the PR about what remains unproven.
MINOR: _reject_duplicated_transcript (polylogue/sources/hooks.py) is a naive length-only heuristic (>2000 chars on 6 hardcoded field names) duplicated verbatim in three independent places — extract a shared helper.
MINOR: _persist_record's Origin.from_string(_ORIGIN_TOKEN_BY_PROVIDER.get(provider_token, "codex-session")) silently defaults any unrecognized provider token to codex-session instead of raising — this could misclassify genuinely-unknown providers as Codex. Raise or route to an explicit unknown-origin bucket instead.
MINOR: _archive_correlate_hermes_context_deliveries (polylogue/api/archive.py) wraps its whole read path in except (sqlite3.Error, ValueError): return (), collapsing "archive not yet initialized" and "archive present but corrupted" into the same silent empty result — distinguish them.`,
  },
]

function fixPrompt(c) {
  return `${CONVENTIONS}

## Continue existing work: ${c.title}

An independent adversarial reviewer found real issues in the existing PR ${c.prUrl} (branch ${c.branch}, beads: ${c.beadIds.join(', ')}). The branch and its worktree already exist at ${c.worktreePath} — cd there directly (do NOT create a new worktree, do NOT create a new branch). If that exact path is gone for some reason, run: git worktree add ${c.worktreePath}-continue ${c.branch} (fetch origin first if needed) and work there instead.

Findings to address:
${c.findings}

Fix the REAL issues — do not just reword comments or add unused code to look like you addressed something. If you disagree with a finding after genuinely investigating it, say so explicitly with your reasoning rather than silently ignoring it; "disagreed" is a legitimate resolution if justified. If a finding genuinely requires resources you don't have (e.g. real Hermes source access), say so and do the best available mitigation rather than faking a fix.

Re-run devtools verify --quick + focused tests for everything you touched. Commit your fixes to the SAME branch (git push, not force unless you have a specific reason). Update the PR description to reflect the fix. Report: branch, pr_url, a findings_addressed list (each finding, its resolution status, and evidence), your verification_summary, and whether you believe this is ready_to_merge.`
}

function reviewPrompt(c, fixResult) {
  return `You are an independent, adversarial code reviewer. This is a RE-review — a prior round already found real blocker/major issues on this branch (${fixResult.branch}, PR ${fixResult.prUrl}), and a fix was just attempted. Your job: verify the fix is real, not cosmetic.

Beads in scope: ${c.beadIds.join(', ')}. Original findings that triggered this fix round:
${c.findings}

The fixer's own report on what they did: ${JSON.stringify(fixResult.findings_addressed, null, 2)}

1. Fetch and check out the branch (or diff against origin/master) and read the ACTUAL current diff — do not trust the fixer's self-report.
2. For every original blocker/major finding, verify independently whether it is now genuinely resolved, still present, or was legitimately deferred with real justification.
3. Run devtools verify --quick yourself to confirm.
4. Watch specifically for cosmetic non-fixes: reworded comments, unused code added to look like effort, a test that still doesn't exercise the real path.

Return your verdict: approved (true only if every blocker/major is now genuinely resolved or legitimately, explicitly deferred — minors/nits don't block approval), findings (anything still wrong), and a summary.`
}

async function runFix(c) {
  const fixed = await agent(fixPrompt(c), {
    label: c.key,
    phase: c.title,
    schema: RESULT_SCHEMA,
  })
  if (!fixed) return { cluster_key: c.key, result: null, review: null }

  const review = await agent(reviewPrompt(c, fixed), {
    label: `${c.key}-review`,
    phase: c.title,
    schema: REVIEW_SCHEMA,
  })

  return { cluster_key: c.key, cluster_title: c.title, pr_url: c.prUrl, result: fixed, review }
}

phase('Excision fix')
phase('Raw-identity fix')
phase('Hermes-bridge fix')

const results = await parallel(CLUSTERS.map((c) => () => runFix(c)))

return { results: results.filter(Boolean) }
