export const meta = {
  name: 'polylogue-p0-p1-campaign-wave1',
  description: 'Wave 1: 8 parallel worktree-isolated clusters. Implementer builds + self-verifies, then an independent code-reviewer subagent adversarially reviews against the bead ACs; loop implementer<->reviewer up to 12 rounds per cluster.',
  phases: [
    { title: 'Raw-identity repair' },
    { title: 'Browser extension' },
    { title: 'Provider/Origin normalization' },
    { title: 'Migration hardening' },
    { title: 'Excision/secret hygiene' },
    { title: 'Interactive performance' },
    { title: 'Sinex evidence mode' },
    { title: 'Hermes bridge' },
  ],
}

const REPO = '/realm/project/polylogue'
const WORKTREE_ROOT = '/realm/worktrees'

const CONVENTIONS = `
You are working in the Polylogue repo (${REPO}), a local AI-session archive tool. Standing conventions for this repo (from CLAUDE.md, condensed):

GIT/PR DISCIPLINE:
- Feature branch from fresh origin/master, name feature/<type>/<short-desc>. Conventional commit prefixes (feat:/fix:/refactor:/test:/docs:/chore:). PR title = squash-merge subject, <=72 chars, imperative.
- PR body needs Summary/Problem/Solution/Verification sections, with EXACT commands run and the output line that matters, not "tests pass".
- Stage by path (git add <file>), never git add -A on non-trivial changes.
- Do NOT use GitHub resolver keywords (Closes/Fixes/Resolves) next to issue/bead numbers. Reference beads as "Ref polylogue-<id>." only.
- Commit after each logical chunk succeeds — you are in an isolated worktree; uncommitted work is lost if it's cleaned up.
- GitHub Actions CI is disabled repo-wide (billing-locked) — do not wait for or expect CI checks. Rely on local verification only. Leave the PR OPEN, do not merge — the orchestrator runs a merge-train afterward.

VERIFICATION:
- devtools verify --quick (format+lint+mypy+render-all-check) for fast iteration; devtools test <files/-k expr> for focused pytest (never raw pytest, never blanket directories).
- Anti-vacuity: your verification must name the exact production dependency exercised and the mutation/removal that would make your new test fail.
- New module under polylogue/ → run devtools render topology-projection && devtools render topology-status, commit the updated files, or render all --check fails.

BEADS:
- bd show <id> --json for full description/design/acceptance_criteria/notes on each assigned bead — the notes field often has superseding scope corrections that override the original description.
- Do NOT close beads yourself; leave a bd update <id> --append-notes "..." with what you did + verification + PR link. The orchestrator closes beads after merge-train.
`

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    branch: { type: 'string' },
    pr_urls: { type: 'array', items: { type: 'string' } },
    bead_status: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          bead_id: { type: 'string' },
          status: { type: 'string', enum: ['satisfied', 'partial', 'deferred', 'already_done', 'misframed'] },
          evidence: { type: 'string' },
        },
        required: ['bead_id', 'status', 'evidence'],
      },
    },
    verification_summary: { type: 'string' },
    self_assessment: { type: 'string' },
  },
  required: ['branch', 'pr_urls', 'bead_status', 'verification_summary'],
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
    ac_check: { type: 'string', description: 'For each bead ID in scope, one line: satisfied / not satisfied and why.' },
    summary: { type: 'string' },
  },
  required: ['approved', 'findings', 'ac_check', 'summary'],
}

function implementPrompt({ title, beadIds, context, massRead }) {
  return `${CONVENTIONS}

## Your cluster: ${title}

Beads assigned (fetch full details for each via \`bd show <id> --json\` before planning): ${beadIds.join(', ')}

Context and constraints specific to this cluster (from prior research this session):
${context}

Start by reading in full: ${massRead}

Work in a fresh worktree branch off origin/master. Implement all beads in this cluster as one coherent pass — they're grouped as closely related work, implement them together rather than as disconnected patches. Verify locally (devtools verify --quick + focused tests for everything you touched). Push your branch and open the PR (or PRs, only split if the beads turn out more independent than expected) but do NOT merge.

An independent reviewer will adversarially check your work against each bead's acceptance criteria next — you will not see their identity or bias, so make your PR description and commit history legible enough for a stranger to verify your claims, not just to convince a skim-reader.

Report back: the exact branch name, PR URL(s), your bead_status for every assigned bead (satisfied/partial/deferred/already_done/misframed with one-line evidence each), and your verification_summary with exact commands + pass/fail output.`
}

function reviewPrompt({ title, beadIds, context }, implResult, priorRounds) {
  const priorNote = priorRounds > 0
    ? `\nThis is review round ${priorRounds + 1} for this branch — prior rounds found blockers that should now be fixed. Check specifically whether the previously-reported blockers are actually resolved, not just whether new code exists.`
    : ''
  return `You are an independent, adversarial code reviewer — NOT the person who wrote this code. Your job is to find reasons this PR should NOT merge, not to be agreeable.

Repo: ${REPO}. Branch to review: ${implResult.branch}. PR(s): ${(implResult.pr_urls || []).join(', ')}.

Cluster under review: ${title}. Beads in scope: ${beadIds.join(', ')}.
Cluster context (for your own understanding of intent, not a substitute for reading the code): ${context}
${priorNote}

Do this:
1. Fetch and checkout the branch (git fetch origin && git worktree add ${WORKTREE_ROOT}/review-${implResult.branch.replace(/\//g, '-')} ${implResult.branch}, or git diff against origin/master if you prefer — your choice, but read the ACTUAL diff, not just the PR description).
2. For EVERY bead ID in scope, pull its acceptance_criteria via bd show <id> --json and check each criterion individually against the diff. Do not accept the implementer's self-report at face value.
3. Look specifically for: claims not backed by the diff ("unified"/"fixed"/"complete" that the code doesn't support), tests that only exercise mocks/stubs rather than the real production path, half-finished pieces, silently dropped beads, verification commands that weren't actually run or whose output doesn't show what's claimed, and any violation of this repo's actuator safety pattern (dry-run/apply/CAS/fail-closed) if the cluster touches repair actuators.
4. Run devtools verify --quick yourself in that worktree to confirm the implementer's verification claim is reproducible, not just asserted.
5. Clean up your review worktree when done (git worktree remove) — leave the implementer's branch itself untouched.

Return your verdict: approved (true only if you'd merge this as-is with no further changes), a findings list (severity-ranked, each with the file it concerns), a per-bead ac_check line, and a summary.`
}

function fixPrompt({ title, beadIds }, implResult, review) {
  const blockers = review.findings.filter((f) => f.severity === 'blocker' || f.severity === 'major')
  return `${CONVENTIONS}

## Fix round for cluster: ${title}

An independent reviewer found issues with your existing work on branch ${implResult.branch} (beads: ${beadIds.join(', ')}). Continue that SAME branch — do not start over.

1. cd ${WORKTREE_ROOT} && git worktree add ${WORKTREE_ROOT}/fix-${implResult.branch.replace(/\//g, '-')}-${Date.now ? '' : ''}${Math.random ? '' : ''}next ${implResult.branch} — (if that exact path exists already, pick a nearby unused path) — this checks out your EXISTING branch with its EXISTING commits, do not create a new branch.
2. Reviewer's findings to address (severity-ranked, blockers/majors must all be resolved; use judgment on minors/nits):
${JSON.stringify(blockers.length ? blockers : review.findings, null, 2)}
Full reviewer summary: ${review.summary}
Per-bead AC check from the reviewer: ${review.ac_check}
3. Fix the real issues — do not just reword comments or add unused code to look like you addressed something. If you disagree with a finding, say so explicitly in your report with your reasoning rather than silently ignoring it.
4. Re-run devtools verify --quick + focused tests. Commit, push to the SAME branch (git push, not force unless you rewrote history for a good reason).
5. Clean up your worktree when done.

Report back the same structure as before: branch, pr_urls, bead_status, verification_summary, and self_assessment addressing each reviewer finding explicitly (fixed / disagreed-because-X / deferred-to-new-bead-because-Y).`
}

async function runCluster(cluster) {
  let impl = await agent(implementPrompt(cluster), {
    label: `${cluster.key}-impl-1`,
    phase: cluster.title,
    schema: RESULT_SCHEMA,
    isolation: 'worktree',
  })
  if (!impl) return { cluster_key: cluster.key, cluster_title: cluster.title, result: null, rounds: 0 }

  let round = 0
  const MAX_ROUNDS = 12
  let lastReview = null
  while (round < MAX_ROUNDS) {
    const review = await agent(reviewPrompt(cluster, impl, round), {
      label: `${cluster.key}-review-${round + 1}`,
      phase: cluster.title,
      schema: REVIEW_SCHEMA,
    })
    lastReview = review
    if (!review || review.approved) break
    round++
    if (round >= MAX_ROUNDS) break
    const fixed = await agent(fixPrompt(cluster, impl, review), {
      label: `${cluster.key}-fix-${round}`,
      phase: cluster.title,
      schema: RESULT_SCHEMA,
    })
    if (fixed) impl = fixed
  }

  return {
    cluster_key: cluster.key,
    cluster_title: cluster.title,
    result: impl,
    final_review: lastReview,
    review_rounds: round,
  }
}

const CLUSTERS = [
  {
    key: 'raw-identity',
    title: 'Raw-identity repair (P0)',
    beadIds: ['polylogue-lkrc', 'polylogue-lkrc.2', 'polylogue-lkrc.3', 'polylogue-yla8', 'polylogue-yla8.6', 'polylogue-t0dy', 'polylogue-57rp', 'polylogue-5k5l.1'],
    massRead: 'polylogue/storage/repair.py, polylogue/sources/live/ (whole dir), polylogue/storage/sqlite/archive_tiers/ (whole dir)',
    context: `This is the most correctness-critical cluster in the whole campaign — it repairs live archive data identity. lkrc/lkrc.2/lkrc.3 form a parent-child chain about browser-capture raws whose origin disagrees with production-parsed identity (unknown-export vs chatgpt-export mismatches); read lkrc's notes carefully, there is an "adversarial loop iteration 5" note listing four specific unresolved proof gaps (capture_mode missing from envelopes, blob_ref.source_path not bound to raw source_path, native_id not bound to reparsed session id, restore_canonical_head omitting fields) that must be addressed, not just the original description. yla8/yla8.6 are about raw materialization ordering and CAS frontier convergence, also parent-child, also read notes for the exact residual scope (yla8.10, a related bead not in your cluster, already closed part of this chain — do not re-do that work, read what it actually closed first via bd show polylogue-yla8.10). t0dy and 57rp are adjacent raw-authority reconciliation bugs. 5k5l.1 is about ChatGPT interpreter asset auth before classifying as expired.
CRITICAL: this touches the live production archive's actuators. Do NOT run any repair actuator against the live archive at /realm/db/polylogue — implement and test exclusively against fixtures/tmp_path in the test suite. Never stop the live polylogued.service. Leave live-archive execution to the operator.
Every repair actuator in this codebase follows a strict pattern: dry-run proves eligibility with an exact digest, apply requires that exact digest, all mutations are all-or-nothing within one transaction, and every non-matching shape fails closed rather than being silently accepted. Match this pattern exactly for any new actuator; do not weaken CAS or infer authority from partial evidence.`,
  },
  {
    key: 'browser-ext',
    title: 'Browser extension (backfill/recovery + capture UI)',
    beadIds: ['polylogue-jlme.3', 'polylogue-jlme.4', 'polylogue-jlme.4.1', 'polylogue-s8gb', 'polylogue-06zm', 'polylogue-yyvg', 'polylogue-bj5h', 'polylogue-wvji', 'polylogue-ys30', 'polylogue-4g3n'],
    massRead: 'browser-extension/ (whole directory: src/, tests/, manifest.json)',
    context: `A real bug was already found and FIXED+MERGED earlier this session in this exact area: the receiver's CORS preflight (polylogue/browser_capture/server.py, the Python daemon side, already on master) was missing Access-Control-Allow-Private-Network, which made the extension popup hang on "Checking receiver..." forever. Pull latest master first. If you still see similar symptoms, look for the same class of issue (missing headers, silently-swallowed fetch failures) before assuming something else.
jlme.3/jlme.4/jlme.4.1 are about backfill-ledger preservation and stale-receiver-contract handling during browser profile recovery — read their notes, they say "extension PR in progress" from earlier in the session, check git log/branches for any half-landed work (git branch -a | grep jlme) before starting from scratch. s8gb and 06zm are about oversized backfill capture recovery and job-ledger survival across profile re-seed.
yyvg + children (bj5h/wvji/ys30/4g3n) are the "Extension redesign: ambient two-way surface" epic — a genuinely large feature surface. Given the size, it's fine to land the foundational layer (ys30, "blended per-message capture dot" Layer 1) solidly with full tests rather than four shallow half-implementations — say explicitly which of the four you completed vs deferred.
Check for an existing test runner (npm test) in browser-extension/ and use it.`,
  },
  {
    key: 'provider-origin',
    title: 'Provider/Origin + action normalization',
    beadIds: ['polylogue-9e5.8', 'polylogue-9e5.8.4', 'polylogue-1frn'],
    massRead: 'polylogue/sources/parsers/codex.py, polylogue/archive/actions/ (whole dir), polylogue/core/enums.py, polylogue/core/sources.py',
    context: `9e5.8 is the sequenced retirement plan for provider->origin vocabulary (see docs/provider-origin-identity.md); 9e5.8.4 is step 2: renaming literal public tokens across CLI flags and HTTP scope names. Read 9e5.8's own notes for the full sequenced plan and current status (some steps may already be done — check git log --oneline --grep="provider.*origin"). GEMINI and DRIVE both collapse to AISTUDIO_DRIVE (non-injective) — read this repo's CLAUDE.md "Vocabulary: Provider vs Origin vs Source" section before touching anything, a naive find-replace is unsafe.
1frn is a separate bug: Codex exec commands aren't normalized correctly for action queries — likely in polylogue/sources/parsers/codex.py or polylogue/archive/actions/actions.py (check git log -- polylogue/archive/actions/actions.py for its latest shape before starting).`,
  },
  {
    key: 'migration-hardening',
    title: 'Migration/schema-forward hardening',
    beadIds: ['polylogue-b5l.1', 'polylogue-pf8s', 'polylogue-qg6x'],
    massRead: 'devtools/archive_schema_fast_forward.py (whole file, ~780 lines), polylogue/storage/sqlite/lifecycle.py, tests/unit/devtools/test_archive_schema_fast_forward.py',
    context: `This is the exact tool the orchestrator used LIVE tonight to cut the production archive over from v35 to v36 — it worked cleanly end to end, treat activate_prepared_forward/plan_clone_forward as a proven-correct baseline, not something to restructure. b5l.1 wants index rebuilds writer-exclusive and crash-resumable — extend the existing _require_service_stopped + rollback try/except pattern rather than inventing a new one. pf8s wants backup attestation cached during migration — look at _require_receipt_identity's existing sha256+size+version binding pattern. qg6x wants schema-forward clone proofs resumable — the receipt schema already persists a "status": "prepared" state that a real run tonight reused across two failed attempts before the third succeeded; understand that existing resumability before adding more.`,
  },
  {
    key: 'excision',
    title: 'Excision / secret hygiene',
    beadIds: ['polylogue-27m'],
    massRead: 'grep -rl "excision\\|redaction\\|sanitiz" polylogue/ --include=*.py (then read whatever that turns up), docs/security.md',
    context: `"Excision and secret hygiene: the archive can forget on purpose." Read this bead's full description/design/acceptance_criteria via bd show — it's the only bead in this cluster so there's room to do it thoroughly. This repo's CLAUDE.md flags a related, PAUSED cluster: "the entire sanitized-export/portfolio cluster came from an AI code-review's offhand suggestion, NOT an operator request... paused for issue-by-issue audit." Confirm 27m is NOT part of that paused cluster before proceeding (it reads as distinct, deliberate archive-level excision, not export sanitization — but verify against the bead's own text).`,
  },
  {
    key: 'interactive-perf',
    title: 'Interactive performance (whole 20d sequence + ng9m)',
    beadIds: [
      'polylogue-20d.14', 'polylogue-20d.2', 'polylogue-20d.1', 'polylogue-20d.12', 'polylogue-20d.13',
      'polylogue-20d.6', 'polylogue-20d.15', 'polylogue-20d.4', 'polylogue-20d.5', 'polylogue-20d.7',
      'polylogue-20d.8', 'polylogue-20d.10', 'polylogue-20d.11', 'polylogue-20d.16', 'polylogue-d22s',
      'polylogue-opc', 'polylogue-oxz', 'polylogue-ng9m',
    ],
    massRead: 'polylogue/daemon/ (whole dir), polylogue/cli/click_app.py, polylogue/sources/live/ (whole dir)',
    context: `The operator explicitly asked for this whole epic to go to ONE subagent with rich prompting and full adversarial review — this is the largest cluster in the campaign, work in dependency order, not bead-ID order:
1. 20d.14 FIRST — named latency budgets in docs/plans/slo-catalog.yaml, the epic's own measurement contract.
2. 20d.2 — remove the ~2s cold-import tax for the daemonless path (fallback, not the main fix).
3. 20d.1 — CLI->daemon fast path over a Unix domain socket. This is the STRUCTURAL fix; treat it as the centerpiece.
4. 20d.12 (conceptually after 20d.1) — daemon result cache.
5. 20d.13 — SSE push instead of polling (relates to webui bead bby.11, not in your cluster — read bby.11 read-only for context, don't implement it, just don't build something incompatible).
6. 20d.6 + 20d.15 (bulk vs live ingest lanes) + ng9m (daemon catch-up memory) together — ng9m's notes say a prior investigation found the harness itself flawed (pickle-based measurement, missing PSS/cgroup/IO signals) and was retargeted to build a better harness FIRST, in the style of the existing #2841 harness (grep for it) — do that harness work as part of this slice.
7. 20d.4/.5/.7/.8/.10/.11 — direct-path/storage-profile fixes; 20d.8 (bound the 43s regen) blocks 20d.10 (post-filter memoization) per the existing graph, do 20d.8 first.
8. 20d.16, d22s, opc, oxz can go in any order; oxz (perf instrumentation doctrine) first if time-constrained since others plausibly want its primitives.
Given the size, it's expected to land the highest-leverage subset (20d.14 + 20d.1 + 20d.2, since those retire most of the epic's own cited evidence) as a strong PR and explicitly defer the rest with named follow-up beads, rather than 18 shallow changes. State your triage reasoning.`,
  },
  {
    key: 'sinex-evidence',
    title: 'Sinex-backed evidence mode',
    beadIds: ['polylogue-303r', 'polylogue-303r.2'],
    massRead: 'grep -rl "sinex" polylogue/ --include=*.py -i (read whatever that turns up in full), docs/architecture.md',
    context: `303r is "Sinex-backed evidence mode: canonical materials and rebuildable projections" — an epic about how Polylogue operates when Sinex is the canonical evidence store (standalone SQLite mode is a PERMANENT product requirement per this repo's CLAUDE.md — never frame this as deprecating sqlite). 303r.2 is "publish Sinex materials and anchored observations with durable retry". Read docs/architecture.md's rings model first. If Sinex itself (sibling project at /realm/project/sinex, NOT in your worktree) needs corresponding changes, note that as an explicit cross-repo follow-up rather than attempting to edit that other repo.`,
  },
  {
    key: 'hermes-bridge',
    title: 'Hermes bridge (non-demo)',
    beadIds: [
      'polylogue-fs1.2', 'polylogue-fs1.4', 'polylogue-fs1.5', 'polylogue-fs1.7',
      'polylogue-fs1.8', 'polylogue-fs1.10', 'polylogue-fs1.11', 'polylogue-fs1.13', 'polylogue-ox0',
    ],
    massRead: 'grep -rl "hermes" polylogue/ --include=*.py -i (read whatever that turns up in full), docs/architecture.md',
    context: `fs1 epic notes cite critical path "fs1.3 -> fs1.11 -> fs1.12" but fs1.3 no longer exists as a bead (already closed/consolidated — bd show polylogue-fs1.3 will likely 404, expected). Your cluster's frontier is fs1.7 (upstream Hermes archival contract + durable lifecycle-event spool) then fs1.11 (read-only recall + effective-context audit loop) — do these first, later items build on them. fs1.8/fs1.10/ox0 are more independent, do after the frontier two land solidly.
EXPLICITLY OUT OF SCOPE: fs1.6 and fs1.12 (demos) — do not implement even if tempting, note as correctly-deferred if you notice the pull.
This epic does NOT give Polylogue write/admin authority over Hermes or vice versa — read fs1's design field for the channel-separation model before anything that looks like cross-system mutation.`,
  },
]

phase('Raw-identity repair')
phase('Browser extension')
phase('Provider/Origin normalization')
phase('Migration hardening')
phase('Excision/secret hygiene')
phase('Interactive performance')
phase('Sinex evidence mode')
phase('Hermes bridge')

const results = await parallel(CLUSTERS.map((c) => () => runCluster(c)))

return { results: results.filter(Boolean) }
