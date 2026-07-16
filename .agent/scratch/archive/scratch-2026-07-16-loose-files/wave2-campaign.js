export const meta = {
  name: 'polylogue-p0-p1-campaign-wave2',
  description: 'Wave 2: rxdo substrate (3 clusters), a7xr mechanical dedup (Haiku flock), architecture, docs/legibility+README-positioning, 212.11 proof-world. Implementer + independent-reviewer loop per cluster.',
  phases: [
    { title: 'rxdo core' },
    { title: 'rxdo.9 measurement' },
    { title: 'rxdo.9 judgment' },
    { title: 'a7xr Haiku flock' },
    { title: 'Architecture' },
    { title: 'Docs/legibility' },
    { title: '212.11 proof-world' },
  ],
}

const REPO = '/realm/project/polylogue'
const WORKTREE_ROOT = '/realm/worktrees'

const CONVENTIONS = `
You are working in the Polylogue repo (${REPO}), a local AI-session archive tool. Standing conventions (condensed from CLAUDE.md):

GIT/PR: feature/<type>/<short-desc> branch off origin/master. Conventional commit prefixes. PR title <=72 chars imperative; PR body needs Summary/Problem/Solution/Verification with EXACT commands + output. Stage by path, never git add -A. No GitHub resolver keywords next to bead numbers — "Ref polylogue-<id>." only. Commit each logical chunk (isolated worktree, uncommitted work is lost). GitHub Actions CI is disabled repo-wide (billing-locked) — do not wait for it. Leave PR OPEN, do not merge — orchestrator runs merge-train after review.

VERIFICATION: devtools verify --quick for fast iteration; devtools test <files/-k expr> for focused pytest (never raw pytest, never blanket directories). Anti-vacuity: name the exact production dependency exercised and the mutation that would make your test fail. New module under polylogue/ -> devtools render topology-projection && devtools render topology-status.

BEADS: bd show <id> --json for full description/design/acceptance_criteria/notes before planning — notes often override the original description with superseding scope. Do NOT close beads yourself; bd update <id> --append-notes "..." with what you did + verification + PR link.
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
    ac_check: { type: 'string' },
    summary: { type: 'string' },
  },
  required: ['approved', 'findings', 'ac_check', 'summary'],
}

function implementPrompt({ title, beadIds, context, massRead }) {
  return `${CONVENTIONS}

## Your cluster: ${title}

Beads assigned (bd show <id> --json for full detail first): ${beadIds.join(', ')}

Context: ${context}

Start by reading in full: ${massRead}

Implement all beads as one coherent pass. Verify locally. Push and open the PR (or PRs if genuinely more independent than expected) but do NOT merge.

Report: branch, PR URL(s), bead_status per bead (satisfied/partial/deferred/already_done/misframed + one-line evidence), verification_summary with exact commands + pass/fail output.`
}

function reviewPrompt({ title, beadIds, context }, implResult, priorRounds) {
  const priorNote = priorRounds > 0
    ? `\nThis is review round ${priorRounds + 1} — prior rounds found blockers that should now be fixed. Check they're actually resolved, not just that new code exists.`
    : ''
  return `You are an independent, adversarial code reviewer — not the author. Find reasons this should NOT merge.

Repo: ${REPO}. Branch: ${implResult.branch}. PR(s): ${(implResult.pr_urls || []).join(', ')}.
Cluster: ${title}. Beads: ${beadIds.join(', ')}. Context: ${context}${priorNote}

1. Fetch/checkout the branch (git worktree add ${WORKTREE_ROOT}/review-${implResult.branch.replace(/\//g, '-')} ${implResult.branch}) and read the actual diff.
2. For every bead ID, pull acceptance_criteria via bd show and check each criterion against the diff — don't trust the self-report.
3. Look for: unsupported claims, tests that only exercise mocks, half-finished pieces, dropped beads, unrun verification claims.
4. Run devtools verify --quick yourself.
5. Clean up your review worktree when done.

Return: approved (true only if you'd merge as-is), findings (severity-ranked with file), a per-bead ac_check line, summary.`
}

function fixPrompt({ title, beadIds }, implResult, review) {
  const blockers = review.findings.filter((f) => f.severity === 'blocker' || f.severity === 'major')
  return `${CONVENTIONS}

## Fix round: ${title}

Reviewer found issues on branch ${implResult.branch} (beads: ${beadIds.join(', ')}). Continue that SAME branch — do not start over: cd ${WORKTREE_ROOT} && git worktree add ${WORKTREE_ROOT}/fix-${implResult.branch.replace(/\//g, '-')}-next ${implResult.branch} (pick a nearby unused path if that exists already).

Findings to address (blockers/majors must all be resolved):
${JSON.stringify(blockers.length ? blockers : review.findings, null, 2)}
Full summary: ${review.summary}

Fix the real issues; if you disagree with a finding, say so with reasoning. Re-verify, commit, push to the SAME branch, clean up your worktree.

Report same structure as before plus self_assessment addressing each finding (fixed / disagreed-because-X / deferred-to-new-bead-because-Y).`
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

  return { cluster_key: cluster.key, cluster_title: cluster.title, result: impl, final_review: lastReview, review_rounds: round }
}

const CLUSTERS = [
  {
    key: 'rxdo-core',
    title: 'rxdo core (query/evidence contracts)',
    beadIds: ['polylogue-rxdo.2', 'polylogue-rxdo.3', 'polylogue-rxdo.4', 'polylogue-rxdo.5', 'polylogue-rxdo.6', 'polylogue-rxdo.8', 'polylogue-rxdo.11'],
    massRead: 'polylogue/archive/query/ (whole dir), polylogue/insights/ (whole dir)',
    context: `The rxdo epic ("Evidence and analysis contracts: definitions, runs, relations, findings, judgments") is deliberately being built as SUBSTRATE before any actual demo implementation — the operator wants this to run unconditionally ahead of the 212.* demo work. rxdo.2 (protocol-versioned query definitions + promoted durable relations) is P1 and foundational — do it first, the others build on stable query/result-set refs existing. rxdo.4 (AssertionKind.FINDING) reuses the existing candidate->judge lifecycle, do not invent a parallel one. rxdo.3 (query-run telemetry) needs a frame/evaluation-world/privacy contract — read the bead's design carefully, this is not just "log queries". rxdo.5 (StandingQueryStage), rxdo.6 (DSL reference operands from query:/result-set:/cohort:), rxdo.8 (analysis recipes as DB-native objects), rxdo.11 (improvement-loop registry) round out the cluster — read each bead's own notes for the twelve recursive-loop failure modes mentioned in rxdo's own acceptance criteria and make sure new work has guards against them, not just happy-path code.`,
  },
  {
    key: 'rxdo9-measurement',
    title: 'rxdo.9 measurement substrate',
    beadIds: ['polylogue-rxdo.9.1', 'polylogue-rxdo.9.2', 'polylogue-rxdo.9.3', 'polylogue-rxdo.9.4', 'polylogue-rxdo.9.5', 'polylogue-rxdo.9.8', 'polylogue-rxdo.9.9'],
    massRead: 'polylogue/archive/query/ (whole dir), polylogue/insights/ (whole dir) — same area as rxdo-core, read that cluster is being worked in parallel by a sibling agent, avoid colliding on the same files if avoidable',
    context: `Part of rxdo.9 "Analysis rigor program: frame-exact validity, judgments, and experiments" — this slice is the measurement PRIMITIVES half: rxdo.9.1 (metric:<hash> content-addressed metric definitions) and rxdo.9.2 (ratios as derived objects: numerator_ref + denominator_ref) are foundational, do these first since rxdo.9.13 (rankers, in the sibling judgment cluster) depends on them conceptually. rxdo.9.3 (pre-registration with graph-provable ordering), rxdo.9.4 (holdout cohorts), rxdo.9.5 (standing-query alert budget: cooldowns/magnitude floors/deviation), rxdo.9.8 (bootstrap CIs for sampled results, exactness propagation), rxdo.9.9 (evidence ancestry walker: circularity/epoch skew/expired refs) — read each bead for exact scope, these are individually well-specified statistical/provenance primitives, not vague features.`,
  },
  {
    key: 'rxdo9-judgment',
    title: 'rxdo.9 judgment workflow',
    beadIds: ['polylogue-rxdo.9.6', 'polylogue-rxdo.9.7', 'polylogue-rxdo.9.10', 'polylogue-rxdo.9.11', 'polylogue-rxdo.9.12', 'polylogue-rxdo.9.13', 'polylogue-rxdo.9.14', 'polylogue-rxdo.9.15', 'polylogue-rxdo.9.16'],
    massRead: 'polylogue/insights/ (whole dir), polylogue/cli/ (grep for existing judge/triage commands like polylogue-p5g mentions)',
    context: `Part of rxdo.9, the judgment-workflow half (distinct code area from the sibling measurement-substrate cluster, which owns metrics/ratios/rankers primitives — rxdo.9.13 rankers depends on those conceptually, coordinate if you can see their work, but proceed independently if not). rxdo.9.11 (comparative judgment objects: pairwise + n-wise per-dimension) and rxdo.9.12 (judges as actors, human or agent, with measured calibration) are foundational for this half. rxdo.9.6 (blinded judgment view: mask provenance until verdict) and rxdo.9.7 (paired negative controls on findings) are methodology-integrity features — read carefully, these exist specifically to prevent this project's own analysis from fooling itself. rxdo.9.10 (experiment analysis projection over stc definitions and cohort results), rxdo.9.14 (active elicitation sessions: resorter loop with blinding), rxdo.9.15 (judge cascades: agent screens route to sparse operator gold), rxdo.9.16 (judgment UX surface: inbox, micro-moments, deliberate sessions) — given 9 beads is a lot, triage explicitly if needed (rxdo.9.11/9.12 first, defer UX-heavy ones like 9.16 with named follow-ups if time-constrained) rather than shipping 9 shallow slices.`,
  },
  {
    key: 'a7xr-haiku-flock',
    title: 'a7xr mechanical dedup (Haiku flock)',
    beadIds: [],
    isHaikuFlock: true,
    haikuItems: [
      { id: 'polylogue-48h', desc: 'Consolidate SQLite introspection helpers (10 copies of _table_exists and similar) into one shared helper.' },
      { id: 'polylogue-a7xr.3', desc: 'message_type_backfill reconstructs prose unordered and unfiltered — fix ordering/filtering to match the canonical read path.' },
      { id: 'polylogue-a7xr.5', desc: 'FTS trigger DDL declared twice: archive_tiers/index.py vs fts_lifecycle — single-source it.' },
      { id: 'polylogue-a7xr.7', desc: 'Role synonym vocabulary maintained by hand in two directions with no shared source — consolidate to one direction-agnostic table/map.' },
      { id: 'polylogue-a7xr.8', desc: 'Index-tier sibling-path derivation pasted ~7x with divergent existence checks — extract one shared helper matching the strictest existing check.' },
      { id: 'polylogue-a7xr.9', desc: 'Mechanical helper dedup sweep: scalar coercion quadruplet, _table_exists-style duplicates not covered by 48h.' },
      { id: 'polylogue-a7xr.11', desc: 'Prune protocols.py zero-consumer protocols + dead repo kwarg query parameters.' },
      { id: 'polylogue-a7xr.12', desc: 'neighbor_candidates needs a 4-method protocol, not the current 20-method one — narrow it (do this AFTER a7xr.11 since it blocks this one per the dependency graph).' },
      { id: 'polylogue-a7xr.15', desc: 'payloads.py: generic from_row for the 74 identical-name copy-paste lines — replace with one generic implementation.' },
      { id: 'polylogue-a7xr.16', desc: 'Table-drive the hand-aligned column triplicates in archive_tiers.' },
      { id: 'polylogue-a7xr.4', desc: 'One percentile implementation: three algorithms across five copies currently exist — pick the correct one, delete the other four call sites, point them at the single implementation.' },
      { id: 'polylogue-1a9', desc: 'Remove dead session-commit stubs + unused web-construct row + stale references (chore, mechanical deletion — verify each removal target truly has zero call sites first).' },
      { id: 'polylogue-dab', desc: 'Stop materializing run-projection cache rows; drop the DDL after confirming zero remaining readers.' },
      { id: 'polylogue-pf1', desc: "Sync/async divergence: diff the twin backends against the codebase's own '10 known gotchas' list for this pattern and fix any that have drifted." },
    ],
  },
  {
    key: 'architecture',
    title: 'Architecture (1r9c + a7xr features/bugs)',
    beadIds: ['polylogue-1r9c', 'polylogue-a7xr.1', 'polylogue-a7xr.2', 'polylogue-yp0', 'polylogue-0aj', 'polylogue-a7xr.10', 'polylogue-a7xr.13', 'polylogue-a7xr.14', 'polylogue-a7xr.17', 'polylogue-c9y'],
    massRead: 'polylogue/storage/ (whole dir, this is the ~11.3k-line control center 1r9c targets), polylogue/api/ (~5.9k lines), polylogue/daemon/ (~4.6k lines)',
    context: `This cluster is judgment-heavy, not mechanical — do not treat it like the sibling Haiku-flock cluster. 1r9c ("Decompose Polylogue execution control centers") is the big one: produce a source-grounded hotspot map first (call boundaries, ownership seams, import/layer constraints), then land AT LEAST ONE coherent extraction slice that makes a named control center materially smaller with no duplicate execution path — read 1r9c's own AC carefully, it explicitly wants a prioritized sequence with the rest filed as durable child beads, not everything done in one PR. a7xr.1 (sqlite connection leak sweep) and a7xr.2 (converger/repair disagreement on session-profile staleness) are real bugs, treat them with full test rigor. yp0 (daemon internal event bus, loops subscribe instead of polling — blocks polylogue-9e5.7, check that bead read-only for context) and 0aj (declared write-effects chain: post-commit effects as registry entries) are genuine small features. a7xr.10 (kill-or-adopt the search-provider lane — production currently bypasses the abstraction, investigate and decide), a7xr.13 (api/contracts write-surface shadow adapters verify copies not the real thing — fix the verification gap), a7xr.14 (collapse the one-operation operations-contract framework to concrete code), a7xr.17 (code_refs in operation/artifact catalogs are unresolved strings — make them real refs), c9y (package topology legibility: boundary doctrine for the 28-package structure — more a documentation/doctrine deliverable than code) round out the cluster.`,
  },
  {
    key: 'docs-legibility',
    title: 'Docs/legibility + README-positioning',
    beadIds: ['polylogue-3tl.8', 'polylogue-3tl.9', 'polylogue-ttu', 'polylogue-3tl.12', 'polylogue-3tl.17', 'polylogue-3tl.19', 'polylogue-3tl.10'],
    massRead: 'docs/ (whole dir), devtools/docs_surface.py, devtools/verify_doc_commands.py, devtools/render_topology_status.py, .agent/scratch/readme-positioning-2026-07-14/ (whole dir — this has a full external design study: a candidate README, executive brief, decision map, component catalog, and a WORKING reader-comprehension-test HTML tool, all cross-verified against live source already this session)',
    context: `3tl.9 and ttu explicitly need to share one mechanism per their own notes (turning devtools/docs_surface.py's hand-maintained DOCS_REFERENCE_ENTRIES/README_DOC_TITLES tuple into something generated from doc front-matter) — sequence them together, do not implement the shared mechanism twice. 3tl.12 (README de-meta/de-persuasion) should use .agent/scratch/readme-positioning-2026-07-14/README.polylogue.receipts-first.md as a starting DRAFT to reconcile against current master, not authority — the core idea already validated: lead with ONE deterministic claim-vs-evidence proof (polylogue demo receipts --compact, confirmed live and working) instead of five equally-weighted bullets. 3tl.17 (visual-tape/tour-artifact drift gate) has a much more precise contract already written into its own bd notes (exact owners, 7-step strict-check sequence, seeded-failure test list, determinism boundary) — read the bead's notes field, not just its description. 3tl.19 (reader-comprehension test harness) has an actual working prototype at .agent/scratch/readme-positioning-2026-07-14/reader-test-runner.html — adapt/verify it against this project's real docs/demos.md scoring doctrine rather than building from scratch. 3tl.8 (GitHub surface polish) should follow whatever hero framing 3tl.12 lands on, sequence it after. 3tl.10 (launch kit) is mostly a verification pass — material already exists in .agent/scratch/legibility-kit-2026-07-10/, cross-check claims against the live public-claims registry rather than writing new content.`,
  },
  {
    key: '212-11-proof-world',
    title: '212.11 proof-world on real archive data',
    beadIds: ['polylogue-212.11'],
    massRead: 'grep -rl "demo\\|proof.world\\|fixture" polylogue/demo/ devtools/ --include=*.py -il (read whatever that turns up), .agent/scratch/legibility-kit-2026-07-10/02b-demo-portfolio-expanded.md if it exists',
    context: `212.11 ("Incident 14:32 — one shared deterministic proof world for all flagship demos") should use a REAL, representative slice of the operator's own archive (at /realm/db/polylogue) rather than synthetic-only fixtures, per explicit operator direction — the point is representative data, not full determinism; some non-determinism from real data is fine.

CRITICAL PRIVACY STEP — you have delegated authority to vet content, but must exercise real care: select a candidate slice of real archive sessions (read-only queries against /realm/db/polylogue, e.g. via the polylogue CLI/MCP — never write to the live archive), screen it for anything sensitive (secrets/credentials, private third-party personal information, anything embarrassing or NSFW — the operator says "almost anything else goes, NSFW is the real line, and there shouldn't be much of that in typical dev-work sessions"). Produce the candidate slice PLUS a clear written summary of what it contains (session count, rough topics, why you judge it clean) as a distinct, clearly-labeled deliverable — do NOT bake it into the shared fixture path yet. Leave that as a flagged, easy-to-review artifact (e.g. a markdown summary + the extracted session refs) for a quick operator spot-check before it becomes the permanent shared proof-world fixture. This is the one part of this cluster that should NOT be treated as fully autonomous — everything else in the bead (the deterministic-ish harness/fixture machinery itself) can be built and merged normally.`,
  },
]

phase('rxdo core')
phase('rxdo.9 measurement')
phase('rxdo.9 judgment')
phase('a7xr Haiku flock')
phase('Architecture')
phase('Docs/legibility')
phase('212.11 proof-world')

async function runHaikuItem(item, clusterTitle) {
  const impl = await agent(
    `${CONVENTIONS}\n\n## Mechanical cleanup: ${item.id}\n\n${item.desc}\n\nRead bd show ${item.id} --json for full acceptance criteria first. Prefer a small scripted/grep-based sweep over many manual edits if the duplication pattern is uniform. Work in a fresh worktree branch off origin/master, verify (devtools verify --quick + focused tests), push, open a PR (do not merge). Report: branch, pr_urls, bead_status (one entry for ${item.id}), verification_summary.`,
    { label: `haiku-${item.id}`, phase: clusterTitle, schema: RESULT_SCHEMA, isolation: 'worktree', model: 'haiku' }
  )
  return { item_id: item.id, result: impl }
}

const results = await parallel([
  () => runCluster(CLUSTERS[0]),
  () => runCluster(CLUSTERS[1]),
  () => runCluster(CLUSTERS[2]),
  () => parallel(CLUSTERS[3].haikuItems.map((item) => () => runHaikuItem(item, CLUSTERS[3].title))).then((r) => ({ cluster_key: CLUSTERS[3].key, cluster_title: CLUSTERS[3].title, haiku_results: r.filter(Boolean) })),
  () => runCluster(CLUSTERS[4]),
  () => runCluster(CLUSTERS[5]),
  () => runCluster(CLUSTERS[6]),
])

return { results: results.filter(Boolean) }
