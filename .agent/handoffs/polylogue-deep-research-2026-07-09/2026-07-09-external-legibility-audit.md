---
created: 2026-07-09
purpose: Read-only audit + plan across 7 polylogue-3tl external-legibility beads (+ 6l6 grab-bag)
status: findings-and-plans, no implementation performed
project: polylogue
---

# External legibility audit — 2026-07-09

Scope: `polylogue-3tl.3`, `.7`, `.8`, `.9`, `.15`, `polylogue-ttu`, `polylogue-6l6`.
Excludes `polylogue-3tl.14` (handled elsewhere). Method per bead: read the bead's
description/design/AC, then inspect live repo/GitHub state with `gh api`, `rg`,
and direct file reads — no code or docs written, no `bd` mutation, no commits.

---

## 1. polylogue-3tl.3 — Claim-vs-evidence leaderboard variant (multi-model)

**Finding: the per-model breakdown already exists and already includes an open
model.** `.agent/demos/claim-vs-evidence/claim-vs-evidence.report.json` has a
`by_model` array (alongside `by_origin`, `by_handler_class`, `by_tool`) with rows
for `claude-haiku-4-5-20251001`, `claude-opus-4-6`, `claude-sonnet-4-20250514`,
`claude-opus-4-5-20251101`, `claude-opus-4-20250514`, `claude-opus-4-8`, an
`unknown` bucket, and **`deepseek-v4-pro`** — silent_proceed=22,
classified_outcomes=22, silent_rate_among_classified=1.0 (100%, tiny n),
silent_rate_lower_bound=5.6%. So the "closed vs open on identical task classes"
comparison the bead wants is *partially* live today, just not rendered as a
leaderboard and not screenshot-safe (n=22 for DeepSeek is exactly the kind of
thin cell the bead's coverage gate says to refuse rather than publish).

**What's missing:**
- No Hermes rows appear in the current report (check whether Hermes sessions
  exist in the archive at task-class granularity the harness uses; if present
  but absent from `by_model`, the harness's model-name extraction may not be
  covering that origin's usage-event shape).
- **No cost/cache columns anywhere in the report schema** (`report_version`,
  `rates`, `totals`, `by_model`, etc. — none carry `cost_usd`, cached/uncached
  lanes, or subscription-credit figures). This is squarely blocked on
  `polylogue-f2qv` (open epic, "Provider usage & cost honesty"): its children
  `f2qv.3` (dual cost view) and `f2qv.4` (single LiteLLM pricing source) are
  P1/open. The report's own `rates` key exists but contains counts, not $.
- **No coverage/n_min refusal logic visible in the artifact** — the report
  emits `deepseek-v4-pro` at n=22 without a visible "cell below n_min, refuse"
  marker. The bead's own AC explicitly demands this ("coverage gate REFUSES
  cells below n_min rather than publishing thin comparisons... empty backing
  evidence renders unknown/not-supported, not zero"). Whether the *generator*
  (`devtools workspace claim-vs-evidence`) has this gate internally and simply
  didn't trigger it, or lacks it entirely, needs a source read of that command
  module (not done in this pass — flag as the first thing an implementer
  checks, since it's the actual AC-blocking gap, not a data-availability gap).
- Adversarial-quoting hardening (stage-separated scoring, human spot-check
  subset, self-judging contamination controls, methodology-up-front) is not
  evidenced in the current packet's README/COLD_READER_GATE — those files
  describe methodology (marker-based classifier, calibration precision/recall)
  but not a multi-model-specific adversarial framing.

**Plan for the implementer:** (1) locate and read the `claim-vs-evidence`
generator module to confirm/deny the n_min-refusal claim before building on
top of it. (2) Extend the existing `by_model` render path into a leaderboard
table (reuse, don't rebuild — the aggregation is already correct-shaped). (3)
Cost/cache columns are hard-blocked on f2qv.3/f2qv.4 landing first — do not
fake dual-cost columns ahead of that. (4) Investigate why Hermes has zero
presence in `by_model` (ingestion gap vs harness gap). (5) n=22 for DeepSeek
needs either a larger sample frame or an explicit "insufficient n" cell before
this becomes a public leaderboard — publishing it as-is today would violate
the bead's own adversarial-quoting discipline.

**Follow-up bead proposals:**
- *"claim-vs-evidence: verify/implement n_min coverage refusal in the
  generator"* — one-line scope: confirm whether `devtools workspace
  claim-vs-evidence` already gates thin cells; if not, add the refusal before
  any multi-model table is published.
- *"Investigate Hermes-origin absence from claim-vs-evidence by_model rows"* —
  one-line scope: determine if Hermes sessions lack the tool-result/usage
  fields the harness keys on, or if the harness's model-name resolution
  misses Hermes's provider payload shape.

---

## 2. polylogue-3tl.7 — Proven install matrix

**Finding: the release machinery is fully built across every requested lane —
and has never fired once.** `git tag -l` returns zero tags; `gh release list`
returns nothing. `pyproject.toml` is still `version = "0.1.0"`. Concretely:

| Lane | Workflow | State |
| --- | --- | --- |
| PyPI (wheel/sdist, Trusted Publishing, Sigstore, CycloneDX SBOM) | `.github/workflows/release.yml` | Built, triggers on `v*.*.*` tag push; **never triggered** (no tag exists) |
| GHCR container (`docker run`) | `.github/workflows/container.yml` | Builds+smokes (`--version`) on every PR/push to master already (amd64), pushes on tag; PR-time smoke already proves `docker run --entrypoint polylogue <tag> --version` works today on master-tagged images, just not release-tagged ones |
| Homebrew tap | `.github/workflows/homebrew-bump.yml` | Triggers on tag push, polls PyPI sdist, `brew audit --strict --online`; **never fired** |
| Nix flake / `nix run` | `flake.nix` (`apps.<system>` at line 286) + FlakeHub push workflow | `nix run github:Sinity/polylogue[#polylogued\|#polylogue-mcp]` is the **only channel `docs/installation.md` currently documents as available today** — and per that doc it *does* work (cachix binary cache wired) |
| pipx/uvx from PyPI | — | Not documented anywhere (`rg pipx\|uvx README.md docs/installation.md` = zero hits) — can't even be attempted pre-release |

`docs/installation.md` itself is unusually honest already: it states in its
first paragraph *"PyPI, Homebrew, container images, and browser-store
extension packages are release-channel targets; do not treat them as available
until the release workflow has published and smoke-tested those artifacts."*
This is good — it means 3tl.7's "state the story honestly" AC is substantially
pre-satisfied for the non-Nix lanes (they're marked aspirational, not silently
broken). The gap is the *scheduled CI matrix itself* (weekly + pre-release
fresh-environment smoke), which does not exist as a workflow — every existing
workflow is push/tag/PR-triggered, none is `schedule:`-triggered, and none
does a genuinely fresh-environment `uvx`/`pipx install` (the closest is
`container.yml`'s PR-time `docker run --version` smoke, and
`release.yml`'s `verify-distribution` which runs on the *build host*, not a
clean matrix runner).

Windows: no mention found in `docs/installation.md` at all (not even a "WSL2
only" statement) — the bead's AC #2 ("Windows story stated honestly") is
currently unmet by omission, not by a false claim.

**Plan:** this bead cannot be executed to completion without first cutting a
release (there is no PyPI/Homebrew/GHCR-tagged artifact to smoke against on a
schedule). Two honest sub-sequencings:
1. Design the scheduled matrix workflow now (ubuntu+macos × {pipx-from-PyPI,
   brew-tap, docker-run-from-GHCR, nix-run} × smoke script) so it's ready to
   turn on the moment a release exists — this is exactly what "design, don't
   build" scoped the bead to.
   `nix run` and `docker run`-from-master-branch-tag lanes *can* be smoke-tested
   today since Nix flake outputs and master-branch container tags already
   exist without a version tag.
2. Add the missing Windows/WSL2 sentence to `docs/installation.md` — trivial,
   in-scope for this bead's AC #2, doesn't require a release.

**Follow-up bead proposals:**
- *"Cut first tagged release (v0.1.0) to unblock PyPI/Homebrew/GHCR smoke
  lanes"* — one-line scope: this is the actual blocking dependency for 3/4 of
  the matrix; likely deserves to be a named prerequisite bead rather than
  silently discovered mid-implementation of 3tl.7.
- *"Add Windows/WSL2 install story to docs/installation.md"* — one-line scope:
  currently silent, no release dependency, doable standalone.

---

## 3. polylogue-3tl.8 — GitHub surface polish

**Direct `gh api repos/Sinity/polylogue` read — current state vs the bead's
target list:**

| Item | Current | Target (per bead) | Gap |
| --- | --- | --- | --- |
| Description | `"AI conversation archive — parse, search, and explore exports from ChatGPT, Claude, Codex, Gemini"` | The positioning one-liner from `polylogue-3tl` epic notes: *"the local flight recorder for AI work — a cross-provider system of record where every metric resolves to raw bytes"* | **Stale** — repo description still uses the pre-repositioning "chat export viewer" framing the epic explicitly diagnosed as underselling the tool 100x |
| Topics | `[]` (empty) | `ai, sqlite, local-first, claude, chatgpt, archive, observability, agent-memory, ...` | **Completely unset** — zero GitHub Topics means zero topic-based discovery |
| Homepage URL | `None` | Docs/pages site URL | **Unset** |
| Social preview image | `uses_custom_open_graph_image: None` / no custom OG image | Rendered card | **Unset** (default GitHub avatar-based card) |
| Discussions | `has_discussions_enabled: False` | Bead recommends OFF until launch | **Already matches recommendation** — no action needed |
| Wiki | `has_wiki_enabled: True` | Not addressed by bead but a stray default surface | Bead doesn't mention it; flag as a likely-should-be-OFF default (empty wiki reads worse than none, same logic bead applies to Discussions) |
| Issue templates | 4 structured templates present (`01-feature-or-change.yml`, `02-bug-or-regression.yml`, `03-cleanup-or-refactor.yml`, `04-research-or-decision.yml`) + `config.yml` (`blank_issues_enabled: false`) | Stranger-facing bug/question templates alongside internal ones | These 4 read as **operator/agent-facing** (feature/change, cleanup/refactor, research/decision are internal-workflow shaped) — no obviously stranger-facing "I hit a bug as a user" or "how do I..." template exists distinctly from `02-bug-or-regression.yml`, which may already suffice; worth a content read (not done here) to check first-person framing |
| README badges | CI status, Python version, MIT license | + PyPI version, docs link (bead says "honest ones only") | **PyPI badge would currently be dishonest** (no PyPI release exists — see finding #2 above); docs-link badge is addable now |
| SECURITY.md | **Absent** (`test -f SECURITY.md` → not found) | Present, with loopback-posture statement + contact | **Missing entirely.** Note: `docs/daemon-threat-model.md` and `docs/security.md` already exist with the substantive content (per docs tree listing) — this may be a case of "write the doctrine, forgot the GitHub-recognized filename," i.e. a `SECURITY.md` stub that points at `docs/security.md` may be nearly free |
| Pinned finding artifacts | N/A — no finding has a published URL yet (blocked on `polylogue-3tl.4`, not in this pass's scope) | Pin 2 findings once `3tl.4` lands | **Correctly sequenced as blocked**, not a gap in this pass |

**Plan:** this bead is almost entirely `gh api`/`gh repo edit`-scriptable and
has no code dependencies — genuinely the fastest bead of the seven to execute
once someone is authorized to touch live GitHub repo settings. Sequencing per
the bead's own note ("after 3tl.1 one-liner exists") is satisfied — `3tl.1` is
closed and the one-liner exists in the epic notes and current README first
line ("**Polylogue is the system of record for AI work.**"). The repo
description update should almost certainly reuse that exact closed-bead
copy rather than reinvent it.

**Follow-up bead proposals:**
- *"Write SECURITY.md pointing at docs/security.md + docs/daemon-threat-model.md"*
  — one-line scope: minimal stub, doctrine already exists, just needs the
  GitHub-recognized entry point.
- *"Audit issue templates for a genuinely stranger-facing bug/question path"*
  — one-line scope: read the 4 existing YAML templates' actual field/label
  content (not done in this pass) to confirm `02-bug-or-regression.yml` reads
  naturally for a first-time external user vs an internal agent.

---

## 4. polylogue-3tl.9 — Docs coverage lint design

**Finding: every inventory the lint needs to be built from already exists as
a typed, importable Python object** — this bead is much closer to
implementation-ready than "design a lint" implies; the design work is mostly
*wiring*, not invention:

| Surface | Existing inventory | Location |
| --- | --- | --- |
| CLI commands/verbs | `iter_command_paths()` | `polylogue/cli/command_inventory.py` (already consumed by `devtools/verify_doc_commands.py`) |
| MCP tools | `EXPECTED_TOOL_NAMES` — verified 96 entries via `awk`/grep count | `tests/infra/mcp.py:15` |
| Config keys | `config_inventory_by_key()` — has its own executable inventory test already (`tests/unit/core/test_inventory_covers_loaded_defaults_and_public_polylogue_config_properties`) asserting the inventory covers every loaded default + public `PolylogueConfig` property | `polylogue/config.py`, tested in `tests/unit/core/test_config_inventory.py` |
| Daemon routes | `ROUTE_CONTRACTS: tuple[RouteContract, ...]` — typed dataclass registry with kind/stability/auth_policy per route, explicit docstring: *"docs, tests, OpenAPI generation... do not infer security semantics from handler names"* | `polylogue/daemon/route_contracts.py` |
| Docs tree (what's currently documented) | `DOCS_REFERENCE_ENTRIES` / `README_DOC_TITLES` — a static tuple of `DocsEntry(title, path, description)` | `devtools/docs_surface.py` (158 lines total) |
| Existing lint precedent to model on | `devtools/verify_doc_commands.py` (502 lines) — already does a scoped version of exactly this pattern for CLI/devtools/polylogued commands: scans README+docs markdown code spans, resolves tokens against `command_catalog.COMMANDS`/`iter_command_paths`, and is explicitly framed as "keep #1262/#869/#2438 closed: doc drift... should fail a CI gate" | same file |
| The "same set-diff pattern as the topology gate" the bead cites | `devtools/render_topology_status.py` (176 lines) — reads a target YAML + `ROOT.glob("polylogue/**/*.py")`, diffs realized vs declared, renders a generated markdown dashboard between `<!-- BEGIN/END GENERATED -->` markers | same pattern is directly reusable for a docs-coverage dashboard |

**Gap is real but narrow:** none of these four inventories (config, MCP tools,
routes, CLI commands) is currently *cross-referenced against the docs tree* —
`verify_doc_commands.py` checks docs→commands (a doc's command example must be
real) but not the reverse (every real command must appear in *some* doc). No
existing lint does the reverse direction the bead wants for any of the four
surfaces. `DOCS_REFERENCE_ENTRIES`/`README_DOC_TITLES` is presently a **hand-maintained
static tuple** (not generated from the docs tree itself), so it's already the
right shape to diff against but is a second source of truth that itself needs
to be either generated or kept honest — same tension `polylogue-ttu` (bead
#6 below) flags independently ("the index is GENERATED... an unindexed doc
fails render all --check").

**Plan:** implementation shape is: (1) new `devtools/verify_docs_coverage.py`
following the `verify_doc_commands.py`/`render_topology_status.py` pattern
exactly — pull the four inventories, grep the docs tree (or the front-matter
tagging `ttu` proposes) for each surface's identifier, named-missing-entry
failure. (2) This bead and `polylogue-ttu` should probably **share the docs
front-matter/tiering mechanism** rather than build it twice — `ttu`'s plan to
extend `render docs-surface` to emit a tiered index from front-matter is the
same generator surface this bead's docs-tree side needs to query. Flag this
dependency explicitly to whoever picks up either bead first.

**Follow-up bead proposals:**
- *"Reverse-direction doc-commands lint: every CLI command/MCP tool/config
  key/daemon route reachable from docs"* — one-line scope: this is 3tl.9's own
  literal AC #1, but worth a dedicated bead separate from the visual-freshness
  (AC #2) and devloop-gate (AC #3) halves of 3tl.9 if this bead's scope grows
  — the grab-bag precedent (6l6) explicitly authorizes this kind of split.
- *"Visual-freshness lint for docs screenshots/GIFs against visual-tapes specs"*
  — one-line scope: 3tl.9 AC #2 is a materially different mechanism (perceptual
  diff, not set-diff) from AC #1 and could ship independently.

---

## 5. polylogue-3tl.15 — Skeptic-section card for README (draft included)

**Grounding artifact found:** `.agent/demos/claim-vs-evidence/` is the current,
regenerated (2026-07-04, index schema v24) packet, and it is *already* the
right shape for this card — it exists specifically to answer "what did the
next turn do after a structured failure," which is precisely the
grep-cannot-see-this claim the bead wants proven. Its `PUBLIC_REPRODUCTION.md`
already contains a private-data-free reproduction recipe using
`polylogue demo seed` + `actions where is_error:true | group by followup_class
| count`.

**Positioning inconsistency the bead flags is confirmed live:** the bead says
"docs teach bare-token FTS... while the strict command floor (#1842) requires
signalled intent." I did not do a full grep of `docs/getting-started.md` /
`docs/search.md` in this pass (time-boxed), but `PUBLIC_REPRODUCTION.md`'s own
reproduction commands correctly use **signalled** forms throughout (`polylogue
--plain --format json actions where is_error:true | ...` — a field-syntax/
pipeline expression, not a bare token) — so the demo packet itself is not the
source of the inconsistency; the inconsistency (if it survives) is specifically
in the older getting-started/search.md prose the bead names, which an
implementer should grep directly (`rg -n '"polylogue [a-z]' docs/getting-started.md
docs/search.md` for bare-looking examples) before drafting the fix.

**Draft card** (ready-to-use prose, per the bead's instruction that this one
"IS close to copy-ready"):

> ### Why not just `grep ~/.claude/projects`?
>
> `grep` finds text. It cannot tell you whether a tool call succeeded, whether
> the agent noticed a failure, what a session actually cost, or which of the
> matching lines are duplicates from a resumed/forked conversation replaying
> its own parent.
>
> Take one concrete question: **"When a tool call failed, did the next agent
> turn acknowledge it, or silently proceed?"** `grep -r "error"
> ~/.claude/projects` returns lines of prose — text that *mentions* failure,
> with no reliable way to tell an acknowledged failure from a silent one, or to
> avoid counting the same failure twice across a forked session that replays
> its parent's transcript.
>
> Polylogue answers it from structure, not prose: every tool call and its
> result are paired rows in the `actions` view (`tool_use` joined to
> `tool_result` by `tool_id`), each result carries a provider-reported
> `tool_result_is_error` / `tool_result_exit_code` — never guessed from
> wording — and lineage-aware session composition means a forked
> conversation's replayed prefix is counted once, not once per fork. Over the
> full archive (42,033 structured tool-result failures, index schema v24),
> asking "did the next turn acknowledge it" this way finds a **24.1% silent
> lower bound** (rising to 37.0% under a stricter next-3-turn window) — a
> number `grep` cannot produce at all, because it has no concept of "next
> turn," "acknowledged," or "the same failure counted once."
>
> Reproduce the method (no private data required):
> ```bash
> export POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-demo
> polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json
> polylogue --plain --format json actions where is_error:true | group by followup_class | count
> ```
> Full methodology, sample frame, and caveats:
> [claim-vs-evidence finding](.agent/demos/claim-vs-evidence/README.md) *(will
> become a docs-site URL once `polylogue-3tl.4` lands — do not link the raw
> repo path in the published README)*.

**Plan:** the comparison-table version (bead explicitly defers this until
after a public finding URL exists) is correctly out of scope until `3tl.4`
ships. The prose card above can ship independently and does not require
`3tl.4` — it can cite the in-repo packet path for now and be updated to the
docs-site URL in the same PR that lands `3tl.4`, or as a fast-follow.

**Follow-up bead proposals:**
- *"Grep docs/getting-started.md + docs/search.md for bare-token FTS examples
  contradicting the #1842 command floor"* — one-line scope: targeted grep pass
  the bead calls for but this audit didn't have budget to complete exhaustively.
- *"Wire skeptic card's finding link to the docs-site URL once 3tl.4 ships"* —
  one-line scope: trivial follow-up, avoid it blocking the initial card merge.

---

## 6. polylogue-ttu — Docs information architecture

**Confirmed orphan gap, with exact counts.** `find docs -name "*.md"` returns
**68 files** (including `audits/`, `design/`, `providers/`, `plans/`, `retro/`
subdirectories). `devtools/docs_surface.py`'s `DOCS_REFERENCE_ENTRIES` tuple —
the single generated source `docs/README.md` renders from — lists **28**
entries, and `README_DOC_TITLES` (what the top-level `README.md` links) is a
**19**-entry subset of that. So roughly **40 of 68 docs/*.md files are
unreachable from either generated index** — this is close to, not exactly,
the bead's own "54 entries / ~26 linked" estimate (the bead's numbers look
like an earlier or coarser count; the live number is worse: 68 files, 28 in
the fuller index).

**Concretely orphaned (present in `docs/`, absent from
`DOCS_REFERENCE_ENTRIES`):** `agent-forensics.md`, all of `audits/*.md` (4
files, including 3 from just this week: 2026-07-09 daemon-loop-lock,
hash-boundary-census, race-window-audit), `cloud-agents.md`, `cost-model.md`,
`daemon.md` (note: `daemon-threat-model.md` *is* indexed, `daemon.md` is not
— an odd asymmetry), most of `design/*.md` (only `design/README.md` is
indexed; `agent-first-mcp.md`, `archive-storytelling.md`, `project-memory.md`,
`query-action-workflows.md` [superseded by the indexed `product/workflows.md`?],
`query-set-algebra.md`, `second-brain.md`, `session-lineage-model.md`,
`time-machine.md`, `whole-product.md` are all orphaned), `export.md`,
`hooks.md`, `insights.md`, `insights-rigor-matrix.md`, `onboarding.md`,
`plans/demo-corpus-construct-audit.md`, `plans/query-pipeline-substrate.md`,
`projection-render-spec.md`, `providers/chatgpt.md` / `claude-ai.md` /
`claude-code.md` / `gemini.md` / `index.md` / `openai-codex.md` (only
`providers/README.md` is indexed — the 5 per-provider pages plus a stray
`index.md` are all orphaned), `retro/2026-05-24-1498-cascade.md` (explicitly
cited as required reading in this repo's own CLAUDE.md — orphaned from the
docs tree despite being load-bearing), `schema-annotations.md`, `schema.md`,
`security.md` (relevant to the `3tl.8` SECURITY.md gap above — the content
exists, just unreachable), `test-economics.md`, `topology-status.md` (itself
a generated dashboard, orphaned from its own docs index),
`web-route-readiness-states.md`. (Not orphaned, confirmed indexed: `glossary.md`,
`dev-loop.md` as "Branch-Local Development Loop", `provider-origin-identity.md`,
`daemon-threat-model.md`.)

**Stale-doc candidates worth checking in the triage pass** (not verified line-
by-line in this audit, flagged by name/date pattern only): `docs/onboarding.md`
and `docs/export.md` are plausible candidates for describing a removed/renamed
CLI era given the repo's history of archived CLI verbs (`generate`/`qa` → 
`demo`/`analyze` per this repo's own CLAUDE.md gotchas) — an implementer should
diff their command examples against `verify_doc_commands.py`'s pass/fail before
assuming staleness, since that lint would already catch a *command* drift (just
not an orphan-from-index drift, which is `ttu`'s distinct axis).

**Plan:** the bead's own 4-part design (tier into guide/reference/internals/
decisions+plans, front-matter-driven generated index, stale-triage-to-archive-
not-delete, one PR) is sound and matches the repo's existing generated-surface
discipline. Concretely:
1. `DOCS_REFERENCE_ENTRIES`/`README_DOC_TITLES` in `devtools/docs_surface.py`
   need to become **generated from front-matter**, not a hand-maintained tuple
   — otherwise the new tiered index just becomes a third hand-maintained list
   alongside the two that already exist there today (the tuple itself, and
   `README.md`'s own hardcoded links).
2. Because `3tl.9`'s coverage lint (bead #4 above) needs the *same* "is this
   doc reachable from an index" primitive, sequence these two beads together
   or explicitly hand off the front-matter mechanism from whichever lands
   first.
3. Old-path → new-path moves: every orphan listed above is a `git mv`
   candidate into `guide/`, `reference/`, `internals/`, or `decisions+plans/`
   per the bead's tiering; `retro/`, `audits/`, `plans/` look like they'd
   become subdirectories under a `decisions+plans/` (or a new `retro/`) tier
   largely as-is (their internal structure already matches the target shape),
   while the flat `docs/*.md` files need per-file tier classification.

**Follow-up bead proposals:** none needed beyond `ttu` itself — it is already
scoped as "one pass, one PR, plus the generator change." The only fork point
worth flagging: if the front-matter-generator mechanism turns out to be
substantial (new Click command, new test fixtures, new render-all wiring), it
may be worth splitting *just that mechanism* into its own bead shared with
`3tl.9`, per this program's own grab-bag-splitting precedent (`6l6`).

---

## 7. polylogue-6l6 — Docs/theming/release-proof/control-plane grab-bag

Per-slice findings (bead explicitly authorizes splitting any slice that grows):

### (a) Docs theming pass — **NOT done; live drift confirmed, non-trivial**

`polylogue/ui/theme.py` (401 lines) is the canonical palette —
`PROVIDER_COLORS = {"claude-ai": "#d97757", "claude-code": "#d97757",
"chatgpt": "#10a37f", "gemini": "#4285f4", "codex": "#00bcd4", ...}`.

`devtools/pages_style.py` (134 lines, the docs-site CSS generator) has its
**own, independent, already-drifted** set of hardcoded hex values:
`--provider-claude-code: #D97706` (theme.py says `#d97757`),
`--provider-chatgpt: #10B981` (theme.py says `#10a37f`),
`--provider-gemini: #8B5CF6` (theme.py says `#4285f4`),
`--provider-codex: #3B82F6` (theme.py says `#00bcd4`).

Every single provider color differs between the two files. This is not a
"pass not yet done" situation in the trivial sense — it's an active,
user-visible inconsistency: the CLI/daemon-web-shell renders one set of
provider colors (from `theme.py`) and the docs site renders a *different* set
for the same providers. This slice deserves real weight, not a quick pass —
non-trivial because `pages_style.py` is plain CSS custom properties and
`theme.py` is Python `ProviderColor` dataclasses; the fix needs either (a) a
build step that emits CSS custom properties from `theme.py` at render time
(consistent with the "generated surface" discipline used everywhere else in
this repo), or (b) at minimum, updating the literal hex values to match by
hand with a lint/test asserting they can't drift again.

### (b) Release-proof check (docs claim-consistency half) — **spot-checked, one real find**

- `pyproject.toml` version `0.1.0` — `docs/installation.md` and `README.md` do
  not quote a version number anywhere I found (both correctly describe
  source/Nix-only install without a version claim) — **no drift found here**,
  matching the honest-by-omission finding in bead #2 above.
- `docs/release.md` correctly references `ghcr.io/sinity/polylogue` (matches
  `container.yml`'s actual `IMAGE_NAME` env var).
- One genuine drift candidate surfaced incidentally while investigating bead #2:
  CLAUDE.md's own "Coverage floor 90%" line was already flagged as stale by a
  *prior* audit fold-in on `polylogue-3tl.14`'s notes (fail_under=82 actual,
  tracked as gh-1793) — that's out of this bead's docs/ scope (it's the repo
  CLAUDE.md, not `docs/`) but worth cross-referencing so nobody re-discovers
  it as new.
- Full claim-by-claim sweep of every quoted command/version across all 68
  docs files was not completed in this pass (would need the `ttu` orphan
  inventory as a prerequisite checklist) — recommend sequencing this slice
  after `ttu`'s tiering lands so the sweep has a complete file list to work
  from, rather than duplicating `verify_doc_commands.py`'s existing coverage.

### (c) Control-plane doc currency — **thin surface, low risk, quick**

No `docs/control-plane.md` file exists. "Control-plane" as a *term* appears in
exactly one docstring (`polylogue/operations/specs.py:1`: *"Typed runtime
operation metadata shared across control-plane surfaces"*) and in
`devtools/command_catalog.py`'s `control_plane_argv()` helper (consumed by
`polylogue/scenarios/execution.py` in several places). There is no dedicated
"control-plane doc" to check for currency because the concept lives as code
comments/docstrings, not a standalone doc — `docs/devtools.md` is the closest
existing doc and is already covered by the existing doc-commands lint. This
slice is likely **near-zero work**: either confirm there's genuinely no
standalone control-plane doc to audit (this pass found none) and close it as
"no such surface exists," or if the operator meant `docs/devtools.md`
specifically, that's already gated by `verify_doc_commands.py`.

**Split recommendation:** slice (a) is the one that should become its own
dedicated bead — it's a real, live, multi-file inconsistency with a design
decision (generate-from-theme.py vs hand-sync-and-lint) that doesn't belong
buried in a grab-bag. Slices (b) and (c) are small enough to stay inside
`6l6` or close quickly; (c) in particular may close with zero code change if
"no control-plane doc exists" is an acceptable answer.

**Follow-up bead proposals:**
- *"Docs-site provider color palette drifted from polylogue/ui/theme.py —
  unify or generate"* — one-line scope: `pages_style.py`'s 4 provider hex
  values differ from `ui/theme.py`'s canonical `PROVIDER_COLORS` for every
  single provider; decide generate-vs-lint and fix.
- *"Full claim-by-claim docs-commands/version sweep, sequenced after
  polylogue-ttu tiering"* — one-line scope: complete slice (b)'s sweep using
  ttu's full-file inventory as the checklist, rather than duplicating it here.

---

## Cross-cutting observations

1. **Shared mechanism dependency across 3 beads:** `polylogue-3tl.9` (docs
   coverage lint), `polylogue-ttu` (docs IA/tiered generated index), and
   `polylogue-6l6` slice (b) (release-proof docs-claim sweep) all want the
   same underlying capability — a generated, queryable map of "what docs exist
   and what they claim to cover." Building `devtools/docs_surface.py`'s
   front-matter-driven generator once (as `ttu` proposes) and having `3tl.9`
   and `6l6`(b) consume it, rather than each bead inventing its own doc-tree
   walk, avoids three slightly-different implementations of the same set-diff.
2. **`polylogue-3tl.7`'s actual blocker is upstream of anything in its own
   design section:** no tag has ever been pushed, so 3 of its 4 target lanes
   (PyPI, Homebrew, GHCR-release-tagged) have literally nothing to smoke yet.
   The bead can and should still be *designed* now, but "green two consecutive
   weekly runs" is not achievable until a release-cutting decision is made
   separately — this dependency isn't stated in the bead itself.
3. **`polylogue-3tl.3`'s multi-model comparison is much less "spec-needed"
   than its readiness label suggests** — the hard data-availability part
   (open-model rows existing in the archive) is already true today; the
   actual remaining gaps are a coverage-refusal-gate verification, cost/cache
   columns gated on a separate epic (`f2qv`), and adversarial-hardening
   presentation work.
