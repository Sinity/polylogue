Title: "Deep research: Atropos, tinker-atropos, and NeMo-Relay trace formats — the exact current state for Polylogue's eval/RL export bridge"

Result ZIP: none — deep-research memo, deliver as a single Markdown response
(the orchestrator archives it). Use the deep-research contract.

## Mission

Polylogue is building a Hermes bridge: ingest Hermes Agent `state.db` +
NeMo-Relay observer traces (ATIF/ATOF), produce forensic reports, and
export eval/RL-shaped trajectories. The strategic thesis: production agent
traces should become debugging evidence AND training/eval material. We need
the current, cited ground truth on the target formats and ecosystems before
committing the export layer. Web research with primary sources only
(repos, release notes, docs, code); every load-bearing claim cited and
dated; separate evidence from inference.

Questions:

1. **Atropos** (github.com/NousResearch/atropos): current repo state —
   last merged activity, maintainer attention, the community-environments
   path and its actual merge cadence in 2026; the ScoredDataGroup /
   trajectory data shapes it consumes TODAY (read the code, cite files);
   the `jsonl2html.py` viewer's expected input; whether the May-2025-era
   format assumptions in older notes still hold.
2. **tinker-atropos** (NousResearch/tinker-atropos): what it actually does
   now, its integration surface with Hermes Agent, and what a
   trajectory→LoRA proof-of-concept minimally requires (models, compute,
   formats). Is the "import history → train a small delta" story real with
   current code, and what are its exact inputs?
3. **NeMo-Relay / ATIF/ATOF**: the trace format's current version and
   schema (ATIF-v1.7 was documented — is it still current? cite the spec
   location); how Hermes Agent's observability plugin emits it (config
   flags, file layout, rotation semantics); any format changes in recent
   Hermes releases (check hermes-agent release notes through the newest
   release).
4. **Hermes state.db**: schema drift risk — have recent Hermes releases
   changed `hermes_state.py`-relevant tables (sessions/messages/tool
   calls/token counters/parent sessions/compaction)? Cite the migration
   files or schema code at the current release tag.
5. **Adjacent prior art**: NVIDIA Polar-style trace-to-RL pipelines and
   anything new (since 2026-05) that converts coding-agent archives into
   eval/training data — is anyone else mining HISTORICAL multi-provider
   archives (vs live rollouts)? This calibrates the novelty claim we make
   externally.

## Deliverable

A memo with: per-question findings with citations and dates; a "format
contract" appendix (exact field tables for ATIF, ATOF, ScoredDataGroup,
tinker inputs as of today); a risk table (what could invalidate our export
layer and how to detect it); and a recommended minimal export target
(which ONE format Polylogue should emit first, and why). Flag every
unverifiable/ambiguous point explicitly rather than smoothing it.


---

## Research contract

Use Deep Research, not ordinary implementation mode. Prefer current primary and
official sources; record direct URLs, publication/update dates, access date,
and the exact claim each source supports. Distinguish standards, documented
provider policy, measured behavior, informed inference, and proposal. Search
for counterevidence and incompatible constraints rather than writing a survey
that merely confirms the mission premise.

A current Polylogue project-state archive may be attached for product context.
Inspect relevant source and Beads so the research resolves concrete project
decisions, but do not return speculative patches. Map each conclusion to the
named Polylogue decision/Bead, state what should change, what should not change,
and what local experiment would falsify the recommendation.

Present a substantive, self-contained research report with conclusions,
rationale, source-by-source support, counterevidence, limitations, missing
evidence, Polylogue decision mappings, and the likely value of another
iteration. It must remain useful to a reader who has not opened the attached
project-state archive.

Do not perform an adversarial review unless explicitly requested. On an
ordinary **iterate/continue** request, extend the strongest unresolved research
branch and return a revised complete report. On an explicit **adversarial
review** request, try to falsify the prior memo with counterevidence, later or
more authoritative sources, incompatible policies/standards, hidden product
assumptions, and experiments that would overturn its recommendations. Repair
legitimate findings and report what changed, what remains uncertain, and
whether another pass is worthwhile.
