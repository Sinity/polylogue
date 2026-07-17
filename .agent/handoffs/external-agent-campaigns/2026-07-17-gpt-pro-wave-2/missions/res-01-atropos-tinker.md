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
