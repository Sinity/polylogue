Title: "Resume-candidate quality: fix dead-path Jaccard anti-selection and build the ranking evaluation harness (v1vo)"

Result ZIP: `perf-04-resume-quality-r01.zip`

## Mission

Implement bead `polylogue-v1vo` (P2, read its full record — it carries a
measured live finding from 2026-07-17): `find_resume_candidates`
(`polylogue/insights/resume.py`, `file_overlap` scoring ~line 652) computes
exact-path Jaccard between the caller's current `recent_files` and each
candidate session's historical `file_paths_touched`, weighted 0.25 in the
resume ranking that feeds the SessionStart context preamble and MCP resume
tools. Measured against the live corpus (4,264 polylogue-repo sessions,
27,785 file-touch occurrences): repo layout refactors kill exact-path
overlap, so the ranking ANTI-SELECTS exactly the refactor-ancestor sessions
an operator most wants to resume — sessions whose files have since moved
score zero.

Fix per the bead's decided design (scoring-time, profiles stay honest):

1. In `_profile_paths`/file_overlap scoring: (a) partition candidate paths
   into resolvable vs dead against the resume repo root (existence +
   file-to-package correction); (b) match dead paths by parent-directory
   prefix against the caller's recent_files directories (59% recovery,
   already measured — reproduce the measurement); (c) drop
   still-unresolvable paths from the Jaccard union so they stop deflating
   scores; (d) emit an `overlap_basis` breakdown (exact/dir/dead-excluded)
   so the ranking is explainable in resume output.
2. **Ranking evaluation harness** (the durable half): a small offline
   evaluator that scores ranking quality over synthetic-and-sampled
   scenarios — given a "current work" fixture and a candidate pool with
   known-correct resume targets (construct from lineage: the true parent/
   sibling sessions of a fork family are ground truth), report
   hit@1/hit@3/MRR before vs after the fix. This harness is what makes
   every FUTURE ranking change reviewable instead of vibes.
3. Tests from the bead's AC: 100%-dead-path session with shared directories
   gets nonzero file_overlap; two candidates identical except extra dead
   paths score equally; existing exact-overlap behavior unchanged (no
   regression on resolvable paths); overlap_basis rendered in the resume
   brief surfaces (`get_resume_brief` MCP tool + CLI `continue` path — find
   both consumers and update their projections).

## Constraints

- Scoring change only — do NOT rewrite stored session profiles (evidence
  stays as-captured; adaptation happens at query time). No schema changes.
- The resume surface feeds agent context (SessionStart hook) — keep output
  shapes backward-stable except the additive overlap_basis field; list
  every consumer touched (grep get_resume_brief / find_resume_candidates).
- Weight retuning (the 0.25) is OUT of scope unless the evaluation harness
  shows an obviously dominated setting — if so, propose, don't silently
  change.

## Deliverable emphasis

HANDOFF.md: the fix mechanism, measured before/after ranking metrics from
the harness (on synthetic + snapshot-derived fixtures; label live-corpus
numbers unverified), consumer-surface updates, and harness usage docs for
future ranking work.
