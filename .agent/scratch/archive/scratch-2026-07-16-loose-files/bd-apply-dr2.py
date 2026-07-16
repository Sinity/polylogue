#!/usr/bin/env python3
"""Digest the three on-brief DR2 reports (D01/D02/D07 reruns) into beads. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    line = (r.stdout or r.stderr).strip().splitlines()
    print(("OK  " if r.returncode == 0 else "FAIL") + " bd " + " ".join(args[:2]), "|", line[0][:130] if line else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:300])


# ============ D02: local retrieval stack (DR2-02) ============
bd(
    "update",
    "polylogue-mhx.3",
    "--append-notes",
    "2026-07-06 D02 rerun landed (on-brief this time; preserved as corpus-gpt-pro-2026-07-06/"
    "DR2-02-local-llm-stack.md). Concrete protocol it recommends and this bead adopts: "
    "(1) index a HARD SLICE first, not 40GB — a few thousand sessions rich in fork/resume lineage, "
    "code-heavy, and Polish-English mixed; "
    "(2) four lanes over identical chunks: FTS/BM25 baseline, BGE-M3 dense-only, BGE-M3 dense+sparse "
    "hybrid, hybrid + Qwen3-Reranker-0.6B; "
    "(3) queries and positives come from the archive itself: lineage fork/resume pairs (query = child "
    "summary/title/first-prompt, positive = ancestor + fork-family siblings), best positives are "
    "semantically-near-lexically-far (Polish restatement of English design, refactor preserving intent "
    "not identifiers); hard negatives from near-time neighbors and same-day same-language sessions; "
    "(4) metrics: Recall@k k in 5/10/20/50, MRR@10, nDCG@10, family hit rate, and CROSS-LINGUAL hit "
    "rate (query language differs from positive language — critical for this corpus, BM25 degrades "
    "more in Polish per BEIR-PL); "
    "(5) adoption bar: dense/hybrid must beat FTS by roughly 10-15pp Recall@10 on lineage-family "
    "queries or win clearly on the named failure classes — otherwise FTS stays the source of truth; "
    "(6) escalation path if local loses: swap embedder to Qwen3-Embedding-4B (heavier, ~8GB), keep "
    "reranker, rerun the same benchmark before abandoning local.",
)

bd(
    "update",
    "polylogue-mhx.1",
    "--append-notes",
    "2026-07-06 D02 rerun model registry (candidates to encode, not hardcode): first-stage retrieval "
    "BAAI/bge-m3 (MIT, 1024-dim, 8192 tok, 100+ languages, dense+sparse+multi-vector in one model — "
    "recommended default); light dense-only alternative gte-multilingual-base (305M, explicit Polish "
    "MTEB results) or Qwen3-Embedding-0.6B (32K ctx, Apache-2.0); upgrade path Qwen3-Embedding-4B. "
    "Reranker: Qwen3-Reranker-0.6B (MTEB-R 65.80 / MMTEB-R 66.36 / MLDR 67.28 / MTEB-Code 73.42, "
    "clearly ahead of bge-reranker-v2-m3 and gte-multilingual-reranker-base in the published table). "
    "nomic-embed-text caveat: citable evidence is English-centric — weak fit for a Polish+English "
    "archive. Cloud baseline correction: Voyage-3 is officially a legacy generation; if a cloud lane "
    "survives the bake-off it should target voyage-4-lite (0.02 USD/Mtok) or voyage-4, not voyage-3.",
)

bd(
    "update",
    "polylogue-mhx.6",
    "--append-notes",
    "2026-07-06 cost datapoint from D02 rerun: a full re-embedding pass over the archive at "
    "voyage-3 pricing (0.06 USD/Mtok) is roughly 300-600 USD at a 5-10B token estimate; "
    "voyage-4-lite cuts that to ~100-200 USD; local BGE-M3/GTE marginal cost is time+electricity. "
    "Rebuild-cost asymmetry is the strongest argument for the local lane given the embeddings tier "
    "is rebuildable-by-design (delete + regenerate deterministically from chunks + recipe metadata: "
    "chunk hash, model id, dimension, dtype/quantization, recipe version).",
)

bd(
    "update",
    "polylogue-37t.5",
    "--append-notes",
    "2026-07-06 D02 rerun generator guidance: default local narrator = Qwen3 8B Q4_K_M (~5.2GB, "
    "Apache-2.0, 119 languages) run in NON-THINKING mode for classify/summarize/narrate-measures; "
    "Gemma 3 4B (~3.3GB) is the acceptable floor only when facts are pre-extracted into a structured "
    "object; DeepSeek-R1-Distill-7B rejected as default (reasoning-distill temperament: longer "
    "outputs, unnecessary elaboration for measure narration). Pipeline shape: retrieval -> "
    "deterministic reducer computes the metric/evidence bundle -> model verbalizes ONLY the computed "
    "object with cited evidence. The quality floor for honest narration is a constrained "
    "extract-then-verbalize pipeline, not a bigger reasoning model.",
)

# ============ D07: eval/RL export lane (DR2-07) ============
bd(
    "update",
    "polylogue-fs1.5",
    "--append-notes",
    "2026-07-06 D07 rerun landed (on-brief; preserved as corpus-gpt-pro-2026-07-06/"
    "DR2-07-rl-eval-environment.md). Design it settles and this bead adopts: "
    "(1) first export target = atropos-eval-jsonl profile (messages + scores is viewer-compatible "
    "immediately); NOT Harbor/Terminal-Bench (heavy task materialization), NOT SWE-bench (patch-"
    "centric), NOT OpenAI Evals (deprecation path late 2026), NOT ATIF (observability carrier, no "
    "reward semantics — stays an IMPORT format on fs1.2). "
    "(2) Three-way reward split is the core honesty contract: recorded_reward (derived from archived "
    "evidence, e.g. verify exit_code=0), replay_spec (git SHA, workspace snapshot ref, env manifest, "
    "network policy, timeouts), checkable_reward (null until replay actually reruns the verify "
    "command). Never conflate recorded with checkable. "
    "(3) Field mapping keys on existing evidence — tool_result_is_error, tool_result_exit_code, "
    "verify command text+cwd, git SHA, user.db corrections as session-level weak supervision (do NOT "
    "invent step-level labels), keystone evidence_ref back to the archive. NO fabricated tokens/"
    "masks/logprobs — that lane exists only after a replay substrate exists. "
    "(4) Highest-value slice: CI-passing sessions with an explicit verify command + stable git SHA — "
    "the verify command IS the seed of a rerunnable reward function.",
)

bd(
    "update",
    "polylogue-fs1.10",
    "--append-notes",
    "2026-07-06 consensus (D07 rerun + gpt-pro feedback + DR1 reports agree): define the INTERNAL "
    "schema first — SpecCard + Trajectory + EvidenceRefs — then write adapters outward (Atropos, "
    "Verifiers, Harbor/Terminal-Bench, SWE-style). Do not let any single external format become the "
    "identity of the eval lane.",
)

# ============ D01: positioning (DR2-01) ============
bd(
    "update",
    "polylogue-3tl",
    "--append-notes",
    "2026-07-06 D01 rerun landed (on-brief; preserved as corpus-gpt-pro-2026-07-06/"
    "DR2-01-competitive-landscape.md). Positioning decision it supports: "
    "PRIMARY category claim = flight recorder: 'Polylogue is the local flight recorder for AI work — "
    "a cross-provider system of record where every metric resolves to raw bytes.' Each clause "
    "excludes a crowded incumbent family: local/offline excludes cloud dashboards (LangSmith/Langfuse/"
    "Helicone/Phoenix/Weave — all live-instrumentation-first); cross-provider excludes single-platform "
    "exporters; system-of-record excludes ephemeral tracing UIs and soft 'memory' branding (Limitless); "
    "bytes-resolution excludes dashboard-slop. Nearest neighbors: simonw llm+Datasette (substrate "
    "spirit, but CLI logger not system of record) and W and B HiveMind (closest product motion: daemon "
    "captures coding-agent sessions incl. Claude Code/Cursor imports — but cloud, team-dashboard, "
    "coding-only; watch it). Honesty-benchmark framing DEMOTED to secondary launch artifact: as an "
    "umbrella it reads accusatory and narrows to public verification. Target communities: AI Engineer/"
    "Latent Space, simonw/local-first crowd, coding-agent power users. Anti-goal: do not launch under "
    "observability, memory, or evals labels — each invites the wrong comparison set.",
)

print("--- batch 2 done")
