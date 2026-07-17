# 108. polylogue-mhx.3 — Retrieval quality eval lane: measure FTS vs vector vs hybrid before believing any of them

Priority/type/status: **P2 / task / open**. Lane: **09-embeddings-retrieval**. Release: **J-embeddings-retrieval**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Every retrieval default (auto lane resolves lexical; hybrid RRF constants; --semantic promotion) was chosen by taste, not measurement. Before the substrate grows recall legs (context compilation) and storage optimizations (quantization), build the eval that says which lane actually finds the right sessions for realistic operator queries — the same evidence-first discipline as the heuristics benchmark (9e5.9).

## Existing design note

devtools bench retrieval, campaign run/compare pattern: (1) Labeled set: ~50-100 (query -> expected sessions/messages) pairs; bootstrap from real usage — queries the operator/agents ran followed by which session they actually opened (the archive records its own MCP/CLI usage; affordance-usage machinery can mine query->open chains), then hand-verify. Seeded-corpus subset for the public/CI-runnable variant, live-archive set for the real decision. (2) Contestants: FTS only, vector only, hybrid RRF (current constants), hybrid with 2-3 alternative K constants, and optionally rerank (cross-encoder through the LiteLLM gateway) as a stretch arm. (3) Metrics: recall@5/@10, MRR, plus latency and $ per 1k queries — the decision is quality-per-cost, not quality alone. (4) Output: campaign artifact under .local/, compare mode against baseline; the verdict updates the default retrieval_lane resolution and docs/search.md with citations to the artifact. (5) Reuse: the same labeled set gates quantization (emb-efficiency) — quality deltas from int8/matryoshka must stay within a stated tolerance measured HERE.

## Acceptance criteria

1. `devtools bench retrieval` exists and runs FTS-only, vector-only, hybrid-RRF (current constants), and 2-3 alternative-K hybrid arms (rerank arm optional/stretch) over a committed labeled set of >=50 (query -> expected session/message id) pairs; a public seeded-corpus subset is CI-runnable and a live-archive variant drives the real decision. 2. It emits recall@5, recall@10, MRR, plus p50/p95 latency and $/1k-queries per arm to a `.local/` campaign artifact; `compare` mode diffs a candidate against a stored baseline and returns non-zero on a regression beyond a stated tolerance. 3. The winning lane is written back as the default retrieval_lane resolution AND cited (artifact path + metric deltas) in docs/search.md. 4. The same labeled set is importable by mhx.6 so quantization quality tolerance is measured against it. Verify: `devtools bench retrieval run` produces the artifact; `devtools bench retrieval compare baseline.json candidate.json` exits non-zero on a seeded regression; the docs/search.md citation resolves to the artifact.

## Static mechanism / likely defect

Issue description localizes the mechanism: Every retrieval default (auto lane resolves lexical; hybrid RRF constants; --semantic promotion) was chosen by taste, not measurement. Before the substrate grows recall legs (context compilation) and storage optimizations (quantization), build the eval that says which lane actually finds the right sessions for realistic operator queries — the same evidence-first discipline as the heuristics benchmark (9e5.9). Design direction: devtools bench retrieval, campaign run/compare pattern: (1) Labeled set: ~50-100 (query -> expected sessions/messages) pairs; bootstrap from real usage — queries the operator/agents ran followed by which session they actually opened (the archive records its own MCP/CLI usage; affordance-usage machinery can mine query->open chains), then hand-verify. Seeded-corpus subset for the public/CI-runnable variant, live-archi…

## Source anchors to inspect first

- `polylogue/api/embeddings.py` — Public embedding API seam.
- `polylogue/cli/commands/embed.py` — Operator embedding command and dry-run planner hooks.
- `polylogue/archive/query/retrieval.py` — Retrieval composition seam for FTS/vector/hybrid.
- `polylogue/archive/query/retrieval_search.py` — Search/retrieval runtime implementation.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. devtools bench retrieval, campaign run/compare pattern: (1) Labeled set: ~50-100 (query -> expected sessions/messages) pairs
2. bootstrap from real usage — queries the operator/agents ran followed by which session they actually opened (the archive records its own MCP/CLI usage
3. affordance-usage machinery can mine query->open chains), then hand-verify.
4. Seeded-corpus subset for the public/CI-runnable variant, live-archive set for the real decision.
5. (2) Contestants: FTS only, vector only, hybrid RRF (current constants), hybrid with 2-3 alternative K constants, and optionally rerank (cross-encoder through the LiteLLM gateway) as a stretch arm.
6. (3) Metrics: recall@5/@10, MRR, plus latency and $ per 1k queries — the decision is quality-per-cost, not quality alone.
7. (4) Output: campaign artifact under .local/, compare mode against baseline

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: `devtools bench retrieval` exists and runs FTS-only, vector-only, hybrid-RRF (current constants), and 2-3 alternative-K hybrid arms (rerank arm optional/stretch) over a committed labeled set of >=50 (query -> expected session/message id) pairs
- Acceptance proof: a public seeded-corpus subset is CI-runnable and a live-archive variant drives the real decision.
- Acceptance proof: 2.
- Acceptance proof: It emits recall@5, recall@10, MRR, plus p50/p95 latency and $/1k-queries per arm to a `.local/` campaign artifact
- Acceptance proof: `compare` mode diffs a candidate against a stored baseline and returns non-zero on a regression beyond a stated tolerance.
- Acceptance proof: 3.
- Acceptance proof: The winning lane is written back as the default retrieval_lane resolution AND cited (artifact path + metric deltas) in docs/search.md.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
