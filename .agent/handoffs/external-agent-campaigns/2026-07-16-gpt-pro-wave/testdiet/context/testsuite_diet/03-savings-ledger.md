---
created: 2026-07-16
purpose: Maintain credible test and production LOC savings estimates without double counting
status: active
project: polylogue
---

# Savings ledger

## Baseline and counting rule

2026-07-16 baseline, counting nonblank physical Python lines with
`rg -n '\S' <scope> -g '*.py' | wc -l`:

| Scope | Files | Nonblank LOC | Physical LOC |
| --- | ---: | ---: | ---: |
| `tests/` | 941 | 275,647 | 324,166 |
| `polylogue/` | 994 | 257,455 | 294,722 |
| `devtools/` | 144 | 47,940 | 53,851 |

Forecasts use nonblank LOC. Production (`polylogue/` + `devtools/`) is reported
separately from tests. Generated docs/YAML are tracked in area packets when
removed but excluded from these Python totals.

The workload-profile, scale-tier, canary, receipt, testmon-mutation, and xdist
witness work in `polylogue-1xc.14(.1)` and `polylogue-b054.1.1.3`–`.5` is an
upstream capability baseline, not Diet savings. Its added or removed LOC is not
entered here. A later Diet row records only the additional replacement and
subtraction performed by that Diet cluster, preventing double counting.

`net savings = gross old LOC removed - replacement LOC added`.

Gross candidate population is not savings. A 5,000-line cluster may yield only
1,000 net lines after retaining unique cases and adding stronger laws.

## Two different horizons

The figures below must not be read as an estimate of the final redesigned
suite. They cover only clusters mapped deeply enough to name a replacement and
avoid obvious double counting.

- **Mapped near-term band:** evidence-supported savings from the packets
  currently forecast below.
- **End-state redesign:** the possible size of a suite rebuilt around
  schema-generated corpora, independent models, properties/state machines,
  composition routes, work laws, and rewrite-native MCP/web tests. This is not
  estimated yet.

Net savings already includes expanded coverage:

`net savings = old tests/helpers removed - all replacement tests/harness added`.

For example, deleting 100k lines and adding 25k lines of more expressive
composition/property/harness code is 75k net savings, not 100k. Coverage gains
are tracked separately through obligations, real routes, historical/mutation
sensitivity, generated cases/state sequences, and work bounds—not by pretending
that lower LOC alone is success.

## Historical mapped hypotheses

The bands below predate realized cluster replacement measurements. They remain
useful as prioritization context, but they are not booked savings and must not
be rolled into a suite-size estimate.

| Scope | Cluster | Population / evidence | Net forecast | Confidence | Additive? |
| --- | --- | ---: | ---: | --- | --- |
| tests | retired temporal conductor | 142 physical / 118 nonblank test lines | ~118 | confirmed | yes |
| tests | inert lifecycle/growth harness islands | 322 physical / 259 nonblank incl. growth self-test | ~259 | confirmed consumer trace; deletion still needs focused verify | yes |
| tests | closed-loop coverage/readiness machinery | dedicated and mixed devtools tests | 0.4–1.0k | medium | yes, after function adjudication |
| tests | direct spelling/topology/private-name fossils | census + narrated examples | 0.2–0.6k | medium | **no**; overlaps owning clusters |
| tests | query composition | ~8.7k relevant nonblank LOC | 2.2–3.9k | medium | yes after dominance proof |
| tests | status composition | ~4.9k relevant nonblank LOC | 1.2–2.2k | low–medium | yes after transition model |
| tests | facade contracts | ~4.9k relevant nonblank LOC | 1.2–2.2k | low–medium | yes after authority/route split |
| tests | current web reader | source-spelling-heavy UI cluster | unknown until rewrite | rewrite boundary | excluded |
| tests | current MCP | large mixed suite | unknown until rewrite | rewrite boundary | excluded |
| production | retired temporal conductor | 255 lines | ~255 | confirmed | yes |
| production | closure/scenario coverage claims | 331 whole-file LOC plus consumers | 0.3–0.5k | medium-high | yes if navigation survives without proof claims |
| production | manifest mirrors + quality-reference sections | mixed 1,286-line source surface plus models/commands | 0.3–0.9k | medium | yes after function split |
| production | live dev loop | 3,831 lines | 0 | confirmed keep | no |

The historical mapped planning band is roughly **8–13k net test LOC** (3–5%).
It is deliberately narrow because cross-cutting census hits overlap
query/status/facade clusters and MCP/web are excluded. The current production
lead is roughly **1–2k LOC**, including 255 confirmed; it is not yet an
adjudicated total.

The 259-line inert harness lead fits inside the existing 8–13k planning band;
do not add it on top until owning-cluster overlap is resolved. Corpus/scale
consolidation has no forecast yet because the stronger artifact builder's size
is unknown.

## End-state size scenarios—not forecasts

These rows make the arithmetic explicit without presenting unsupported savings
as a finding:

| Final suite scenario | Net reduction from 275,647 | Meaning |
| --- | ---: | --- |
| 15% smaller | ~41k | meaningful consolidation, still mostly current test shape |
| 25% smaller | ~69k | broad cluster replacement across several major areas |
| one-third smaller | ~92k | strong properties/state/composition replace many case matrices |
| 2× smaller | ~138k | half the current net LOC; requires dominance across most large areas |
| 3× smaller | ~184k | about 92k LOC remain; requires extraordinary redundancy evidence |

**One-third smaller is a hypothesis, not a current estimate.** The
unforecast storage, source, and daemon areas alone contain about 104k nonblank
test LOC. CLI contains another 38k, only part of which is represented in the
query forecast; core has 22.6k, devtools 21.0k, and MCP 10.1k. The current
MCP/web implementations are rewrite boundaries. Expressive generated laws can
cover more combinations and states with fewer lines, so expanded behavioral
coverage is compatible with a substantially smaller suite.

A **few-times reduction** is not supportable from present evidence. It would
require showing that most narrow tests across these areas are dominated rather
than uniquely protecting provider quirks, durable migrations, security/error
boundaries, concurrency, and diagnostics. Production simplification and the
planned rewrites may make it possible, but test cleverness alone does not prove
that.

## How to estimate the end state credibly

Audit representative behavior clusters in the large areas, then extrapolate by
cluster—not by mock/census hit:

1. generate fresh per-test coverage contexts for one storage, source, daemon,
   CLI, and devtools cluster;
2. map unique versus overlapping arcs and current runtime/fixture cost;
3. implement the stronger independent property/state/composition law;
4. run historical and focused mutation sensitivity;
5. delete dominated tests and measure gross removal, replacement LOC, net LOC,
   runtime, and obligation change;
6. use those realized ratios only for structurally similar remaining clusters.

Low/base/high projections remain disabled until all five representative
clusters have landed and have realized rows below:

1. seeded artifact integrity over realized workload canaries;
2. exact query selection and bounded work;
3. convergence debt across restart/retry/quiescence;
4. incremental-versus-rebuild storage equivalence;
5. devtools verification subtraction.

After those five clusters land, begin projections only inside a structurally
similar stratum. Parser/wire compatibility, pure properties, SQLite
state/recovery, cross-surface composition, process/integration, and
verification/tooling need separate realized analogs before extrapolation.
Five heterogeneous results do not justify one suite-wide replacement ratio.
Until a stratum has evidence, leave it unforecast; the one-third reduction
remains a hypothesis and the 8–13k band remains historical context.

## Overlap rules

- Every candidate belongs to one owning cluster. Cross-cutting labels such as
  `source-spelling` and `mock-forwarding` explain why, but are not added again.
- MCP and web-reader candidates remain excluded until rewrite-native
  replacement sizes exist.
- A production verifier and its tests are separate scopes; never blend them.
- Forecast bands shrink only after a file/function inventory exists. They grow
  only for a new non-overlapping packet with direct evidence.

## Realized ledger

Append one row per merged cluster:

| Date/PR | Cluster | Old LOC removed | Replacement LOC | Net test LOC | Old/new runtime | Fixture builds | Production routes | Sensitivity witness | Residual |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| — | — | — | — | — | — | — | — | — | — |

No row is added for a dossier, proposed deletion, census population, or dry
run. A row requires the merged diff and its verification receipt.
