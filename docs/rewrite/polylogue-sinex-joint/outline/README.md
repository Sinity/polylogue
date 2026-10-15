# Where the two outlines meet

Polylogue's outline (`polylogue-spec/outline/`) and Sinex's
(`sinex-spec/outline/`) each describe one finished system. This folder
describes only the boundary between them: every interface the joint
decisions define, and the components on each side that own it.
`boundary.toml` holds the entries; `check_boundary.py` (flake check
`joint-outline`) checks them.

## The boundary in one page

Polylogue keeps the interpretation of AI sessions; Sinex keeps materials,
events and the commit log (joint #1). Everything that crosses goes through
one Polylogue component, `sinex.adapter`, and three Sinex ones,
`protocol.producer`, `daemon.ingress` and `daemon.publisher`, except the
shared crates and conventions.

| Interface | Direction | Polylogue | Sinex | Decisions |
|---|---|---|---|---|
| material transfer | P → S | `sinex.adapter`, `store.inputs` | `protocol.producer`, `daemon.ingress`, `store.content` | 3, 8, 11, 27 |
| framed revisions | P → S | `sinex.adapter`, `model.export` | `protocol.producer`, `daemon.ingress`, `store.admission`, `daemon.writer` | 5, 26 |
| `have` | P asks S | `sinex.adapter` | `protocol.producer`, `daemon.api` | 8, 26 |
| confirmed stream | S → P | `sinex.adapter` | `daemon.publisher` | 8 |
| material read-back (primary) | S → P | `store.inputs`, `blobs.files`, `engine.loop` | `store.content`, `daemon.privacy`, `daemon.api` | 7, 27 |
| resolved objects | P → S | `model.merge`, `model.identity`, `model.export`, `sinex.adapter` | `store.admission`, `automata.reconcilers` | 1, 3, 4, 20, 28 |
| lineage and turns | P → S | `model.lineage`, `model.types` | `model.events`, `store.admission` | 5, 22 |
| annotations and claims | both, same rules | `store.notes`, `engine.notes_logic` | `analysis.runs`, `ctl.cli` | 16 |
| read-as sentences | both, same rule | `query.grammar` | `daemon.api` | 17 |
| machine section of the brief | P asks S | `engine.briefings`, `bin.hook` | `ctl.brief`, `daemon.api` | 18 |
| purge and excision | P orders, S executes | `store.excision`, `sinex.adapter` | `store.archive`, `daemon.privacy` | 2, 6 |
| `hashed-blobs` | shared crate | `blobs.files`, `store.inputs` | `store.content`, `daemon.privacy` | 9, 11, 12, 14, 19, 23, 24 |
| logical day | shared rule | `model.time` | `analysis.descriptors` | 15 |
| visual language | shared tokens | `render.formats`, `bin.http` | `ctl.reading-surface` | 21 |
| pacing | shared signal | `engine.loop`, `engine.collector` | `daemon.writer` | 25 |

Decisions 10 (sequencing) and 13 (xtask core) have no interface and are
listed as internal-only with their reasons.

The three wire types, `Have`, `FramedBatch` and `MaterialFrame`, are written
in both skeletons (`polylogue-spec/skeleton/crates/sinex`,
`sinex-spec/skeleton/crate/sinex-protocol`). The check compares their
variant and field names, not field types, since the skeletons' types are
placeholders. The joint text decides names: topic 03, #26, #27. On
2026-10-12 they agree.

## What the check enforces

- Every component named exists in its project's outline.
- Every decision row is placed at an entry or listed as internal-only with a
  reason.
- Every entry names a component on both sides, unless marked one-sided.
- Every cited topic heading or numbered point exists.
- The shared types' names are equal on both sides.

Probes run when it was written: a renamed field, a misspelled component, an
unplaced decision and a dead topic reference each failed it.

## Gaps found

**Polylogue, fixed here.** Joint #27's same-host placement and the pending
area of unconfirmed inputs were in no component. `sinex.adapter` now owns
both, with a placement interface and `decisions/sinex.md#Modes` cited.

**Sinex, fixed by the Sinex session.**
- Adopting placed material (#27) is now part of `daemon.ingress`: verify the BLAKE3 against the announcement, clone the bytes into a sinex-owned file (decision 27, amended 2026-10-12), commit the registry row, move that file into place, delete the incoming file and what is unverified.
- The per-directory machine section (#18) is a new component, `daemon.brief-section` (slice 6): precomputed per working directory, refreshed as events arrive, served from the socket as a cache read. The `brief-machine-section` boundary now names it instead of `ctl.brief`, which is the operator's CLI brief.

**Polylogue, open.** The outline has no xtask component, so #13 is
internal-only. If both projects come to run the same xtask core, it becomes a
shared-crate entry here.
