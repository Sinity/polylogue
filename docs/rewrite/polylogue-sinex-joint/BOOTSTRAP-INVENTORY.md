# Replacement inventory and bootstrap acceptance

This packet implements realization-plan artifact 2: what the current skeletons prove, what must change before source replacement, and how to check that starting tree. It is not authorization to replace a source repository or move its state. [The realization plan](REALIZATION-PLAN.md) owns milestones and roles; each product's workspace/model/storage specifications own semantics.

## Audit of the current composition trees

Observed at spec revision `c9308e7` on 2026-10-13:

| Property | Polylogue | Sinex |
|---|---|---|
| Cargo packages / Rust files | 9 / 9, including optional Sinex adapter | 13 / 13, including xtask and its macro package |
| Literal `todo!()` occurrences | 137 | 36 |
| Executable entrypoints | One CLI main, which reaches a stub socket client | None: daemon, CLI, session producer and xtask are libraries |
| External libraries | None; database, JSON, UUID, time and async behavior represented by placeholders | None; `Tx` is an empty public trait rather than a role-bound database transaction |
| Type comparison | 3 shared type pairs; 10 function names/parameter counts; 91 named slice paths | 27 shared type pairs |
| Runtime acceptance | No actual daemon, collector/store or client loop | No actual daemon, admission or service loop |

These counts locate work; they are not completeness percentages or growth limits. The type checkers compare overlapping declarations and selected shapes. They neither require every normative type nor compare all field/parameter/return types or behavior. Dependency/component checks prove the declared composition; Cargo proves it compiles with its placeholders.

Specific defects and limits:

- Sinex's checker originally accepted duplicate component markers, a `daemon.api` marker moved into `sinex-model`, and an undeclared external dependency. All three counterexamples returned success with “40/40 components placed.” The corrected checker rejects them and has discriminating mutation cases in its existing check route. This still does not enforce database permissions or runtime ownership.
- Polylogue's `cli::run` discards `client.send` errors and returns zero after a successful connection. `mcp::call` also discards its reply; the daemon covers only a Query request and panics elsewhere. These are composition sketches, not supported surfaces. The real response/exit mapping must preserve empty, refused, degraded, unavailable and transport outcomes before dispatch supports a command.
- Sinex's public `Tx`/`admission::admit` does not encode the private WriterPool/ReadPool boundary in `plan/workspace.md`. A caller can implement the empty trait. Adding a sealed dummy token would hide the hole without establishing a real pool/role boundary; bootstrap must use the actual connection ownership and database grants.
- The local `hashed-blobs` skeleton is a second signature copy, not the working joint crate. Choose one real crate owner and tested versioned pin; do not ship the stub alongside an independent file layer.
- Neither skeleton includes usable xtask commands, packaging/service wiring, applied schema, request transport or registered supported capabilities. Those need explicit implementation and acceptance, not removal of `todo!()` by fake defaults.

## Source-tree disposition

Read-only local source observations on 2026-10-13: Polylogue `17c20e5c6dac1420e67dd6a9b37977fd913307bc`, Sinex `85e6a98df9e2c0c032c2ba78d53eb35bb81790a2`; both checkouts clean. These are audit references, not replacement bases. Reconcile hosted/default heads, other worktrees, WIP and in-flight work again at execution.

| Category | Polylogue replacement | Sinex replacement |
|---|---|---|
| Product tree | Target `crates/`, actual binary, optional backend outside default build; old `polylogue/` Python tree absent | Target `crate/`, actual daemon/CLI binaries; producer binaries arrive with their supported capture slice; old crate tree absent except explicitly justified copied primitives |
| Build/locking | New Cargo workspace/lock/toolchain and Nix build; retire `pyproject.toml`, `uv.lock`, Python devtools | New Cargo lock/toolchain and Nix build; target edges/harness, not copied old xtask machinery |
| Developer/runtime operations | Narrow xtask and declared AgentCTL operations; own start/stop of disposable services | Same ownership; actual PostgreSQL/NATS isolation and role setup |
| Instructions and CI | Revise source AGENTS, contribution/testing docs, hooks, workflow/release inputs and `.agentctl` together | Revise the equivalent source instructions, hooks and review/check routes together |
| Legal/history | Preserve license, attribution and Git history; review generated release/changelog inputs | Same |
| Examples/tests | Synthetic Morrow and selected neutral provider/failure cases; independent expected facts | Synthetic Kestrel and selected neutral parser/transaction/failure cases |
| Spec/decisions | One current home under `docs/rewrite/`, with explicit pins to shared contracts; inbox copy retired as authority after transfer | One current docs home, same rule; carry audit pointers, not all historical reports into instructions |
| Optional later clients | Browser extension, Python client, web UI/native host code only when a slice supports it; preserve old knowledge through refs | Session producer, NixOS module and later capture/surfaces allocated by their actual slices |
| Task/local/private state | Tested Beads routing and ignore rules only; no database, captures, transcripts or private WIP in tracked tree | Same; `.beads` is routing, not permission to reset the old external task store |

Inventory every actual top-level path against these categories when preparing the PR. Existing `scripts/`, `experiments/`, `demo/`, `schemas/`, `.agent/`, old CI configurations and generated docs are not automatically copied or deleted on a name heuristic. Retain only a named consumer, required legal/history material or independently useful test/behavior; preserve the rest through reviewed refs/private bundles before the old tree is removed.

A replacement also changes package consumers outside the repository. Inspect Sinnix packaging, service units, MCP/client installs, hooks, release automation and shared-crate consumers. Pin existing deployments to the preserved implementation until separate cutover. A fresh source HEAD must not cause an automatic update to install a stub product.

Consumer source inspection on 2026-10-15 found these concrete connections. Refresh their revisions and paths at replacement time; this is not installed-state verification:

| Consumer | Predecessor dependency | Replacement disposition |
|---|---|---|
| Sinnix packages, services and agent clients | Polylogue package binaries, `api-python` and Home Manager module; Sinex packages, overlay and NixOS module | Keep the existing consumers on an immutable predecessor input until their successor interfaces and operational cutover are ready |
| Lynchpin package | Builds Polylogue as a Python dependency from `pyproject.toml` and imports `nix/aiosqlite.nix`; Sinnix makes its `polylogueSrc` input follow the same Polylogue input | Preserve that dependency independently or replace its actual callers before changing the shared input; a working new CLI does not replace an importable Python API |
| Sinnix Agent Gateway build | Generates and checks owner contracts from the Polylogue source input | Pin the predecessor contract source or migrate the contract reader and generated inputs together; do not disable its check to accept a new tree |

A lockfile pins a particular build, but an input still targeting `master` will select the replacement on a later update. Before replacement, make predecessor pins explicit for these transitive consumers and inspect local override paths too. Do not preserve obsolete product files in the new workspace solely to satisfy an external caller: migrate that caller or bind it to the preserved implementation. Source replacement must leave both ordinary locked builds and deliberate updates with a truthful, supported dependency choice.

## Transition review

Use this inventory only after the finite retained-work reconciliation required by the realization plan. Bind the replacement candidate to the verified finished default head, preserved ref and restore-tested bundle. Review the exact path disposition and external consumer pins before removing the predecessor tree. A source-preservation receipt protects recoverability; completion evidence settles each started intention separately.

Review the candidate from a fresh checkout with isolated state and its packaged entrypoints. Unsupported signatures stay outside enabled registries, and installed consumers remain pinned to the preserved release. The runtime evaluator runs the minimum foundation cases below on the fixed candidate; the integrator records unmet criteria before any replacement PR is called ready. Publishing the replacement, importing a new task corpus and activating a product are separate outcomes with their own routing and state evidence.

## Minimum executable bootstrap

The starting PR need not implement all future features. It must implement the following foundation honestly; missing items stay open.

| Boundary | Required starting behavior | Discriminating acceptance |
|---|---|---|
| Model and identity | Real domain types, canonical encodings, time quality, provenance/record accounting and structured errors; no placeholder strings standing in for unsupported contracts | Invalid/missing provenance or namespace confusion fails; unknown time remains unknown; identity cases use independently stated bytes/keys |
| Build graph | Actual dependencies/toolchain/lock, target allowed edges and pure-core isolation | Fresh checkout builds; deliberately forbidden edge fails; pure tests run without a database or daemon |
| Storage ownership | Polylogue index writes accessible only to writer; Sinex private writer connection and restricted read-model/analysis roles | Wrong caller cannot acquire write capability; actual database-role write denied; permitted writer route succeeds |
| Shared bytes | One real `hashed-blobs` implementation with tested range/durability contracts | Round-trip and corrupt/bad-range case through the selected crate; selected version recorded |
| Schema | Schema creation through the declared owner and version/change mechanism, isolated state only | Empty apply and supported upgrade agree by definitions; intentionally incompatible schema is rejected |
| Executable entrypoints | Real binaries and narrow check/build/run entrypoints, useful version/help/validation; no supported command/source reaches a stub | Run packaged binaries; unsupported operation returns the declared non-success outcome, no panic or fake empty result |
| Support registration | Explicit enabled capability set tied to implemented paths; later composition signatures excluded | Every registered operation/source can run its stated case; unsupported names are refused |
| Infrastructure | Owned disposable service lifecycle and environment/config validation | Bad config fails; start/stop affects only owned processes/state; restart/cleanup leaves external services untouched |
| Recovery controls | Test-only fault hooks or observable durability boundaries for the first path, excluded from advertised product capabilities | Evaluator observes the relevant commit boundary and interrupts only its owned process; fixed-delay killing is smoke evidence only (`realization/demos/README.md`) |
| Packaging and instructions | Usable successor hooks/checks, supported package output, accurate docs and task routing | Clean candidate obeys actual source/forge gates; consumer pins survive replacement; no obsolete command remains advertised |

Prefer actual types/grants over additional validation machinery. A stub error may truthfully reject an unsupported capability, but it cannot stand in for a required foundation: ownership, identity, storage/schema and working checks must have real behavior. No universal “implemented” marker or proof-score system.

## Order and ownership

1. **Preservation/inventory owner:** reconcile source and consumers, verify recovery, and produce the exact path/package disposition. Complete and integrate retained work on the default branches before M0 preparation, then retire obsolete/unowned checkout leftovers whose contents have a verified disposition and refresh preservation from those heads. Retain functional checkouts with a named owner/purpose; useful isolation has no numeric cap and does not keep a satisfied feature intention open. Do this before replacement or deletion; never publish private refs blindly.
2. **Product foundation owner:** settle target concrete types, error/outcome mapping and the first path's interfaces. Draft consequential connection ownership against both a role-bound store interface and a generic transaction interface; retain the role-bound design already required by Sinex. The generic dummy Tx makes forgery easy and exposes too much to callers.
3. **Store/build owners:** implement foundation contracts and successor build/check/isolated-service routes as one connected candidate. No broad provider fleet before those contracts work. Do not copy whole old tooling to make hooks pass.
4. **Integrator:** run the bootstrap cases against the fixed candidate and package, inspect public/private changes and review actual required forge statuses. Replacement readiness, merge and operational cutover are distinct evidence.
5. **Product lead:** deliver the first source-to-client tracer path (Polylogue Claude Code; Sinex one shell source), then widen to full slice acceptance. Later feature signatures stay honest and unregistered.

The next work in this spec repository is concrete bootstrap type/ownership refinement and executable unsupported/outcome contracts, then the first demonstration packets and requirement/task mapping. Transfer verified behavior from existing prototypes into those target modules selectively. Re-run the same cases through the actual production route before claiming equivalence; a composition checker or isolated prototype cannot prove a deployed client works.

## Verification of this audit batch

Before correction, all three metadata mutations returned exit 0. Afterwards `sinex-spec/skeleton/test_check_skeleton.py` accepts the valid tree and rejects each mutation with its specific diagnostic. Both existing type comparisons pass with the counts above. Integrated job 5907 (`spec-check-9cb00445`) exited 0, including the new mutation cases and both composition builds. Document-reference, link and whitespace checks pass. This is audit/composition evidence; the minimum-bootstrap acceptance table remains unsatisfied.

## Outcome-contract refinement

The follow-up outcome batch moves shared result/failure values into Polylogue model, preserves the query export, replaces raw reply error codes with controlled constructors, and changes store reads/briefing propagation to typed results. CLI completion derives status from the reply and propagates output errors; MCP mapping preserves named basic arguments and returns replies. Unsupported rich options remain refused. Offline value tests cover these cases. The audit's discarded-reply defect is addressed at these call sites; actual socket/stdio transport, encoders, query/read bodies and row rendering remain unimplemented, so bootstrap acceptance is still open.
