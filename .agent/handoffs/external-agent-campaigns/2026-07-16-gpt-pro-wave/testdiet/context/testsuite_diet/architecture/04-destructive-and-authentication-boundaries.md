---
created: 2026-07-16
purpose: Decide executable mutation, confirmation, authorization, and receiver-pairing boundaries for L23-L24
status: recommended-decision
project: polylogue
---

# Destructive and authentication boundaries

## Decision

Turn the existing operation declarations into one executable
`OperationExecutor`. Every CLI, API, MCP, daemon, maintenance, and internal
repair route submits an `OperationRequest`; none calls a storage mutation
directly. The executor resolves targets, checks role/capability and excision
policy, creates a preview, validates confirmation when required, applies the
effect idempotently, and writes a durable receipt.

Keep storage-level guards as defense in depth, but do not ask storage alone to
own user-facing preview, actor authority, or confirmation semantics.

## Operation declaration

`OperationSpec` in `polylogue/operations/specs.py` is the single declaration
authority and must become executable. Each operation declares:

- stable operation id and semantic version;
- read/write/admin/system capability;
- reversibility and destructive class;
- target resolver and disclosure policy;
- whether a preview is mandatory;
- confirmation strength and expiry;
- idempotency and conflict behavior;
- execution handler and affected durable tiers;
- receipt/public projection schema;
- security, recovery, and audit obligations.

`action_contracts.py`, CLI help, MCP role discovery, HTTP policy, examples, and
tests are derived consumers. A static inventory may detect missing bindings but
cannot authorize behavior by containing a name.

## Confirmation levels

| Class | Examples | Requirement |
| --- | --- | --- |
| Reversible low-risk write | add mark/tag/note, save query | write capability; no interactive confirmation; durable receipt |
| Judgment/replacement | accept/reject candidate, overwrite correction | write capability plus expected generation/CAS; explicit intent in request |
| Destructive scoped effect | delete session, excise raw material, reset one derived tier | preview plus confirmation token bound to actor, operation, target-set digest, archive id, expiry, and spec version |
| Broad/irreversible maintenance | durable-tier copy-forward destruction, live authority repair | admin/system capability plus preview receipt and explicit operator authorization of that receipt digest |

A boolean `confirm=true` is only a compatibility adapter. It may be accepted by
a local interactive surface if the surface first obtained the bound preview
token; it is never sufficient as the executor's authority proof.

If the target set changes after preview, execution returns `preview_stale` and
requires a new preview. Dry-run is a real read-only plan using the same target
resolver and policy, not an approximate count path.

## Excision and suppression

Suppression changes public visibility while retaining governed evidence;
excision destroys evidence. They are distinct operations with distinct policy.
All derived projections, blobs, FTS, embeddings, and caches are effects of the
same excision receipt or explicit convergence obligations. An adapter cannot
delete a projection and claim the material excised, nor delete durable bytes
through a lower-level maintenance shortcut.

## Authentication contract

The local system remains single-user; authentication is a machine/receiver
capability boundary, not multi-user RBAC. Every HTTP transport—including fetch,
SSE, POST, assets, backfill ACK, and reconnect—uses the same bearer or scoped
first-party credential and exact Host/Origin policy. Missing/expired/rotated
credentials fail closed without unauthenticated fallback.

Read, write, admin, and system capabilities are explicit inputs to the same
operation executor. MCP discovery exposes only declarations authorized for its
role. Prompts, resources, saved recipes, and attacker-controlled archive text
never acquire capability or confirmation authority.

## Stable receiver pairing

The extension already has the correct emerging shape; complete it rather than
introducing another pairing protocol:

- receiver identity is stable across daemon restarts and non-secret;
- `/identity`/status returns `receiver_id`, API schema/contract range, auth
  requirement, and current runtime evidence under authenticated transport;
- extension pairing stores endpoint, receiver id, compatible contract range,
  token credential reference, last contact, and deliberate dev-override state;
- canonical endpoint recovery is allowed only when the receiver identity is the
  trusted one, or through an explicit re-pair;
- no port scan, browser-profile special case, token disclosure, or auth fallback;
- reset/re-pair preserves capture queue, backfill ledger, and extension instance
  identity;
- every capture receipt names both receiver and extension instance.

The current token-derived `receiver_identity()` is acceptable for the local
single-user contract as long as token rotation is treated as a trust-identity
replacement with an explicit re-pair. A future independently persisted random
receiver id is competitive only if rotation must preserve identity; it is not
needed to complete the present proof.

## Competitive alternatives

| Alternative | Advantage | Why not chosen |
| --- | --- | --- |
| Per-surface `confirm` booleans | Low implementation cost | Bypassable, unbound to target/actor, races target changes |
| Storage-only mutation gateway | Central low-level choke point | Cannot own preview, public authority, disclosure, or operation semantics alone |
| Declaration lint/allowlist | Easy CI coverage | Self-referential; proves spelling, not runtime routing |
| Separate admin implementation per surface | Tailored UX | Recreates policy drift and bypass routes |
| Auto-adopt any healthy localhost receiver | Convenient recovery | Cross-pairs dev/live instances and weakens trust |
| Port scan plus first authenticated response | Handles endpoint drift | Enlarges attack/confusion surface; identity contract makes it unnecessary |
| Multi-user RBAC now | General | Does not match the local single-writer threat model and adds unrelated scope |

## Migration sequence

1. Bind `OperationSpec` declarations to handlers and receipts.
2. Route one destructive vertical slice (session deletion/excision) through the
   executor and prove every surface plus maintenance uses it.
3. Route reset and raw-authority actuation.
4. Route reversible/judgment mutations, preserving their weaker confirmation
   classes and CAS semantics.
5. Delete direct adapter calls only after bypass mutation tests.
6. Finish `polylogue-jlme.5` packaged two-profile/canonical-failover proof on the
   already implemented receiver-pairing state.

## Required proof

- delete/excise/reset invoked through every surface produces the same plan,
  authorization decision, effect identity, and receipt;
- a target-set mutation between preview and apply is rejected;
- removing one executor call or adding a direct storage call makes a real-route
  bypass test fail;
- read roles, prompt injection, missing credentials, rotated tokens, reconnect,
  SSE, assets, and cross-origin requests fail closed;
- two packaged extension profiles pair to one receiver, while a deliberate dev
  receiver remains isolated and queued work survives re-pair.

Primary evidence: `polylogue-layg`, `polylogue-jnj.5`, `polylogue-jn40`,
`polylogue-t46.8`, `polylogue-jlme.2`, `polylogue-jlme.5`, `polylogue-kwsb.1`,
`polylogue-6jjv`; `polylogue/operations/specs.py`,
`polylogue/operations/action_contracts.py`, browser-capture receiver and
extension pairing code.
