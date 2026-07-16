# Maximal Sinex–Polylogue interop contract

The maximal target is not a metadata bridge. It is a domain-on-substrate architecture in which Sinex becomes the durable backend of Polylogue while Polylogue remains a distinct product and semantic authority.

## The governing split

> **Sinex owns durable persistence, evidence lifecycle, replay, coverage, and cross-domain effects. Polylogue owns AI-work identity, normalization semantics, logical composition, query/context behavior, and user experience.**

That split is compatible with Polylogue retaining SQLite. The deployment profile determines what SQLite means.

### Standalone Polylogue

SQLite tiers and the Polylogue blob store are authoritative. No Sinex dependency exists.

### Sinex-backed Polylogue

Sinex stores all irreplaceable evidence and reviewed state. PostgreSQL holds shared projections. SQLite is an offline replica, local FTS/vector accelerator, cache, UI-state store, and durable outbox.

No data class may have two silent masters.

## What Sinex ultimately stores

### Source material

- provider exports;
- coding-agent rollout/session files;
- browser capture payloads;
- hook records;
- attachments;
- provider usage material;
- immutable acquisition snapshots.

### Normalized Polylogue material

- immutable transcript segments;
- normalized messages and content blocks;
- tool calls/results;
- protocol/context material;
- attachment manifests;
- usage observations;
- provider/origin semantics version.

### Durable domain history

- sessions and revisions;
- stable message/block/action identities;
- lifecycle and topology observations;
- usage/cost observations;
- assertions, candidates, judgments, and supersession;
- context images and delivery records;
- deletion/retention state;
- model-effect records.

### Cross-domain derived products

- work episodes;
- Agent Work Packets;
- stale-assertion candidates;
- project/branch/session relations;
- outcome and verification relations.

## What remains Polylogue-owned

Sinex must not define:

- provider message/block normalization;
- human/assistant/protocol/material authoredness;
- tool-call/result pairing semantics;
- logical session root and copied-prefix composition;
- continuation versus fork versus fresh subagent;
- compaction/context boundaries;
- physical versus logical usage accounting;
- assertion kinds and context policy;
- transcript query language;
- context compilation;
- semantic transcript rendering.

Sinex persists and projects those semantics under versioned contracts.

## Identity model

Four identities are mandatory:

1. **Stable domain object ID** — logical session, message, block, assertion, delivery, or attachment identity.
2. **Domain revision ID** — one revision of that object.
3. **Sinex interpretation event ID** — one admitted interpretation in one replay/semantics epoch.
4. **Material occurrence anchor** — exact source or normalized material coordinates.

Content hashes are descriptors, not object identity. Sinex event UUIDs are replay-specific, not durable Polylogue refs. Byte offsets are evidence coordinates, not stable refs after resegmentation.

A cross-resolution ledger must support aliases, reconciliation, and ambiguity. The current `sinex-4j2.7` hope that no new identity table is required should be dropped.

## Transcript revision protocol

A session is a mutable aggregate; admitted events are immutable observations. The bridge needs transaction-like revision completeness.

A revision manifest declares:

- stable session ID;
- domain revision ID;
- parser and semantics versions;
- provider-native material refs;
- immutable normalized segments;
- expected message/block/tool/attachment/usage counts;
- multi-digest descriptors;
- session-local ordering rules;
- privacy class;
- source coverage and completeness;
- batch/settlement ID;
- previous revision;
- expected final frontier.

Admission stages material before compact events are published. Generic NATS payloads carry IDs, typed metadata, and anchors—not bulk transcript bodies.

A reader sees either:

- the previous complete revision;
- the new complete revision;
- or an explicit partial/failed/unsettled state.

It never sees a half-imported session as ordinary complete content.

## Material segmentation

Do not anchor stable identity to offsets in a mutable regenerated export.

Use immutable sealed segments:

```text
session
  └── revision manifest
        ├── provider material refs
        ├── normalized segment 0001
        ├── normalized segment 0002
        ├── attachment manifest
        ├── usage segment
        └── expected counts and digests
```

Each normalized record carries a stable domain ID and source anchors. Resegmentation creates new material anchors and revision records without destroying the stable identity.

## Provenance versus topology

Sinex derivation provenance means “this interpretation was computed from these events.”

Polylogue topology means continuation, fork, copied prefix, subagent, sidechain, compaction, or context inheritance.

These are not interchangeable. Topology belongs in typed Polylogue relationships. Encoding a fork as Sinex derivation would break replay, deletion, and logical accounting.

## Projection model

Sinex/PostgreSQL should eventually provide Polylogue projections for current shared/server use:

- sessions and revisions;
- messages/blocks/actions;
- topology;
- attachments;
- usage and costs;
- assertions and judgment history;
- context deliveries;
- search documents;
- projection/frontier state.

Polylogue SQLite retains equivalent edge projections for standalone/offline/local use. Semantic parity matters; identical DDL does not.

## Rebuild proof

The decisive command is conceptually:

```bash
polylogue rebuild --from-sinex --verify-parity
```

It deletes all rebuildable SQLite tiers and reconstructs from Sinex-held material and domain history.

The parity report classifies:

- exact matches;
- intentionally local UI state;
- unavailable optional embeddings;
- unsupported legacy evidence;
- changed parser semantics;
- deleted/retained material;
- actual defects.

“Sinex is the backend” is not a shipped claim before this proof exists.

## User-authored and reviewed state

In backed mode, Sinex must durably store:

- marks;
- corrections;
- notes and decisions;
- candidate/accepted/rejected/superseded assertions;
- lessons and handoffs;
- judgments;
- context policy;
- durable analyses intended for future use.

SQLite may hold an offline replica and outbox. Window geometry, scroll state, query history, and ephemeral scratch can remain local-only.

## Context

Sinex supplies broad evidence candidates, source health, ambient machine activity, and cross-domain relations. Polylogue owns AI-work context selection and trust policy.

The final delivered context artifact must record:

- selected evidence and assertion refs;
- trust/authorship class;
- omissions and reasons;
- token/message budget;
- redaction policy;
- source-coverage caveats;
- delivery boundary;
- exact bytes or canonical digest delivered.

A generated summary is a workspace artifact. The fact that it was delivered is a durable observation.

## Query composition

Do not flatten Polylogue into generic `core.events` scans.

Use:

- Polylogue domain projections and query semantics for transcript/session/action/context questions;
- Sinex moment/cross-source queries for ambient evidence;
- a composed front door that retains native refs and caveats.

Example:

```text
Polylogue finds the claim and session interval
→ Sinex gathers terminal/Git/browser/source-health evidence
→ Polylogue renders the work packet
→ every leg remains separately resolvable
```

## Embeddings and model effects

Reuse requires a recipe key broader than `(provider, model, content hash)`:

- input content digest;
- canonicalization version;
- chunk selector and chunking version;
- provider/model/model revision;
- dimensions;
- task/input type;
- normalization policy;
- privacy and redaction policy.

Vectors remain rebuildable projections. The model-effect ledger records how they were produced and whether reuse is safe.

## Security and deletion

Transcript-complete storage increases value and blast radius.

Required controls:

- raw-content capability separate from metadata/search;
- transcript/tool/reasoning/attachment/context privacy classes;
- scoped service tokens;
- generic Sinex MCP redacted by default;
- privileged Polylogue retrieval path;
- encrypted host storage;
- export/render redaction;
- model-input allowlists;
- domain-aware tombstones;
- material reference accounting;
- FTS/vector/context/report invalidation;
- physical purge receipts.

Shared-prefix domain relationships must not trigger derivation-style over-cascade.

## Offline behavior

Backed mode needs a durable SQLite outbox:

1. local user writes are recorded with stable object/revision identity;
2. the outbox retries admission to Sinex;
3. Sinex accepts, rejects, or reports conflict;
4. local projection advances only after confirmed authority;
5. conflicts remain visible rather than last-write-wins silently.

## Current correction to project doctrine

`polylogue-6mv` says Sinex must not ingest raw transcripts. Under the maximal direction, that decision is wrong and should be superseded.

The useful security insight survives: raw transcript bodies should not be sprayed into generic durable event payloads or generic MCP responses. They should live in Sinex’s protected material plane with explicit capability and lifecycle policy.

`polylogue-fs1.9` should be re-scoped to low-volume notification, correlation, health, and lag signals. It should not define the storage boundary.

The Sinex `sinex-4j2` epic already contains most of the correct target. Its metadata-only child wording should be reconciled with the epic’s transcript-complete Phase B and rebuild requirement.
