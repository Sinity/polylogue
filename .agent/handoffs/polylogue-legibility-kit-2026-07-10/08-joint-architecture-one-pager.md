# Sinex-backed Polylogue: one-page architecture

## Authority rule

Sinex is the durable evidence and lifecycle backend. Polylogue is the AI-work domain kernel and product. Beads is the work-intent authority. SQLite remains Polylogue's local edge projection and standalone authority when Sinex is absent.

## Data flow

```text
Provider exports / runtime files / browser capture / hooks / attachments
                                |
                   Polylogue acquisition + parsing
                                |
         +----------------------+----------------------+
         |                                             |
         v                                             v
Sinex material registry + CAS             Polylogue-domain observations
provider-native + normalized              stable object IDs + revisions
         |                                             |
         +----------------------+----------------------+
                                v
                   Sinex canonical evidence plane
            material, interpretations, judgments, effects,
             lifecycle, coverage, settlement, operations
                                |
         +----------------------+----------------------+
         v                                             v
Polylogue domain projections                  Cross-source derivations
sessions/messages/blocks/topology             moments/work episodes/
usage/assertions/context deliveries           Agent Work Packets
         |                                             |
         +----------------------+----------------------+
                                v
                     Polylogue product surfaces
                   CLI / web / MCP / context / analysis
                                |
                         SQLite edge replica
                local FTS, offline cache/outbox, UI state
```

## Identity layers

- stable Polylogue object ID — logical session/message/assertion identity;
- Polylogue revision ID — a particular normalized content/semantic revision;
- Sinex event ID — one admitted interpretation in one replay epoch;
- material ref and anchor — exact source evidence;
- content digests — storage/canonicalization descriptors, not object identity.

## Storage rule

Sinex stores both provider-native material and immutable normalized Polylogue segments. Events reference exact records or anchors. Current transcript pages and search indexes are projections.

## Privacy rule

Do not publish bulk raw text in generic NATS event payloads. Do store transcript material in Sinex under sensitive privacy classes and capability-scoped access. Generic Sinex MCP remains redacted/read-only; the Polylogue service receives explicit transcript-content capability.

## Completion proof

The integration reaches its decisive milestone when:

1. a complete transcript revision is staged and settled in Sinex;
2. stable Polylogue refs survive parser replay;
3. PostgreSQL Polylogue projections and the SQLite edge projection agree on declared semantics;
4. all rebuildable SQLite tiers can be dropped and reconstructed from Sinex;
5. durable user assertions and context-delivery history survive the rebuild;
6. a session view includes ambient Sinex evidence with coverage caveats;
7. an Agent Work Packet joins Beads intent, agent work, machine effects, verification, and outcome.
