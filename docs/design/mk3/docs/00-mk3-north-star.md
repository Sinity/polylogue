# MK3 north star: archive workbench, not prettier reader

MK3 should make Polylogue answer the questions a heavy coding-agent user actually has while reading old sessions:

- What happened here, and where is the useful part?
- Is this message typed, pasted, generated, tool output, thinking, or attachment-derived?
- What can I copy, cite, mark, annotate, or hand to another agent?
- Is this session a continuation, fork, sidechain, or subagent of something else?
- Which other chatlogs should be open beside this one?
- Which raw file, hook event, attachment, and source run produced this row?
- What data is missing, stale, estimated, or unresolved?

The core product metaphor is an **evidence cockpit with a stack workspace**. Search/list finds candidates. Conversation reader gives high-fidelity transcript display. Stack/compare lets multiple chatlogs be read together. Topology explains continuations and forks. Paste/attachment surfaces stop treating large hidden context as plain text. Inspector panels expose provenance and derived read models without polluting the main reading flow.

MK3 should keep MK2's daemon-first local model, but it must stop designing around a single conversation pane. The user workflow is usually cross-session: continue a task, compare attempts, recover context after a crash, or mine recurring prompt patterns.

## Design rules

1. Every visible fact has provenance. A chip should be clickable to show whether it came from canonical storage, parser inference, hook evidence, raw JSON, or heuristic detection.
2. The main transcript stays readable. Heavy metadata goes in chips, folds, hover sheets, and the inspector.
3. User state is durable archive state, not browser state. Marks, notes, saved views, recall packs, and workspaces are persisted through shared operations.
4. Branching is first-class. Continuation/fork/sidechain/subagent are not labels; they are topology edges with confidence and unresolved states.
5. Pastes and attachments get their own renderers. A 10k-line paste is not an ordinary paragraph, and an attachment is not just a filename.
6. Advanced panels can exist before the backend is complete, but disabled/partial states must be explicit and useful.
7. Multi-chat reading is a primary mode: tabs for quick switching, stack for related chatlogs, compare for two selected sessions, timeline for chronological synthesis.
