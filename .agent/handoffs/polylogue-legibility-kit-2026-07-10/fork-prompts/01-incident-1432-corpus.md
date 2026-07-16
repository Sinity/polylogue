# Fork prompt 01 — Build the Incident 14:32 evidence corpus

Use the Polylogue and Sinex repositories uploaded in this chat and the analysis already established here. Work independently and deeply. Do not stop at a design memo: produce an implementation-ready, patch-backed shared deterministic evidence corpus named **Incident 14:32**.

The corpus must support both projects without collapsing their ontologies. It should model a small AI-assisted coding incident containing, at minimum:

- a browser/design conversation;
- two coding-agent sessions;
- a copied-prefix fork or continuation;
- a fresh subagent control;
- a compaction boundary and summary;
- one nonzero structured test result followed by an assistant success claim;
- one later successful verification;
- provider usage with cache and reasoning lanes;
- one attachment with actual retained bytes;
- a candidate assertion, accepted assertion, and stale/rejected assertion;
- one context image and delivery snapshot;
- shell, Git, filesystem, browser, and desktop evidence for the same interval;
- one deliberate source outage;
- parser semantics v1 and v2 for one record;
- one ambiguous cross-source duplicate;
- one Beads task with acceptance criteria and a dependency transition.

The critical rigor requirement is that fixture generation and expected-result calculation must not use the same reducer. Create a declarative scenario manifest and an independent oracle manifest covering physical sessions, logical composition, tool outcomes, usage lanes, coverage intervals, parser semantic diff, material hashes, assertion state, and context delivery.

Inspect the existing Polylogue demo machinery under `polylogue/demo`, its construct audit, and Sinex source/test fixture conventions. Reuse real provider-shaped parsers and product paths wherever practical; do not create a demo-only parallel ontology. Preserve public safety: no private text, real secrets, hostnames, or absolute user paths.

Produce:

1. a detailed scenario specification;
2. a fixture and oracle data model;
3. patch-ready changes for Polylogue that seed and verify the corpus;
4. a Sinex fixture bundle or exact follow-on patch plan where direct implementation is too broad;
5. tests proving every declared construct is nonempty and every control behaves as expected;
6. a content manifest and privacy/licensing statement;
7. one patch per repository, generated against the exact snapshot commit;
8. a handoff listing files, commands, test results, unrun checks, and dependent Beads.

Use Beads, not GitHub Issues, as roadmap authority. Propose narrowly scoped new child Beads only where existing ones cannot own the work. Do not claim the full Sinex-backed integration exists. Treat Sinex as the ultimate durable backend target and Polylogue as the AI-work domain kernel.

Store all output under `/mnt/data/incident-1432-corpus/`, including `.patch` files and a top-level `README.md`. Return links to the artifact bundle and the most important files.
