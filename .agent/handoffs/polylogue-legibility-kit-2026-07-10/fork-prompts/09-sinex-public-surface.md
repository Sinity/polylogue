# Fork prompt 09 — Make Sinex externally legible

Use the uploaded Sinex repository and prior analysis. Produce a mergeable external-legibility patch. Sinex’s first screen must no longer read primarily as a Rust/PostgreSQL/NATS architecture document.

Stable category:

> Sinex is the local evidence substrate for digital life and agent work.

Lead with the project’s genuinely distinctive concepts:

- source material versus interpretation;
- occurrence, coining, and persistence time;
- replay that changes interpretation without rewriting source history;
- explicit coverage gaps;
- confidence versus authority;
- current projections versus canonical events;
- agent-facing evidence access;
- Polylogue as the AI-work domain product on the maximal backend path.

Produce:

- rewritten `README.md`;
- top-level docs map with a 30-second/3-minute/30-minute skim ladder;
- product/concepts page;
- demos page that separates smoke verification, capability demos, system safety proofs, and experiments;
- proof-artifacts/claims page;
- maximal Polylogue integration page, present state versus target clearly separated;
- removal or replacement of retired GitHub Issue roadmap links with Beads references or stable docs;
- repository description/topics/social-card checklist.

The deterministic `sinexctl ops verify --demo` walkthrough must be described honestly as an operational smoke proof, not the full thesis demo. The private large-deployment observations must be labeled bounded field evidence, including recovery caveats.

Run documentation checks and any generated-doc commands. Do not modify the core runtime except to fix a directly exposed documentation contract. Store outputs under `/mnt/data/sinex-public-surface/`, with patch, rendered docs preview, and verification receipt.
