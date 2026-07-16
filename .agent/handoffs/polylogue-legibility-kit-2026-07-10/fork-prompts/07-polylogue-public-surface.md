# Fork prompt 07 — Rewrite Polylogue’s public surface

Use the uploaded Polylogue repository and all prior analysis. Produce a mergeable public-surface patch covering the repository README, generated docs site, docs information architecture, demo/proof pages, and Sinex interop page.

Stable category:

> Polylogue is the local flight recorder and system of record for AI work.

Remove “Your AI memory” as the primary site title/tagline. Memory remains one capability, not the category.

The first screen must communicate:

- what Polylogue is;
- why it is more than grep or a chat viewer;
- one private-data-free command that currently works;
- a visible evidence chain;
- current status and limitations.

Add or revise:

- `README.md`;
- Demos page centered on The Receipts, Count It Once, and Honest Refusal;
- Proof/Findings page;
- maximal Sinex interop direction, explicitly labeled not fully implemented;
- docs map and navigation;
- homepage hero and cards;
- GitHub description/topics/social-card checklist;
- a cold-reader skim ladder.

Use current repository facts only. Supported current install lanes should be source checkout and Nix unless you actually verify more. Roadmap authority is Beads, not GitHub Issues. Distinguish facts, capabilities, field observations, and aspirations. Do not claim memory uplift, cost-by-outcome, complete Sinex backend, or general release readiness.

Inspect and update generated-source files, not only generated output. Run page generation, docs surface checks, command-reference/link/drift tests available in the repo. Generate a patch and rendered-site preview under `/mnt/data/polylogue-public-surface/`.

Return links to the patch, proposed README, rendered homepage screenshot/HTML, and verification receipt.
