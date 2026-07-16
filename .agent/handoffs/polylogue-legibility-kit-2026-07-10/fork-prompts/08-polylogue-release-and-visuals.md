# Fork prompt 08 — Produce the Polylogue install, visual, and launch kit

Use the uploaded Polylogue repository and prior analysis. Work as a release/proof engineer, not a copywriter.

Build the artifacts needed so publication becomes a decision rather than a scramble:

1. prove the current source-checkout lane from a clean checkout;
2. prove the Nix one-shot lane from a clean state;
3. inspect PyPI, Homebrew, OCI, NixOS, browser-extension, and other declared lanes, but mark each `proven`, `wired-not-proven`, or `not-currently-supported` based on actual execution evidence;
4. generate deterministic recordings for the current tour and any available flagship demo from canonical commands;
5. ensure recordings and reports share the same packet/run ID;
6. create a checksum manifest;
7. create launch assets: repository description, topics, social-card copy, short announcement, long announcement, demo captions, FAQ, limitations block, and press/analyst paragraph;
8. add a leak scan for private paths, hostnames, tokens, emails, and transcript text in generated artifacts.

Do not invent package availability. Do not make broad performance claims. Preserve failed install attempts as receipts rather than deleting them.

Inspect release workflows, `docs/installation.md`, release readiness docs, visual tape tooling, Pages generation, and existing demo-tour assets. Run the narrowest appropriate verification, and explain what could not be executed in the environment.

Store the complete kit under `/mnt/data/polylogue-launch-kit/`, including patches for repository-owned scripts/docs. Return the launch bundle link and a one-page go/no-go matrix.
