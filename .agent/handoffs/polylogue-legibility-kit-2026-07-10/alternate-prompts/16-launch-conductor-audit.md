# Fork 16 — Launch conductor, integration audit, and cold-reader gate

Act as the integration and release conductor for the Polylogue and Sinex external-legibility swarm. Use Beads as roadmap authority. Do not implement a large feature unless it is the only way to unblock integration; your primary job is to make independent branches cohere and to reject ungrounded public claims.

## Inputs

Expect branches or patches from narrative, semantic rendering, demo, claims, install/media, web reliability, Sinex demo, and joint architecture lanes.

## Mission

Create a release-candidate integration branch and evidence packet through aggressive patch absorption, conflict resolution, and validation. Git history elegance is secondary. Semantic correctness and explicit ownership are not.

## Merge policy

Recommended order:

1. web reliability and trust-floor blockers;
2. narrative/docs registries;
3. fixture and demo substrate;
4. semantic rendering contract;
5. terminal/web renderers;
6. claims/findings generators;
7. install/media/launch artifacts;
8. Sinex narrative and demo contracts;
9. joint architecture docs/protocols.

Use squash or direct patch application freely. Never silently resolve two competing domain decisions; record the conflict and choose under the declared authority matrix.

## Required audit

1. **Category consistency:** one category sentence per project across README, site, package metadata, media, and launch packet.
2. **Claim reachability:** every public claim resolves to a ledger entry and evidence artifact.
3. **Construct validity:** each promoted demo has question, claim, construct, fixture/intervention, product boundary, oracle, negative controls, caveats, refs, and regeneration path.
4. **Current versus planned:** no roadmap feature described as shipped.
5. **Privacy scrub:** no secrets, usernames, private paths, hostnames, private session IDs, or unlicensed source text.
6. **Roadmap truth:** no GitHub Issue navigation where Beads are authoritative.
7. **Generated-artifact drift:** docs, pages, snapshots, media, and ledgers regenerated and clean.
8. **Cold install:** supported install path works in a clean environment.
9. **Cold reader:** an uninvolved reader can answer category, payoff, evidence versus derivation, one refused claim, current status, and evidence drill-down path.
10. **Backend direction:** no surviving text incorrectly says Sinex can never store transcripts in the ultimate design.

## Single-machine scheduling

Respect resource tokens:

- one heavy Rust build at a time;
- one browser/E2E lane at a time;
- one large Polylogue verification run at a time;
- cheap docs/static lanes can run in parallel;
- avoid concurrent package-manager mutation of the same cache or virtual environment.

## Deliverables

Produce:

- integrated patch/branch;
- launch manifest with commit/patch IDs;
- validation matrix and exact commands;
- public-claim diff;
- privacy scrub result;
- cold-reader questionnaire and results;
- accepted, rejected, and deferred branch list with reasons;
- final “ship / do not ship” decision and blocking Beads.

Reject impressiveness without evidence. Prefer one strong, reproducible story over a broad but incoherent showcase.
