# Fork 05 — Polylogue public claims ledger and findings shelf

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, especially `polylogue-3tl.4`, `polylogue-3tl.16`, proof-artifact documentation, and existing tracked field packets.

## Mission

Turn scattered proof artifacts into a public evidence shelf where every claim is bounded, classed, caveated, and resolvable.

## Owned scope

Own:

- machine-readable public claims schema/ledger;
- human-readable claims and findings pages;
- generators and drift checks;
- links from existing proof-artifact documentation;
- no changes to the underlying experiments or runtime behavior.

## Claim classes

At minimum distinguish:

- deterministic public proof;
- private-archive field observation;
- current product capability;
- performance observation;
- negative result;
- planned capability.

Every claim entry should carry:

- stable claim ID;
- exact wording;
- class;
- date and revision;
- corpus or fixture;
- product boundary;
- command/query;
- evidence artifacts and refs;
- result;
- caveats;
- unsupported interpretations;
- regeneration or verification path;
- Beads ownership where ongoing.

## Required findings

Index at least:

1. deterministic demo facts;
2. the claim-versus-evidence field finding with sampling/calibration caveats;
3. physical-versus-logical usage gap from the tracked live archive;
4. context/handoff pilot, including its stale/ahead negative case;
5. real-archive performance timeout or other negative operational finding;
6. honesty anti-demo returning unsupported/unavailable rather than guessing.

Do not convert private field observations into universal claims.

## Drift gate

Add a repository check that catches at least:

- missing artifact paths;
- duplicate claim IDs;
- claims with no caveat field where the class requires one;
- GitHub Issue roadmap links where Beads are authoritative;
- generated pages out of date with the ledger.

## Deliverables

Produce the patch, generated shelf, schema/ledger, validation output, and a short editorial note recommending which three findings deserve landing-page links.
