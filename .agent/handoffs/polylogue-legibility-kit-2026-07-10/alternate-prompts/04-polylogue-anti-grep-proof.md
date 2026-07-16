# Fork 04 — Polylogue anti-grep proof card

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, particularly `polylogue-3tl.15` and existing demo/claims contracts.

## Mission

Create the smallest rigorous public proof that Polylogue performs a domain operation that grep or ordinary transcript search cannot perform.

Do not claim that grep is useless. Demonstrate a precise semantic distinction.

## Preferred construct

Use structural tool failure:

- a transcript with prose containing “error” but no failed tool outcome;
- a transcript with a structurally failed tool result whose output does not contain “error”;
- provider-normalized fields such as exit status or `is_error`;
- a query that returns the latter and excludes the former;
- exact evidence refs and source material drill-down.

A second optional card may show copied lineage and physical-versus-logical counts.

## Owned scope

Own a small deterministic fixture or reuse existing fixture IDs, one proof-card generator/page, focused tests, and generated public artifacts. Avoid changing global README copy, renderer architecture, or query grammar unless a true product defect blocks the proof.

## Independent oracle

The fixture manifest must declare expected structural outcomes independently of the query result. The proof fails if expected labels are derived from the same implementation path being tested.

## Output form

Produce a compact side-by-side artifact:

```text
Text search for “error”       → matches prose, misses semantic distinction
Polylogue structural query    → failed action(s), typed status, exact refs
```

Then explain the exact construct in no more than 200 words.

## Falsifiers

The card must fail when:

- the structural query includes the prose-only control;
- it excludes the failure-without-keyword control;
- evidence refs do not resolve;
- provider-specific fields leak into the public query contract;
- the artifact is regenerated from a stale or different fixture world.

## Validation and deliverables

Run focused parser/query/ref tests and determinism checks. Deliver a patch, HTML/Markdown card, machine-readable packet, test transcript, and the exact public claim wording suitable for the claims ledger.
