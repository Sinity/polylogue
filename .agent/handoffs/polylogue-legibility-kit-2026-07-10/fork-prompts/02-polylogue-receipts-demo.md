# Fork prompt 02 — Implement Polylogue “The Receipts”

Use the uploaded Polylogue repository and the complete prior analysis in this chat. Implement the strongest first-contact Polylogue demo: **The Receipts**, aligned with `polylogue-212.2` and especially the ready follow-up `polylogue-xyel`.

Do not merely make a report script. Use existing product primitives and the current Demo Finding Packet contract. Extend that contract only where necessary to carry a primary construct, oracle, falsifier, positive/negative/missing-evidence controls, exact material/evidence refs, scope, and recording parity.

The demo must show a two-column claim-versus-observed view:

- assistant claim: “All tests pass; the fix is complete” or equivalent synthetic text;
- observed structural tool outcome: paired test invocation, nonzero exit status, duration, and result ref.

Required controls:

1. nonzero exit is classified as failure;
2. zero exit with the word “failed” somewhere in text remains success;
3. claim with no paired result becomes `insufficient_evidence` or the project’s equivalent—not failure or success;
4. malformed or ambiguous pairing produces a loud caveat;
5. the verdict derives from structured outcome fields, never prose regex.

The demo must resolve every visible claim, outcome, and follow-up classification to stable refs and source evidence. It must emit a validated packet, report, queries, evidence rows, checks, and raw run transcript. The recording must be generated from the same canonical command.

Inspect `devtools/demo_packet.py`, the existing packet stub, anti-demo packet, claim-vs-evidence tooling, `polylogue/demo`, CLI demo commands, renderer code, and relevant tests. Reuse the deterministic corpus; if Incident 14:32 is not yet implemented in this fork, add the smallest self-contained provider-shaped fixture required and make the patch easy to rebase onto that corpus later.

Produce a patch-ready worktree, tests, generated public-safe packet, and recording source. Run targeted tests and the demo from a clean temporary archive. Report exact commands and exit codes. Do not overclaim prevalence or agent intent.

Keep public wording aligned with: “Polylogue is the local flight recorder and system of record for AI work.” Do not use “Your AI memory” as the primary category.

Store outputs under `/mnt/data/polylogue-receipts-demo/` and return links to the patch, packet, report, and verification receipt.
