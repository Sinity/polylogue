# Incident 14:32: independent oracle corpus

This is a frozen, public-safe software-work incident designed to test Polylogue, Sinex, and their future integration without letting either product define its own answer key.

The corpus has one memorable disagreement: a test exits 1, then the assistant claims all tests pass. A later edit and rerun genuinely repair the test. Around that core it adds the controls needed to test broader constructs:

- prose containing the word `error` but no failed action;
- a copied-prefix fork and a fresh-subagent control;
- a compaction summary that omits the initial failure and false claim;
- terminal, Git, Beads, source-health, assertion, and context-delivery evidence;
- a browser coverage outage;
- two parser interpretations of the same source record with one stable domain identity.

`oracle.json` was authored independently of product output. `verify_incident.py` derives facts from the source materials and compares them with that frozen oracle. A product demo must then compare its own output to the same oracle; generating the oracle from the product would invalidate the proof.

Run:

```bash
python incident-1432/verify_incident.py --root incident-1432 --out-dir incident-1432/output
```

The verifier checks:

1. material hashes and manifest completeness;
2. claim/failure/recovery ordering and structural outcomes;
3. the anti-grep control;
4. physical versus logical lineage accounting;
5. source coverage semantics;
6. parser replay identity and supersession;
7. assertion eligibility and context delivery;
8. Beads/Git/terminal agreement;
9. absence of absolute paths and obvious credentials.

The corpus is a contract fixture. It does not establish field prevalence, scale, performance, memory uplift, deletion, or backend completion.
