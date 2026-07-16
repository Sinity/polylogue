# Repository proposal patches

These patches are generated against the exact snapshots inspected for this package.

| Project | Base commit | Patch |
|---|---|---|
| Polylogue | `f6c1da99` | `polylogue-external-legibility.patch` |
| Sinex | `b70a08d9` | `sinex-external-legibility.patch` |

Apply from a clean checkout at the corresponding base commit:

```bash
git apply --index /path/to/polylogue-external-legibility.patch
# or
git apply --index /path/to/sinex-external-legibility.patch
```

The Polylogue patch changes public copy, generated-site source, docs navigation, demos/proof/findings pages, a claims ledger, the maximal Sinex direction, and the shipped deterministic tour. The tour now starts with a structural failure receipt, aggregates typed failures, demonstrates composed lineage, emits an explicit non-claims section, and retains the complete audit in machine-readable output. It does not implement the full semantic renderer or Sinex data plane.

The Sinex patch rewrites the public entry point, adds a skim-oriented documentation map, product/concepts page, falsifiable demo portfolio, proof/claims index, public claims ledger, and maximal Polylogue backend contract. It also changes the integration-authority document to distinguish the current metadata-thin bridge from the target architecture. It does not change runtime code.

Run the verification commands summarized in `../15-validation-report.md` after applying. Review all public wording against the current Beads and code if applying to a later commit.
