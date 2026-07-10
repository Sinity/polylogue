# Demo Tour Install-Path Proof

This proof was run on July 10, 2026 from the local package source using
`uvx --from <repo>` as the pre-release equivalent of the public command:

```bash
uvx --from <repo> polylogue demo tour --out-dir <tmp-output> --force --format json
```

Result:

- Status: passed
- End-to-end `uvx` process time, including resolution/install: 29.27s
- First evidence result after the command began: 2.109s
- Tour execution time reported by Polylogue: 6.038s
- Demo archive: 13 sessions, 55 messages, overlays present
- Declared fixture constructs: 34/34 satisfied
- Query/read steps: claim versus receipt, failed-actions aggregate, composed lineage, archive facets
- Problems: none

The environment emitted package-index warnings about obsolete release artifacts;
those warnings did not change the command exit status or proof result. This is a
source-install receipt, not yet a published-index receipt.

A second clean source-install proof exercised the self-contained first-contact command directly:

```bash
env -u POLYLOGUE_ARCHIVE_ROOT UV_NO_CACHE=1 \
  uvx --from <repo> polylogue demo receipts --format json
```

Result:

- Status: passed
- End-to-end process time, including a no-cache build and dependency install: 14.75s
- The command seeded its own archive under the current working directory
- Verdict: `contradicted_at_claim_time_then_repaired`
- Failed structural receipt: exit code 1
- Later recovery receipt: exit code 0
- Anti-grep control: two text hits for `error`, zero failed actions

The release-path command is intended to be:

```bash
uvx polylogue demo receipts
```

A published-index proof remains a release gate rather than a current claim.
