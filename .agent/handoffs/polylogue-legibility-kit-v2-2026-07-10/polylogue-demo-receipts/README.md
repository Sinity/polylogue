# Clean source-install receipt

Executed on July 10, 2026 from an empty directory with no
`POLYLOGUE_ARCHIVE_ROOT`:

```bash
env -u POLYLOGUE_ARCHIVE_ROOT UV_NO_CACHE=1 \
  uvx --from <polylogue-repository> \
  polylogue demo receipts --format json
```

The command built Polylogue from the supplied source checkout, created its own
private deterministic archive under the current directory, and returned a
successful evidence verdict.

Observed result:

- process time including no-cache build and dependency installation: 14.75s;
- verdict: `contradicted_at_claim_time_then_repaired`;
- failed structural tool receipt: exit code 1;
- later repair receipt: exit code 0;
- anti-grep control: two textual `error` hits and zero failed actions;
- command-created archive: `polylogue-receipts-demo/archive`.

The long package-index warning stream was excluded from this public packet. It
reported obsolete historical package artifacts, not a Polylogue failure.
