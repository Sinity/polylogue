# Polylogue hot-daemon delivery

Base: `master` at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.

Contents:

- `polylogue-hot-daemon-source/`: complete modified source tree without `.git` or build caches.
- `polylogue-hot-daemon.patch`: complete binary-capable Git patch, including added files.
- `polylogue-hot-daemon-summary.md`: implementation and verification handoff.
- `polylogue-hot-daemon-validation.md`: repository identity, diff statistics, and check results.
- `verification.log`: complete formatter/linter/test output.
- `verification-summary.tsv`: machine-readable exit statuses.
- `perf-02-hot-daemon.md`: original task specification.

Apply the patch from the matching base checkout with:

```sh
git apply --3way polylogue-hot-daemon.patch
```
