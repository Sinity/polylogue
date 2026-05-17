# Release Checklist

This is the operator-facing checklist for cutting a `polylogue` release.
The release procedure itself is documented in
[CONTRIBUTING.md](../CONTRIBUTING.md#versioning-and-releases); this file
captures the things to *check* at cut time.

## Pre-flight

- [ ] Open PRs reconciled or explicitly deferred.
- [ ] `devtools verify` passes on the cut commit.
- [ ] `devtools render-all --check` clean.
- [ ] CI green on `master`.
- [ ] `CHANGELOG.md` Unreleased entries reviewed and grouped under
      `Added`, `Changed`, `Fixed`, `Security`, `Removed`.
- [ ] `SCHEMA_VERSION` bumped if any change in this slice modified
      `polylogue/storage/sqlite/schema_ddl.py` or its inputs.
- [ ] No `until: "vX.Y.Z"` exception in `docs/plans/*.yaml` references
      this release without a resolution PR.
- [ ] `pip-audit` green (covered by CI; re-run locally if anything
      changed since the last run).

## Cut

- [ ] In `CHANGELOG.md`, move the `Unreleased` section to a `[X.Y.Z] —
      YYYY-MM-DD` heading. Leave a fresh empty `Unreleased` block above
      it.
- [ ] `pyproject.toml` `version` → `X.Y.Z`.
- [ ] Commit as `chore: release X.Y.Z` (no other changes in this commit).
- [ ] Sign and annotate the tag: `git tag -s vX.Y.Z -m "polylogue X.Y.Z"`.
- [ ] Push the tag: `git push origin vX.Y.Z` (or
      `git pst` — uses `--follow-tags`).

Pushing the tag triggers
[`release.yml`](../.github/workflows/release.yml), which builds and
smokes the wheel + sdist (`devtools verify-distribution-surface`),
publishes them to PyPI via OIDC Trusted Publishing, and builds + pushes
the OCI image to `ghcr.io/sinity/polylogue`. A
[Trusted Publisher](https://docs.pypi.org/trusted-publishers/) for
`Sinity/polylogue` / `release.yml` / environment `pypi` must be
registered on PyPI; otherwise the publish step will fail with an OIDC
error and the tag must be re-cut after re-running the workflow.

## Post

- [ ] Draft GitHub release notes from the `[X.Y.Z]` section of
      `CHANGELOG.md`.
- [ ] Verify `polylogue --version` on the tagged commit shows the
      expected version, commit hash, and clean dirty state.
- [ ] Confirm the PyPI release page at
      `https://pypi.org/project/polylogue/X.Y.Z/` shows wheel + sdist.
- [ ] Confirm the GHCR image tag `ghcr.io/sinity/polylogue:X.Y.Z`
      exists and `podman pull` succeeds.
- [ ] Smoke `pip install polylogue==X.Y.Z` in a clean venv and run
      `polylogue --help`, `polylogued --help`, `polylogue-mcp --help`.
- [ ] Smoke the container: `podman run --rm
      ghcr.io/sinity/polylogue:X.Y.Z polylogue --version`.

## Rollback

If a tagged release is found broken before downstream picks it up:

- [ ] Delete the tag locally and on origin
      (`git tag -d vX.Y.Z; git push --delete origin vX.Y.Z`).
- [ ] Open a `release: revert vX.Y.Z` PR that reverts the
      `chore: release X.Y.Z` commit.
- [ ] Add a `Fixed` entry to `CHANGELOG.md` Unreleased noting the
      rollback and the issue.

If the release has already shipped, prefer cutting `X.Y.Z+1` over
rewriting history; tag rewrites are disruptive once external links exist.
