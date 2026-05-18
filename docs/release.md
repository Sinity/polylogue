# Release Checklist

Polylogue releases are driven by [release-please]
(`.github/workflows/release-please.yml`). Every push to `master` reconciles a
single open "release PR" that bumps `pyproject.toml`, rolls the `Unreleased`
heading in `CHANGELOG.md` to `[X.Y.Z] — YYYY-MM-DD`, and updates
`.release-please-manifest.json`. Merging that PR pushes a signed `vX.Y.Z` tag,
which triggers [`release.yml`](../.github/workflows/release.yml) to build and
publish to PyPI (Trusted Publishing / OIDC) and `ghcr.io/sinity/polylogue`.

This file is the operator-facing cut-time checklist. The manual procedure
(see "Manual Fallback" below) is retained only for cases where release-please
is unavailable or misbehaves.

## Conventional Commits

release-please reads conventional commit subjects from the squash-merged
history on `master` and decides the next version. The mapping configured in
`release-please-config.json` is:

| Prefix     | Bump  | Changelog section |
| ---------- | ----- | ----------------- |
| `feat:`    | minor | Added             |
| `fix:`     | patch | Fixed             |
| `perf:`    | patch | Changed           |
| `feat!:` / `BREAKING CHANGE:` footer | minor (pre-1.0) / major | Added + breaking notice |
| `refactor:`, `docs:`, `test:`, `chore:`, `build:`, `ci:`, `style:` | none | hidden |

While the project is pre-1.0, `bump-minor-pre-major` is set, so breaking
changes bump the minor segment instead of the major. Patch bumps for
`feat:`-only cycles are disabled (`bump-patch-for-minor-pre-major: false`).

This is exactly the conventional-commit policy already enforced by
[CONTRIBUTING.md](../CONTRIBUTING.md#commits). Authors do not need to do
anything different; release-please consumes the existing history.

## Pre-flight (before merging a release PR)

- [ ] `devtools verify` passes on the cut commit.
- [ ] `devtools render-all --check` clean.
- [ ] CI green on `master`.
- [ ] `CHANGELOG.md` diff in the release PR looks right — entries grouped under
      the configured sections, no stray Unreleased content left over.
- [ ] `SCHEMA_VERSION` bumped if any change in this slice modified
      `polylogue/storage/sqlite/schema_ddl.py` or its inputs.
- [ ] No `until: "vX.Y.Z"` exception in `docs/plans/*.yaml` references this
      release without a resolution PR.
- [ ] `pip-audit` green (covered by CI; re-run locally if anything changed
      since the last run).

## Cut (the normal path)

1. Merge the open release PR titled `chore(release): X.Y.Z`. The squash-merge
   commit lands on `master`.
2. release-please pushes a signed `vX.Y.Z` annotated tag at that commit.
3. The tag push triggers [`release.yml`](../.github/workflows/release.yml),
   which builds the wheel + sdist, smokes them via
   `devtools verify-distribution-surface`, publishes to PyPI via OIDC Trusted
   Publishing, and builds + pushes the OCI image to
   `ghcr.io/sinity/polylogue`.

If the release PR looks wrong (missing entries, wrong bump, stale title),
close it without merging and adjust the source commits — release-please will
recompute and open a new one on the next push to `master`.

PyPI Trusted Publishing requires a one-time registration on pypi.org under
"Publishing" → "Add a new pending publisher" with `owner = Sinity`,
`repo = polylogue`, `workflow = release.yml`, `environment = pypi`. If this
is missing the publish step fails with an OIDC error and the tag must be
re-cut after re-running the workflow.

## Post

- [ ] Draft GitHub release notes from the `[X.Y.Z]` section of
      `CHANGELOG.md` (release-please will also create a GitHub release that
      can be edited rather than written from scratch).
- [ ] Verify `polylogue --version` on the tagged commit shows the expected
      version, commit hash, and clean dirty state.
- [ ] Confirm the PyPI release page at
      `https://pypi.org/project/polylogue/X.Y.Z/` shows wheel + sdist.
- [ ] Confirm the GHCR image tag `ghcr.io/sinity/polylogue:X.Y.Z` exists and
      `podman pull` succeeds.
- [ ] Smoke `pip install polylogue==X.Y.Z` in a clean venv and run
      `polylogue --help`, `polylogued --help`, `polylogue-mcp --help`.
- [ ] Smoke the container: `podman run --rm
      ghcr.io/sinity/polylogue:X.Y.Z polylogue --version`.

## Rollback

If a tagged release is found broken before downstream picks it up:

- [ ] Delete the tag locally and on origin
      (`git tag -d vX.Y.Z; git push --delete origin vX.Y.Z`).
- [ ] Open a `revert: chore(release): X.Y.Z` PR that reverts the squash-merge
      release commit. release-please will reconcile on the next push and open
      a new release PR for the corrected next version.
- [ ] Add a `fix:` commit on `master` noting the rollback and the issue —
      this also enters the next release's changelog under "Fixed".

If the release has already shipped, prefer cutting `X.Y.Z+1` over rewriting
history; tag rewrites are disruptive once external links exist.

## Manual Fallback

If release-please is unavailable (action removed, GitHub outage, the bot's
output is unsalvageable), the manual procedure is:

1. In `CHANGELOG.md`, move the `Unreleased` section to a `[X.Y.Z] —
   YYYY-MM-DD` heading. Leave a fresh empty `Unreleased` block above it.
2. Update `pyproject.toml` `version` → `X.Y.Z`.
3. Update `.release-please-manifest.json` to `{".": "X.Y.Z"}` so the next
   release-please run does not regress the version.
4. Commit as `chore: release X.Y.Z` (no other changes in this commit).
5. Sign and annotate the tag: `git tag -s vX.Y.Z -m "polylogue X.Y.Z"`.
6. Push the tag: `git push origin vX.Y.Z` (or `git pst` —
   uses `--follow-tags`).

The downstream `release.yml` workflow does not care whether the tag was cut
by release-please or by hand — it triggers on any annotated `vX.Y.Z` tag.

[release-please]: https://github.com/googleapis/release-please
