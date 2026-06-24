# Release Checklist

Polylogue releases are driven by [release-please]
(`.github/workflows/release-please.yml`). Every push to `master` reconciles a
single open "release PR" that bumps `pyproject.toml`, rolls the `Unreleased`
heading in `CHANGELOG.md` to `[X.Y.Z] — YYYY-MM-DD`, and updates
`.release-please-manifest.json`. Merging that PR pushes a signed `vX.Y.Z` tag,
which triggers [`release.yml`](../.github/workflows/release.yml) to build and
publish the Python distributions to PyPI (Trusted Publishing / OIDC). The same
tag also triggers [`container.yml`](../.github/workflows/container.yml), which
builds and publishes the slim and distroless OCI images to
`ghcr.io/sinity/polylogue`.

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
- [ ] `devtools render all --check` clean.
- [ ] `devtools release verify-distribution` passes locally when packaging or
      runtime dependencies changed.
- [ ] `devtools release build-package` passes when Nix packaging or dependency
      closure changed.
- [ ] `devtools workspace deployment-smoke --json` captures installed command
      versions, daemon URL, browser-capture receiver URL, archive root, and
      resource signals on the deployment host when this release claims deployed
      package readiness.
- [ ] CI green on `master`.
- [ ] `CHANGELOG.md` diff in the release PR looks right — entries grouped under
      the configured sections, no stray Unreleased content left over.
- [ ] The owning archive-tier schema version was bumped if this slice modified
      `polylogue/storage/sqlite/archive_tiers/*.py` DDL.
- [ ] No `until: "vX.Y.Z"` exception in `docs/plans/*.yaml` references this
      release without a resolution PR.
- [ ] `pip-audit` green (covered by CI; re-run locally if anything changed
      since the last run).

These pre-flight checks are source-safe: they build from the checkout, install
or inspect the produced artifact, and run against an empty or explicit archive
root. Browser first-paint proof is opt-in and local-host only:
`devtools workspace deployment-smoke --browser --browser-executable ...` records
the resolved Chrome/Chromium path, DOM bytes, and screenshot bytes for the web
root, but it is not a claim about MCP DevTools navigation, copied profiles,
authenticated provider pages, or production archive timings.

## Cut (the normal path)

1. Merge the open release PR titled `chore(release): X.Y.Z`. The squash-merge
   commit lands on `master`.
2. release-please pushes a signed `vX.Y.Z` annotated tag at that commit.
3. The tag push triggers [`release.yml`](../.github/workflows/release.yml),
   which runs, in order:
   - `build-and-smoke` — builds wheel + sdist, runs
     `devtools release verify-distribution`, and verifies PyPI long-description
     renderability with `twine check`. The distribution verifier installs the
     wheel in a clean venv, removes source-tree `PYTHONPATH`, imports runtime
     entrypoint modules from the installed artifact, and then smokes the CLI,
     daemon, MCP, and query-parser paths so a missing runtime dependency fails
     before publication.
   - `installed-smoke` matrix — installs the freshly-built wheel into a fresh
     `python -m venv` on `{ubuntu-latest, macos-latest} × py{3.11, 3.12, 3.13}`
     and exercises `polylogue --version`, `polylogue --help`,
     `polylogue --plain analyze --count`, `polylogued --help`,
     `polylogue-mcp --help`, and `python -m polylogue --version` against an
     empty `POLYLOGUE_ARCHIVE_ROOT`. This is the
     OS/Python-version coverage gate: PyPI publication waits on it.
   - `sbom` — emits a CycloneDX SBOM (`*.cdx.json` and `*.cdx.xml`) from the
     wheel's resolved dependency closure and uploads it as the
     `sbom-artifacts` workflow artifact.
   - `publish-pypi` — signs wheel + sdist with Sigstore keyless OIDC
     (`sigstore/gh-action-sigstore-python`), publishes to PyPI via
     Trusted Publishing (no API token), and retains the `.sigstore` bundles
     as the `signing-artifacts-publish-pypi` workflow artifact.
   - `publish-pypi-mcp` / `publish-pypi-hooks` — sign and publish the wrapper
     distributions for the MCP console script and Claude Code hook sidecar after
     their installed-smoke jobs pass.
   - [`container.yml`](../.github/workflows/container.yml) — runs on the same
     tag, builds the runtime and distroless images, emits push-path provenance
     and SBOM attestations through Docker Buildx, and pushes the `:X.Y.Z`,
     `:X.Y`, and `:latest` tag family to `ghcr.io/sinity/polylogue`.
   - [`homebrew-bump.yml`](../.github/workflows/homebrew-bump.yml) — runs in
     parallel on the same tag, polls PyPI for the freshly published sdist,
     rewrites `url` / `sha256` / `version` in the formula at
     [`nix/homebrew-tap-template/Formula/polylogue.rb`](../nix/homebrew-tap-template/Formula/polylogue.rb),
     regenerates resource blocks via `brew update-python-resources`, runs
     `brew audit --strict --online`, and opens / refreshes a PR against
     [`Sinity/homebrew-tap`](https://github.com/Sinity/homebrew-tap).

The generated artifacts from this release graph are: top-level wheel + sdist,
wrapper wheel + sdist for `polylogue-mcp`, wrapper wheel + sdist for
`polylogue-hooks`, Sigstore bundles for every Python artifact, CycloneDX SBOM
JSON/XML for the top-level wheel dependency closure, packed browser-extension
artifacts from [`extension-release.yml`](../.github/workflows/extension-release.yml),
container images and push-path attestations from `container.yml`, and the
Homebrew tap PR produced after PyPI has the sdist.

The Homebrew bump workflow needs a `HOMEBREW_TAP_TOKEN` secret in this repo
— a fine-grained GitHub PAT with `contents: write` + `pull-requests: write`
on the tap repository. Configure it once under "Secrets and variables" →
"Actions" → "Repository secrets". Without it, the workflow logs a 404 when
checking out the tap and the bump PR is not opened.

If the release PR looks wrong (missing entries, wrong bump, stale title),
close it without merging and adjust the source commits — release-please will
recompute and open a new one on the next push to `master`.

PyPI Trusted Publishing requires a one-time registration on pypi.org under
"Publishing" → "Add a new pending publisher" with `owner = Sinity`,
`repo = polylogue`, `workflow = release.yml`, `environment = pypi`. If this
is missing the publish step fails with an OIDC error and the tag must be
re-cut after re-running the workflow.

The same release tag also publishes the wrapper distributions
[`polylogue-mcp`](https://pypi.org/project/polylogue-mcp/) and
[`polylogue-hooks`](https://pypi.org/project/polylogue-hooks/) (#1309). Each
PyPI project needs its own Trusted Publisher binding:

| PyPI project       | `environment`   | Wraps                                       |
| ------------------ | --------------- | ------------------------------------------- |
| `polylogue`        | `pypi`          | full archive runtime (CLI + daemon + MCP)   |
| `polylogue-mcp`    | `pypi-mcp`      | `polylogue-mcp` console script (pins `polylogue==X.Y.Z`) |
| `polylogue-hooks`  | `pypi-hooks`    | `polylogue-hook` console script (zero runtime deps) |

Register all three pending publishers before the first tagged release reaches
the `publish-pypi-mcp` / `publish-pypi-hooks` jobs. Each registration uses the
same `owner = Sinity`, `repo = polylogue`, `workflow = release.yml` triple and
differs only in the `environment` name.

Sigstore keyless signing requires no project-side configuration; it derives
identity from the workflow's `id-token: write` OIDC claim and publishes to
the public Sigstore transparency log. Consumers can verify the bundle with
`python -m sigstore verify identity --cert-identity
https://github.com/Sinity/polylogue/.github/workflows/release.yml@refs/tags/vX.Y.Z
--cert-oidc-issuer https://token.actions.githubusercontent.com
polylogue-X.Y.Z-py3-none-any.whl polylogue-X.Y.Z-py3-none-any.whl.sigstore`.

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
- [ ] Smoke `pip install polylogue-mcp==X.Y.Z` in a clean venv and confirm
      `polylogue-mcp --help` works (pins the matching `polylogue==X.Y.Z`).
- [ ] Smoke `pip install polylogue-hooks==X.Y.Z` in a clean venv and confirm
      `polylogue-hook SessionStart` accepts a sample JSON payload on stdin.
      `pip list` should show only `polylogue-hooks` plus pip/setuptools/wheel.
- [ ] Smoke the container: `podman run --rm
      ghcr.io/sinity/polylogue:X.Y.Z polylogue --version`.
- [ ] Confirm the `signing-artifacts-publish-pypi` workflow artifact on the
      release run contains a `.sigstore` bundle for each `*.whl` and `*.tar.gz`.
- [ ] Confirm the `sbom-artifacts` workflow artifact contains both
      `polylogue-X.Y.Z.cdx.json` and `polylogue-X.Y.Z.cdx.xml`.

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
