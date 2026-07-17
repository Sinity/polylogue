# Installation

Polylogue is available from PyPI, the Sinity Homebrew tap, and its Nix flake.
Container and browser-extension artifacts remain version-specific: a GitHub
tag or release page alone is not proof that those channels carry a matching
artifact, so use the linked verification for the channel you plan to install
from.

| Channel | Audience | Install after its artifact smoke passes | Verification |
| --- | --- | --- | --- |
| `nix develop` | Local development, source checkout, verification | `nix develop` | [Contributing](../CONTRIBUTING.md) |
| Nix flake | NixOS / nix-darwin users | `nix run github:Sinity/polylogue -- --help` | [`flake.nix`](../flake.nix) |
| PyPI / pipx | Python CLI users | `pipx install polylogue` | `polylogue --version`, `polylogued --help`, `polylogue-mcp --help` |
| GHCR | Container deployments | `podman pull ghcr.io/sinity/polylogue:X.Y.Z` | `podman run --rm --entrypoint polylogue ghcr.io/sinity/polylogue:X.Y.Z --version` |
| Homebrew | macOS/Linuxbrew users | `brew tap sinity/polylogue && brew install polylogue` | `brew test polylogue` |
| Browser extension | Browser-capture users | Download the matching release `.zip` / `.xpi` | verify `build-manifest.json` version equals `X.Y.Z` |
| NixOS / Home Manager module | Managed local daemon deployment | see below | `systemctl status polylogued.service` |

For the release-channel recovery sequence and exact smoke matrix, see the
[release checklist](release.md#backfill-an-existing-tag-without-re-cutting-it).

The source and Nix paths expose the same three console scripts: `polylogue`,
`polylogued`, and `polylogue-mcp`.

## Source checkout

Clone the repository and enter the dev shell:

```bash
git clone https://github.com/Sinity/polylogue.git
cd polylogue
nix develop
```

From inside the dev shell:

```bash
polylogue --help
polylogue ops status
polylogued run
polylogue-mcp --help
```

Use this route for local work, operator dogfooding, browser-capture debugging,
and repository verification. The dev shell also provides `devtools`, which owns
generated docs, verification, deployment smoke probes, and release checks.

## Nix / NixOS quick start

The flake exposes app outputs for the CLI, daemon, and MCP server. Use
`github:Sinity/polylogue` for a one-shot smoke of the current published flake,
or `nix develop` inside a checkout when you need the repo-local `devtools`
surface. One-shot commands are enough for smoke testing a host:

```bash
nix run github:Sinity/polylogue -- --help
nix run github:Sinity/polylogue#polylogued -- run
nix run github:Sinity/polylogue#polylogued -- browser-capture status
nix run github:Sinity/polylogue#polylogue-mcp -- --help
```

Inside a checkout, use the dev shell when you want the repo-local command
surface and verification helpers:

```bash
nix develop -c polylogue --help
nix develop -c polylogued run
nix develop -c devtools workspace deployment-smoke --browser --browser-executable "$(command -v google-chrome)"
```

On NixOS hosts where Chrome lives in the per-user profile rather than under a
distribution path, pass that path explicitly for browser first-paint diagnostics:

```bash
nix develop -c devtools workspace deployment-smoke --browser \
  --browser-executable /etc/profiles/per-user/$USER/bin/google-chrome
```

That smoke launches a fresh headless profile against Polylogue's web root. It
does not claim MCP DevTools control, copied-profile cookies, or live provider
pages; those are local operator proof modes covered by
[`docs/dev-loop.md`](dev-loop.md).

On NixOS, import the flake module shown in the Nix section below. It creates a
managed `polylogued.service`; check that deployed surface with:

```bash
systemctl status polylogued.service
polylogue ops status
```

## PyPI and Homebrew

`pipx` is the recommended isolated Python CLI install. `uv tool` provides the
same isolation model if uv is already your package frontend:

```bash
pipx install polylogue
# or: uv tool install polylogue
polylogue --version
polylogued --help
polylogue-mcp --help
```

The Homebrew tap installs the published PyPI release into a formula-owned
virtual environment:

```bash
brew tap sinity/polylogue
brew install polylogue
brew test polylogue
```

## Containers

For containers, use `--version` for the concise human build identity and query
the packaged runtime's `VERSION_INFO.commit` for the full revision before
deploying:

```bash
podman pull ghcr.io/sinity/polylogue:X.Y.Z
podman run --rm --entrypoint polylogue ghcr.io/sinity/polylogue:X.Y.Z --version
podman run --rm --entrypoint python ghcr.io/sinity/polylogue:X.Y.Z \
  -I -c 'from polylogue.version import VERSION_INFO; print(VERSION_INFO.commit)'
git ls-remote https://github.com/Sinity/polylogue refs/tags/vX.Y.Z
```

The published workflow performs this smoke for both slim and distroless images;
it requires the concise version to match the source prefix and the machine-
readable runtime field to equal the complete source revision.

### Recovering an already-created tag

If a GitHub Release exists without its artifacts, do not re-cut its tag. Follow
the [release checklist recovery sequence](release.md#backfill-an-existing-tag-without-re-cutting-it),
which requires the tag, checked-out source, and GitHub Release target commit to
agree before publication.

For local container or formula experiments before their artifacts exist, build
from the checkout:

```bash
docker buildx build -f packaging/Containerfile --target runtime    -t polylogue:slim       .
docker buildx build -f packaging/Containerfile --target distroless -t polylogue:distroless .
```

For formula work, build the in-repo template explicitly:

```bash
brew install --build-from-source nix/homebrew-tap-template/Formula/polylogue.rb
brew test polylogue
brew audit --strict --online polylogue
```

## Nix flake

The flake exposes three apps, a default package, a NixOS module, and a Home
Manager module. Build artifacts for `x86_64-linux` are mirrored to the
[`sinity` cachix cache](https://sinity.cachix.org); enable it on a fresh host
to avoid rebuilding the dependency closure.

### One-shot invocation

```bash
nix run github:Sinity/polylogue                       # polylogue (CLI)
nix run github:Sinity/polylogue#polylogued -- --help  # daemon
nix run github:Sinity/polylogue#polylogue-mcp         # MCP server
```

Enable the binary cache for hit-on-first-pull:

```bash
nix-env -iA cachix -f https://cachix.org/api/v1/install   # if cachix is not on PATH
cachix use sinity
```

### External flake input

```nix
{
  inputs.polylogue.url = "github:Sinity/polylogue";
  # or, via FlakeHub:
  # inputs.polylogue.url = "https://flakehub.com/f/Sinity/polylogue/*";

  outputs = { self, nixpkgs, polylogue, ... }: {
    # Use polylogue.packages.x86_64-linux.default in any derivation
    # or wire the NixOS / HM module below.
  };
}
```

### NixOS module

```nix
{ inputs, pkgs, ... }: {
  imports = [ inputs.polylogue.nixosModules.default ];

  services.polylogue = {
    enable = true;
    package = inputs.polylogue.packages.${pkgs.system}.default;
    settings = {
      archive.root = "/var/lib/polylogue";
      daemon-api.port = 8766;
      browser-capture.port = 8765;
      embedding.enabled = false;
    };
  };
}
```

The module wires a `polylogued.service` systemd unit running `polylogued run`
(watch + browser-capture + HTTP API). `services.polylogue.settings.*` renders
the same `polylogue.toml` schema used outside Nix; `polylogued run` reads that
effective config instead of receiving duplicated `ExecStart` flags. Keep bearer
tokens in environment/secret-manager wiring when possible, because setting them
in `settings.*` writes them into the generated TOML. Per-unit hardening defaults
(`Nice = 10`, `IOSchedulingClass = idle`, `MemoryHigh = 2G`, `MemoryMax = 2G`)
are tunable under `services.polylogue.service.*`; they are deployment policy,
not archive config.

After deploying, use `polylogue config --format json` to inspect effective
values and source layers, and `devtools workspace deployment-smoke --json` to capture the
package versions, archive root, daemon URL, browser-capture receiver URL,
optional browser executable path, and cgroup resource signals visible on the
host.

### Home Manager module

```nix
{ inputs, pkgs, ... }: {
  imports = [ inputs.polylogue.homeManagerModules.default ];

  programs.polylogued = {
    enable = true;
    package = inputs.polylogue.packages.${pkgs.system}.default;
    settings.daemon-api.port = 8766;
  };
}
```

The HM module declares a `systemd.user.services.polylogued` unit and renders the
same TOML schema as the NixOS module. Set `programs.polylogued.autoStart =
false;` to keep the unit available but not started at login.

### Building locally

```bash
nix build .#polylogue           # the package
nix build .#api-python          # python interpreter with polylogue importable
nix flake check                 # smoke + format + lint
```

## Browser extension

The Manifest V3 browser-capture extension lives in
[`browser-extension/`](../browser-extension/) and ships as packed artifacts
attached to every GitHub release by
[`.github/workflows/extension-release.yml`](../.github/workflows/extension-release.yml).

| Artifact | Browsers | Use |
| --- | --- | --- |
| `polylogue-browser-capture-<version>-chrome.zip` | Chrome, Chromium, Edge, Brave, Vivaldi | Developer-mode unpacked install |
| `polylogue-browser-capture-<version>-firefox.xpi` | Firefox | Temporary install via `about:debugging` |
| `polylogue-browser-capture-<version>-store-submission.json` | Store operators | Version, source revision, Gecko ID, SHA-256, and size for both archives |
| `store-screenshots-<tag>.tar.gz` | — | Submission media for store listings |

Download the artifacts from the
[latest release](https://github.com/Sinity/polylogue/releases/latest).
The version is locked to the polylogue Python project version on the same
tag — the workflow rewrites `manifest.json` from `pyproject.toml` before
packaging so the extension version always matches the daemon protocol it
was built against. Before uploading to a store, verify each downloaded archive
against the matching `store-submission.json`; it binds the pair to the release
version and complete source revision.

### Install paths

The full per-browser steps live in
[`browser-extension/README.md`](../browser-extension/README.md#install). At
a glance:

1. **Start the receiver.** `polylogued browser-capture serve` (or
   `polylogued run` for the full daemon). Default endpoint
   `http://127.0.0.1:8765`.
2. **Chrome / Chromium (unpacked from clone).** `chrome://extensions` →
   Developer mode → Load unpacked → select `browser-extension/`.
3. **Chrome / Chromium (packed `.zip`).** Download the release `.zip`,
   extract, then Load unpacked against the extracted directory.
4. **Firefox.** Download the release `.xpi`. Visit
   `about:debugging#/runtime/this-firefox` → Load Temporary Add-on → select
   the `.xpi`.

### Distribution status

| Channel | Status | Tracking |
| --- | --- | --- |
| Packed `.zip` on GitHub Releases | release-backed; verify the matching asset exists | this doc |
| Packed `.xpi` on GitHub Releases | release-backed; verify the matching asset exists | this doc |
| Chrome Web Store | not submitted (store sign-up + review have non-engineering blockers) | follow-up |
| Mozilla AMO | not submitted (same blockers) | follow-up |

The extension is local-only by design — it talks to `127.0.0.1:8765` and
nothing else. See [docs/browser-capture.md](browser-capture.md) for the
receiver protocol and `docs/daemon-threat-model.md` for the security model.

### Local rebuild

```bash
cd browser-extension
npm install
npm run validate    # in-tree manifest validator
npm run build       # → dist/polylogue-browser-capture-*-{chrome.zip,firefox.xpi}
npm run screenshots # optional; needs Playwright Chromium
```

`scripts/build.mjs --version X.Y.Z` overrides the autodetected version when
producing a one-off bundle.
