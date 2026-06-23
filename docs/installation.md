# Installation

Polylogue ships through several distribution channels. Pick the one that
matches how you run other tools on the host.

| Channel | Audience | Reference |
| --- | --- | --- |
| `pip install polylogue` | Python users on any OS | [PyPI](https://pypi.org/project/polylogue/) |
| `brew install sinity/tap/polylogue` | macOS / Linuxbrew users | see below |
| `nix run github:Sinity/polylogue` | NixOS / nix-darwin users | [`flake.nix`](../flake.nix) |
| `ghcr.io/sinity/polylogue` | Container / Kubernetes / Compose | see below |

The sections below cover ordinary Linux/Python installs, Nix/NixOS, containers,
and Homebrew. The same three console scripts are exposed by every channel:
`polylogue`, `polylogued`, and `polylogue-mcp`.

## Ordinary Linux / Python

Use `pipx` when you want Polylogue as a user-level tool without mixing it into
an application virtualenv:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install polylogue
polylogue --help
polylogue init
polylogued run
polylogued browser-capture status
```

A plain virtualenv works the same way and keeps the installed scripts under one
explicit directory:

```bash
python -m venv ~/.local/share/polylogue-venv
~/.local/share/polylogue-venv/bin/pip install --upgrade pip polylogue
~/.local/share/polylogue-venv/bin/polylogue init
~/.local/share/polylogue-venv/bin/polylogued run
```

For a system service, point the unit at the venv or package-managed
`polylogued` executable and set `POLYLOGUE_ARCHIVE_ROOT` only when the default
XDG archive location is not desired.

## Nix / NixOS quick start

The flake exposes app outputs for the CLI, daemon, and MCP server. One-shot
commands are enough for smoke testing a host:

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

On NixOS, import the flake module shown in the Nix section below. It creates a
managed `polylogued.service`; check that deployed surface with:

```bash
systemctl status polylogued.service
polylogue ops status
```

## Container image

The image is published to
[GitHub Container Registry](https://github.com/Sinity/polylogue/pkgs/container/polylogue)
by [`.github/workflows/container.yml`](../.github/workflows/container.yml).
Two variants ship from the same [`packaging/Containerfile`](../packaging/Containerfile):

| Tag pattern | Stage | Base | Notes |
| --- | --- | --- | --- |
| `:latest`, `:vX.Y.Z`, `:vX.Y`, `:master-<sha>` | `runtime` | `python:3.13-slim-bookworm` | Includes `curl` + `tini`. Default. `docker exec` debug supported. |
| `:latest-distroless`, `:vX.Y.Z-distroless`, … | `distroless` | `gcr.io/distroless/python3-debian12:nonroot` | No shell, no package manager. Smaller and tighter attack surface. |

Both variants are built for `linux/amd64` and `linux/arm64`.

### Volume contract

The image declares two volumes; mount these to host paths or named volumes:

| Mount point | Environment variable | Purpose |
| --- | --- | --- |
| `/data/archive` | `POLYLOGUE_ARCHIVE_ROOT` | SQLite database, FTS indexes, blob store. |
| `/etc/polylogue` | `POLYLOGUE_CONFIG_DIR` (`XDG_CONFIG_HOME`) | `polylogue.toml`, credentials, OAuth tokens. |

The runtime variant also exposes `8766/tcp` (daemon HTTP API).

Run as the built-in non-root user `polylogue` (UID/GID `10001`). When using
host bind-mounts, chown the target directories first:

```bash
mkdir -p /srv/polylogue/{archive,config}
sudo chown -R 10001:10001 /srv/polylogue
```

### Quick start

Foreground, daemon HTTP API on the host:

```bash
docker run --rm \
  -p 8766:8766 \
  -v polylogue-archive:/data/archive \
  -v polylogue-config:/etc/polylogue \
  ghcr.io/sinity/polylogue:latest
```

Check the health endpoint:

```bash
curl http://127.0.0.1:8766/api/health
```

Run the CLI inside the same image (the runtime variant ships `polylogue` and
`polylogue-mcp` on `PATH`):

```bash
docker run --rm \
  -v polylogue-archive:/data/archive \
  -v polylogue-config:/etc/polylogue \
  --entrypoint polylogue \
  ghcr.io/sinity/polylogue:latest --version
```

### docker compose

A complete sample lives at [`docs/docker-compose.yaml`](docker-compose.yaml).
It configures the volume contract, healthcheck, non-root UID, read-only root
filesystem, and `cap_drop: [ALL]`. Bring it up with:

```bash
docker compose -f docs/docker-compose.yaml up -d
docker compose -f docs/docker-compose.yaml ps
```

### Distroless variant

For deployments that prefer a smaller attack surface and don't need shell
access for debugging:

```bash
docker pull ghcr.io/sinity/polylogue:latest-distroless
docker run --rm -p 8766:8766 \
  -v polylogue-archive:/data/archive \
  ghcr.io/sinity/polylogue:latest-distroless
```

The distroless image omits `HEALTHCHECK` (no `curl` available); configure
liveness/readiness probes at the orchestrator (compose, Kubernetes) against
the same `GET /api/health` endpoint.

### Building locally

```bash
docker buildx build -f packaging/Containerfile --target runtime    -t polylogue:slim       .
docker buildx build -f packaging/Containerfile --target distroless -t polylogue:distroless .
```

For multi-arch local builds, set `--platform linux/amd64,linux/arm64` and
arrange a registry to push to (buildx cannot `load` a multi-platform image
into the local daemon).

### Backup

The archive lives entirely under `POLYLOGUE_ARCHIVE_ROOT`. Backups are file-
level snapshots of the named volume / bind-mount target. Stop the daemon (or
use the SQLite `.backup` command from inside the running container) before
copying to avoid catching mid-checkpoint state.

## Homebrew

Polylogue ships through a dedicated tap at
[`Sinity/homebrew-tap`](https://github.com/Sinity/homebrew-tap). The tap is
auto-bumped from PyPI by
[`.github/workflows/homebrew-bump.yml`](../.github/workflows/homebrew-bump.yml)
on every `vX.Y.Z` tag push, so `brew upgrade polylogue` tracks the most
recent release without manual formula edits. The formula shape itself is
templated in [`nix/homebrew-tap-template/`](../nix/homebrew-tap-template/)
so future edits flow through this repo.

The formula installs into a private virtualenv under
`$(brew --prefix)/Cellar/polylogue/X.Y.Z`, depending on `python@3.13`,
and exposes three binaries on `PATH`:

| Binary | Role |
| --- | --- |
| `polylogue` | Query CLI for the archive. |
| `polylogued` | Local convergence + HTTP daemon. |
| `polylogue-mcp` | MCP server for AI coding agents. |

### One-shot install

```bash
brew tap sinity/tap
brew install polylogue
polylogue --version
```

### Daemon service (opt-in)

The formula declares a `service` block, but Polylogue does not install a
system service by default. Start the daemon explicitly:

```bash
brew services start polylogued
brew services info polylogued
```

This wires a LaunchAgent on macOS and a systemd-user unit on Linuxbrew.
First-run creates the XDG archive under
`~/Library/Application Support/polylogue/` (macOS) or
`${XDG_DATA_HOME:-~/.local/share}/polylogue/` (Linux). Override with
`POLYLOGUE_ARCHIVE_ROOT` to relocate the database, FTS indexes, and blob
store.

### Local iteration on the formula

When changing the template, verify the candidate before publishing:

```bash
brew install --build-from-source nix/homebrew-tap-template/Formula/polylogue.rb
brew test polylogue
brew audit --strict --online polylogue
```

The bump workflow runs `brew update-python-resources` followed by the
same `brew audit --strict --online` gate before opening the PR against
the tap.

## Nix flake

Polylogue ships as a flake. The flake exposes three apps, a default package,
a NixOS module, and a Home Manager module. Build artifacts for `x86_64-linux`
are mirrored to the [`polylogue` cachix
cache](https://polylogue.cachix.org); enable it on a fresh host to avoid
rebuilding the dependency closure.

The flake is also published to [FlakeHub](https://flakehub.com/flake/Sinity/polylogue)
for discoverability — either input URL form (`github:` or `https://flakehub.com/f/...`)
resolves to the same flake.

### One-shot invocation

```bash
nix run github:Sinity/polylogue                       # polylogue (CLI)
nix run github:Sinity/polylogue#polylogued -- --help  # daemon
nix run github:Sinity/polylogue#polylogue-mcp         # MCP server
```

Enable the binary cache for hit-on-first-pull:

```bash
nix-env -iA cachix -f https://cachix.org/api/v1/install   # if cachix is not on PATH
cachix use polylogue
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
      daemon.port = 8766;
    };
  };
}
```

The module wires a `polylogued.service` systemd unit running `polylogued run`
(watch + browser-capture + HTTP API). All `services.polylogue.settings.*`
options map to entries in the generated `polylogue.toml` and are documented
inline in [`nix/module.nix`](../nix/module.nix). Per-unit hardening defaults
(`Nice = 10`, `IOSchedulingClass = idle`, `MemoryHigh = 1G`, `MemoryMax = 2G`)
are tunable under `services.polylogue.service.*`.

### Home Manager module

```nix
{ inputs, pkgs, ... }: {
  imports = [ inputs.polylogue.homeManagerModules.default ];

  programs.polylogued = {
    enable = true;
    package = inputs.polylogue.packages.${pkgs.system}.default;
    settings.daemon.port = 8766;
  };
}
```

The HM module declares a `systemd.user.services.polylogued` unit. Set
`programs.polylogued.autoStart = false;` to keep the unit available but not
started at login.

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
| `store-screenshots-<tag>.tar.gz` | — | Submission media for store listings |

Download the artifacts from the
[latest release](https://github.com/Sinity/polylogue/releases/latest).
The version is locked to the polylogue Python project version on the same
tag — the workflow rewrites `manifest.json` from `pyproject.toml` before
packaging so the extension version always matches the daemon protocol it
was built against.

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
| Packed `.zip` on GitHub Releases | shipped | this doc |
| Packed `.xpi` on GitHub Releases | shipped | this doc |
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
