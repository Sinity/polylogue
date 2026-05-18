# Installation

Polylogue ships through several distribution channels. Pick the one that
matches how you run other tools on the host.

| Channel | Audience | Reference |
| --- | --- | --- |
| `pip install polylogue` | Python users on any OS | [PyPI](https://pypi.org/project/polylogue/) |
| `nix run github:Sinity/polylogue` | NixOS / nix-darwin users | [`flake.nix`](../flake.nix) |
| `ghcr.io/sinity/polylogue` | Container / Kubernetes / Compose | see below |

This document focuses on the container channel; the other channels are linked
out so they can evolve independently.

## Container image

The image is published to
[GitHub Container Registry](https://github.com/Sinity/polylogue/pkgs/container/polylogue)
by [`.github/workflows/container.yml`](../.github/workflows/container.yml).
Two variants ship from the same `Containerfile`:

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
docker buildx build --target runtime    -t polylogue:slim       .
docker buildx build --target distroless -t polylogue:distroless .
```

For multi-arch local builds, set `--platform linux/amd64,linux/arm64` and
arrange a registry to push to (buildx cannot `load` a multi-platform image
into the local daemon).

### Backup

The archive lives entirely under `POLYLOGUE_ARCHIVE_ROOT`. Backups are file-
level snapshots of the named volume / bind-mount target. Stop the daemon (or
use the SQLite `.backup` command from inside the running container) before
copying to avoid catching mid-checkpoint state.

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
