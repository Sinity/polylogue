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
