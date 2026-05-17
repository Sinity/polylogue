# syntax=docker/dockerfile:1.7
#
# Polylogue OCI image. Multi-stage:
#   1. builder — uses `uv` to build the wheel from the checkout
#   2. runtime — slim python image, installs only the built wheel
#
# Exposes three console scripts (polylogue, polylogued, polylogue-mcp) on PATH.
# Operators are expected to mount durable state:
#   -v polylogue-data:/data    (XDG_DATA_HOME — archive SQLite, blobs)
#   -v polylogue-config:/config (XDG_CONFIG_HOME — polylogue.toml, secrets)
#
# Example invocations:
#   podman run --rm ghcr.io/sinity/polylogue:latest polylogue --version
#   podman run --rm -v polylogue-data:/data ghcr.io/sinity/polylogue:latest \
#     polylogue stats
#   podman run -d --name polylogued -v polylogue-data:/data \
#     -v polylogue-config:/config -p 7777:7777 \
#     ghcr.io/sinity/polylogue:latest polylogued run --enable-api

# ---- builder ------------------------------------------------------------
FROM python:3.13-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_PROGRESS=1 \
    UV_LINK_MODE=copy

# uv is the canonical build driver in this repo.
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /src
COPY . /src

# Build wheel into /dist. Hatchling embeds polylogue/_build_info.py via
# hatch_build.py, so the resulting wheel knows its own version + commit.
RUN uv build --wheel --out-dir /dist /src

# ---- runtime ------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POLYLOGUE_FORCE_PLAIN=1 \
    XDG_DATA_HOME=/data \
    XDG_CONFIG_HOME=/config \
    XDG_CACHE_HOME=/cache

# Runtime deps:
#   ca-certificates — outbound HTTPS (PyPI, Voyage, Drive)
#   tini             — PID 1 for clean SIGTERM/SIGINT signalling
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

# Install the built wheel without dev/extras. Dependencies are resolved
# from PyPI against the wheel's declared metadata.
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Non-root runtime user; archive writes happen under /data so it must own
# the mount target. Operators that bind-mount a host path are expected to
# either chown it to UID 1000 or override --user explicitly.
RUN groupadd --system --gid 1000 polylogue \
 && useradd  --system --uid 1000 --gid polylogue --home /home/polylogue --create-home polylogue \
 && mkdir -p /data /config /cache \
 && chown -R polylogue:polylogue /data /config /cache

USER polylogue
WORKDIR /home/polylogue
VOLUME ["/data", "/config"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["polylogue", "--help"]
