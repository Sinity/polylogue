"""Verify immutable campaign-genesis evidence without accessing live task state.

The reindex campaign's genesis records the exact historical Git blobs that
were reviewed during its migration.  This module deliberately reads only
those pinned blobs: it never discovers, imports, mutates, or republishes a
current task tracker.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

CAMPAIGN_GENESIS_SCHEMA = "polylogue.campaign-genesis/v1"
CAMPAIGN_ID = "reindex-2026"
SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class CampaignGenesis:
    """A verified, revision-pinned historical campaign input set."""

    campaign_id: str
    snapshots: dict[str, tuple[str, str, str]]


def verify_campaign_genesis(path: Path, *, cwd: Path | None = None) -> CampaignGenesis:
    """Load ``path`` and prove every declared historical Git blob digest.

    ``cwd`` selects the Git repository containing the historical objects.  No
    working-tree file except the supplied genesis record is read.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"campaign genesis is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("campaign genesis must be an object")
    expected = {"schema", "campaign_id", "input_snapshot", "migration_snapshot", "formula_snapshot"}
    if set(payload) != expected or payload.get("schema") != CAMPAIGN_GENESIS_SCHEMA:
        raise RuntimeError("campaign genesis has an unexpected schema")
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("campaign genesis identity is invalid")

    snapshots: dict[str, tuple[str, str, str]] = {}
    for key in ("input_snapshot", "migration_snapshot", "formula_snapshot"):
        snapshot = payload[key]
        if not isinstance(snapshot, dict) or set(snapshot) != {"revision", "path", "sha256"}:
            raise RuntimeError(f"campaign genesis {key} must name revision, path, and sha256")
        revision, blob_path, expected_digest = (snapshot[field] for field in ("revision", "path", "sha256"))
        if not all(isinstance(value, str) and value for value in (revision, blob_path, expected_digest)):
            raise RuntimeError(f"campaign genesis {key} contains an invalid value")
        if len(expected_digest) != SHA256_LENGTH or any(char not in "0123456789abcdef" for char in expected_digest):
            raise RuntimeError(f"campaign genesis {key} sha256 is invalid")
        resolved_revision = _git_revision(revision, cwd=cwd)
        blob = _git_blob(resolved_revision, blob_path, cwd=cwd)
        actual_digest = hashlib.sha256(blob).hexdigest()
        if actual_digest != expected_digest:
            raise RuntimeError(f"campaign genesis {key} digest does not match its pinned Git object")
        snapshots[key] = (resolved_revision, blob_path, actual_digest)
    return CampaignGenesis(campaign_id=CAMPAIGN_ID, snapshots=snapshots)


def _git_revision(revision: str, *, cwd: Path | None) -> str:
    if revision.startswith("-"):
        raise RuntimeError("revision must be a non-option Git revision")
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"unable to resolve revision {revision!r}")
    return result.stdout.strip()


def _git_blob(revision: str, path: str, *, cwd: Path | None) -> bytes:
    result = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=cwd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"unable to read {path!r} at revision {revision!r}")
    return result.stdout
