"""Local attestation for backup verification receipts.

The attestation prevents a backup directory from authorizing a durable
migration using only public, locally recomputable hashes.  It protects against
artifact-only fabrication and accidental transplantation.  It is not a
privilege boundary against arbitrary code running as the same Unix user,
which can read the per-tier key.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

from polylogue.paths import state_home

VERIFICATION_RECEIPT_FORMAT = "polylogue-backup-verification-receipt-v2"
ATTESTATION_FORMAT = "polylogue-backup-verification-attestation-v1"
ATTESTATION_ALGORITHM = "hmac-sha256"
ATTESTATION_KEY_BYTES = 32

_DOMAIN = b"polylogue:backup-verification:v2\0"
_KEY_DIR = "backup-attestations"


class BackupAttestationError(RuntimeError):
    """Raised when a receipt lacks valid local verification authority."""


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def tier_attestation_id(live_tier_path: Path) -> str:
    """Return the stable local identity for one resolved durable tier."""
    canonical = str(live_tier_path.expanduser().resolve(strict=False)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def attestation_key_path(live_tier_path: Path) -> Path:
    """Resolve the independently-derived key path for a live durable tier."""
    return state_home() / _KEY_DIR / f"{tier_attestation_id(live_tier_path)}.key"


def _prepare_key_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise BackupAttestationError(f"backup attestation key directory is not a real directory: {path}")
    if metadata.st_uid != os.geteuid():
        raise BackupAttestationError(f"backup attestation key directory is not owned by the current user: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        path.chmod(0o700)


def _load_key(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise BackupAttestationError(f"backup attestation key is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise BackupAttestationError(f"backup attestation key is not a regular file: {path}")
    if metadata.st_uid != os.geteuid():
        raise BackupAttestationError(f"backup attestation key is not owned by the current user: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise BackupAttestationError(f"backup attestation key permissions are too broad: {path}")
    key = path.read_bytes()
    if len(key) != ATTESTATION_KEY_BYTES:
        raise BackupAttestationError(f"backup attestation key has invalid length: {path}")
    return key


def load_attestation_key(live_tier_path: Path) -> bytes:
    """Load the existing per-tier attestation key without minting it."""
    return _load_key(attestation_key_path(live_tier_path))


def load_or_mint_attestation_key(live_tier_path: Path) -> bytes:
    """Load or atomically mint the per-tier attestation key."""
    path = attestation_key_path(live_tier_path)
    _prepare_key_directory(path.parent)
    if path.exists():
        return _load_key(path)

    key = os.urandom(ATTESTATION_KEY_BYTES)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(key)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return _load_key(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)
    return _load_key(path)


def _attestation_message(receipt: dict[str, object], metadata: dict[str, object]) -> bytes:
    return _DOMAIN + _canonical_bytes(receipt) + b"\0" + _canonical_bytes(metadata)


def sign_verification_receipt(receipt: dict[str, object], *, authority_paths: Mapping[str, Path]) -> dict[str, object]:
    """Attach one domain-separated HMAC for each durable tier authority."""
    attestations: list[dict[str, object]] = []
    for tier, live_tier_path in sorted(authority_paths.items()):
        key = load_or_mint_attestation_key(live_tier_path)
        metadata: dict[str, object] = {
            "format": ATTESTATION_FORMAT,
            "algorithm": ATTESTATION_ALGORITHM,
            "tier": tier,
            "resource_id": tier_attestation_id(live_tier_path),
            "key_id": hashlib.sha256(key).hexdigest(),
        }
        mac = hmac.new(key, _attestation_message(receipt, metadata), hashlib.sha256).hexdigest()
        attestations.append({**metadata, "mac": mac})
    return {**receipt, "attestations": attestations}


def verify_verification_receipt(receipt: dict[str, object], *, tier: str, live_tier_path: Path) -> None:
    """Fail closed unless ``receipt`` carries the live tier's valid HMAC."""
    raw_attestations = receipt.get("attestations")
    if not isinstance(raw_attestations, list):
        raise BackupAttestationError("backup verification receipt has no authenticated attestations")
    matching = [item for item in raw_attestations if isinstance(item, dict) and item.get("tier") == tier]
    if len(matching) != 1:
        raise BackupAttestationError(f"backup verification receipt has no unique {tier} attestation")
    raw_attestation = matching[0]
    expected_resource_id = tier_attestation_id(live_tier_path)
    if raw_attestation.get("format") != ATTESTATION_FORMAT:
        raise BackupAttestationError("backup verification receipt attestation format is unsupported")
    if raw_attestation.get("algorithm") != ATTESTATION_ALGORITHM:
        raise BackupAttestationError("backup verification receipt attestation algorithm is unsupported")
    if raw_attestation.get("resource_id") != expected_resource_id:
        raise BackupAttestationError("backup verification receipt belongs to a different live tier")

    key = load_attestation_key(live_tier_path)
    if raw_attestation.get("key_id") != hashlib.sha256(key).hexdigest():
        raise BackupAttestationError("backup verification receipt attestation key does not match")
    actual_mac = raw_attestation.get("mac")
    if not isinstance(actual_mac, str) or len(actual_mac) != 64:
        raise BackupAttestationError("backup verification receipt has no valid attestation MAC")

    unsigned_attestation = {key_: value for key_, value in raw_attestation.items() if key_ != "mac"}
    unsigned_receipt = {key_: value for key_, value in receipt.items() if key_ != "attestations"}
    expected_mac = hmac.new(
        key,
        _attestation_message(unsigned_receipt, unsigned_attestation),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(actual_mac, expected_mac):
        raise BackupAttestationError("backup verification receipt attestation MAC is invalid")


__all__ = [
    "ATTESTATION_ALGORITHM",
    "ATTESTATION_FORMAT",
    "BackupAttestationError",
    "VERIFICATION_RECEIPT_FORMAT",
    "attestation_key_path",
    "load_attestation_key",
    "load_or_mint_attestation_key",
    "sign_verification_receipt",
    "tier_attestation_id",
    "verify_verification_receipt",
]
