"""Receiver-authoritative queue for authenticated ChatGPT Chat launch jobs."""

from __future__ import annotations

import base64
import binascii
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import wraps
from io import BytesIO
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, TypeVar, cast
from uuid import uuid4

import orjson

from polylogue.browser_capture.models import (
    BrowserLaunchAttachment,
    BrowserLaunchEvent,
    BrowserLaunchHandoffRequest,
    BrowserLaunchJob,
    BrowserLaunchJobControlRequest,
    BrowserLaunchJobRequest,
    BrowserLaunchJobUpdateRequest,
)
from polylogue.browser_capture.sol_pro_prompt import (
    SOL_PRO_PROMPT_PROFILE,
    build_sol_pro_prompt,
    sol_pro_prompt_sha256,
)
from polylogue.paths import browser_capture_spool_root

LAUNCH_JOB_DIRNAME = "launch-jobs"
LAUNCH_JOB_MAX_FILES = 2_000
LAUNCH_ATTACHMENT_MAX_BYTES = 16 * 1024 * 1024
LAUNCH_JOB_MAX_ATTACHMENT_BYTES = 16 * 1024 * 1024
LAUNCH_EVENT_LIMIT = 200
LAUNCH_LEASE_SECONDS = 180
LAUNCH_HANDOFF_MAX_BYTES = 64 * 1024 * 1024
LAUNCH_HANDOFF_MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
LAUNCH_HANDOFF_MAX_FILES = 2_000
LAUNCH_HANDOFF_NAME = "polylogue-sol-pro-launch-handoff.zip"
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")
_LAUNCH_LOCK = RLock()
_T = TypeVar("_T")


def _serialized(function: Callable[..., _T]) -> Callable[..., _T]:
    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> _T:
        with _LAUNCH_LOCK:
            root = launch_job_root(cast(Path | None, kwargs.get("spool_path")))
            root.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(root / ".queue.lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                return function(*args, **kwargs)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    return cast(Callable[..., _T], guarded)


class BrowserLaunchConflictError(RuntimeError):
    """Raised when a job id or state transition conflicts with durable state."""


class BrowserLaunchLeaseError(RuntimeError):
    """Raised when a non-owner attempts to mutate a leased job."""


class BrowserLaunchQuotaError(RuntimeError):
    """Raised when receiver-owned launch storage would exceed its bound."""


class BrowserLaunchHandoffError(RuntimeError):
    """Raised when a provider result is not the required cohesive handoff."""


class BrowserLaunchStateError(RuntimeError):
    """Raised when durable launch state exists but cannot be read safely."""


def launch_job_root(spool_path: Path | None = None) -> Path:
    base = spool_path if spool_path is not None else browser_capture_spool_root()
    return base / LAUNCH_JOB_DIRNAME


def _safe_token(value: str) -> str:
    token = _SAFE_TOKEN.sub("-", value).strip("-.")
    if not token:
        raise ValueError("empty launch token")
    return token[:160]


def _job_dir(root: Path, job_id: str) -> Path:
    return root / _safe_token(job_id)


def _job_path(root: Path, job_id: str) -> Path:
    return _job_dir(root, job_id) / "job.json"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_job(root: Path, job: BrowserLaunchJob) -> None:
    raw = orjson.dumps(
        job.model_dump(mode="json", exclude_none=True), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
    )
    _atomic_write(_job_path(root, job.job_id), raw + b"\n")


def _read_job(path: Path) -> BrowserLaunchJob | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        raise
    try:
        return BrowserLaunchJob.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        raise BrowserLaunchStateError(f"corrupt launch job state: {path.name}") from exc


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _backoff_with_jitter(job: BrowserLaunchJob, base_seconds: int, *, cap_seconds: int) -> int:
    exponential = min(cap_seconds, base_seconds * (2 ** min(max(job.attempts - 1, 0), 5)))
    seed = hashlib.sha256(f"{job.job_id}:{job.attempts}:{job.cooldown_reason}".encode()).digest()[0]
    return int(exponential + int(exponential * 0.1 * seed / 255))


def _event(job: BrowserLaunchJob, kind: str, *, detail: str | None = None, owner: str | None = None) -> None:
    job.events = [
        *job.events,
        BrowserLaunchEvent(event_id=uuid4().hex, at=_iso(_now()), kind=kind, detail=detail, owner_instance_id=owner),
    ][-LAUNCH_EVENT_LIMIT:]


def list_launch_jobs(*, spool_path: Path | None = None) -> list[BrowserLaunchJob]:
    root = launch_job_root(spool_path)
    if not root.exists():
        return []
    jobs = [_read_job(path) for path in root.glob("*/job.json")]
    return sorted(
        (job for job in jobs if job is not None),
        key=lambda job: (job.created_at, job.queue_position, job.job_id),
        reverse=True,
    )


@_serialized
def enqueue_launch_job(request: BrowserLaunchJobRequest, *, spool_path: Path | None = None) -> BrowserLaunchJob:
    root = launch_job_root(spool_path)
    root.mkdir(parents=True, exist_ok=True)
    if sum(1 for _ in root.glob("*/job.json")) >= LAUNCH_JOB_MAX_FILES:
        raise BrowserLaunchQuotaError("launch job file quota exceeded")
    job_id = _safe_token(request.job_id) if request.job_id else uuid4().hex
    target = _job_dir(root, job_id)
    if target.exists():
        raise BrowserLaunchConflictError(f"launch job already exists: {job_id}")

    attachments: list[BrowserLaunchAttachment] = []
    decoded: list[tuple[BrowserLaunchAttachment, bytes]] = []
    total = 0
    for index, item in enumerate(request.attachments):
        try:
            content = base64.b64decode(item.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"invalid base64 for launch attachment {item.name!r}") from exc
        if len(content) > LAUNCH_ATTACHMENT_MAX_BYTES:
            raise BrowserLaunchQuotaError(f"launch attachment exceeds {LAUNCH_ATTACHMENT_MAX_BYTES} bytes")
        total += len(content)
        if total > LAUNCH_JOB_MAX_ATTACHMENT_BYTES:
            raise BrowserLaunchQuotaError("launch job attachment byte quota exceeded")
        digest = hashlib.sha256(content).hexdigest()
        attachment = BrowserLaunchAttachment(
            attachment_id=f"a{index + 1}-{digest[:16]}",
            name=item.name,
            mime_type=item.mime_type,
            size_bytes=len(content),
            sha256=digest,
        )
        attachments.append(attachment)
        decoded.append((attachment, content))

    now = _now()
    queue_position = max((job.queue_position for job in list_launch_jobs(spool_path=spool_path)), default=0) + 1
    parsed_not_before = _parse(request.not_before)
    if request.not_before is not None and parsed_not_before is None:
        raise ValueError("launch not_before must be an ISO-8601 timestamp")
    not_before = parsed_not_before or now
    job = BrowserLaunchJob(
        job_id=job_id,
        prompt_profile=SOL_PRO_PROMPT_PROFILE,
        prompt_prefix_sha256=sol_pro_prompt_sha256(),
        job_title=request.job_title,
        scope_prompt=request.scope_prompt,
        prompt=build_sol_pro_prompt(request.job_title, request.scope_prompt),
        attachments=attachments,
        cadence_minutes=request.cadence_minutes,
        required_output=request.required_output,
        mode=request.mode,
        model_slug=request.model_slug,
        model_label=request.model_label,
        effort_label=request.effort_label,
        thinking_effort=request.thinking_effort,
        queue_position=queue_position,
        created_at=_iso(now),
        updated_at=_iso(now),
        not_before=_iso(not_before),
        next_attempt_at=_iso(not_before),
    )
    _event(job, "created")
    try:
        for attachment, content in decoded:
            _atomic_write(target / "attachments" / attachment.attachment_id, content)
        _write_job(root, job)
    except Exception:
        if target.exists():
            for path in sorted(target.glob("**/*"), reverse=True):
                if path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    path.rmdir()
            target.rmdir()
        raise
    return job


def _latest_submitted_at(jobs: list[BrowserLaunchJob]) -> datetime | None:
    timestamps = [
        _parse(event.at)
        for job in jobs
        for event in job.events
        if event.kind == "submitted" and _parse(event.at) is not None
    ]
    return max((value for value in timestamps if value is not None), default=None)


@_serialized
def claim_due_launch_job(
    owner_instance_id: str,
    *,
    spool_path: Path | None = None,
    lease_seconds: int = LAUNCH_LEASE_SECONDS,
) -> BrowserLaunchJob | None:
    """Lease one due submission while already-submitted chats continue in parallel."""

    root = launch_job_root(spool_path)
    jobs = list_launch_jobs(spool_path=spool_path)
    now = _now()
    # Upload/preflight/submit is the rate-sensitive critical section. A
    # submitted chat retains its monitor owner, but must not serialize the
    # potentially hour-long provider response; burst cadence is measured
    # between successful submissions instead.
    active = {"leased", "uploading", "submitting"}
    for job in jobs:
        expiry = _parse(job.lease_expires_at)
        if job.status in active and expiry and expiry > now:
            # A second worker generation can briefly overlap the first while
            # sharing the profile's session-scoped executor id. Never return
            # a leased/uploading job reentrantly: that would dispatch another
            # tab. Only submit-intent recovery is safe to resume in place.
            if job.lease_owner == owner_instance_id and job.status == "submitting":
                return job
            return None
        if job.status in active and (expiry is None or expiry <= now):
            job.status = "queued"
            job.phase = "lease_expired"
            job.lease_owner = None
            job.lease_expires_at = None
            _event(job, "lease_expired")
            _write_job(root, job)

    for job in jobs:
        if job.status != "submitted":
            continue
        expiry = _parse(job.lease_expires_at)
        if expiry is not None and expiry > now:
            continue
        job.lease_owner = owner_instance_id
        job.executor_instance_id = owner_instance_id
        job.lease_expires_at = _iso(now + timedelta(hours=6))
        job.updated_at = _iso(now)
        _event(job, "monitor_adopted", owner=owner_instance_id)
        _write_job(root, job)
        return job

    global_cooldown_until = max(
        (
            parsed
            for job in jobs
            if job.cooldown_reason in {"rate_limited", "safety_locked"}
            if (parsed := _parse(job.next_attempt_at)) is not None
        ),
        default=None,
    )
    if global_cooldown_until is not None and global_cooldown_until > now:
        return None

    latest_submitted = _latest_submitted_at(jobs)
    candidates = sorted(
        jobs,
        key=lambda job: (not job.manual_priority, job.queue_position, job.created_at, job.job_id),
    )
    for job in candidates:
        if job.status not in {"queued", "cooldown"}:
            continue
        due_inputs = [_parse(job.not_before)]
        if not job.manual_priority:
            due_inputs.append(_parse(job.next_attempt_at))
        due = max(filter(None, due_inputs), default=now)
        if latest_submitted is not None and not job.manual_priority:
            due = max(due, latest_submitted + timedelta(minutes=job.cadence_minutes))
        if due > now:
            if job.status != "cooldown" or job.next_attempt_at != _iso(due):
                job.status = "cooldown"
                job.phase = "cadence_wait"
                job.cooldown_reason = "cadence"
                job.next_attempt_at = _iso(due)
                job.updated_at = _iso(now)
                _write_job(root, job)
            continue
        job.status = "leased"
        job.phase = "leased"
        job.lease_owner = owner_instance_id
        job.executor_instance_id = owner_instance_id
        job.lease_expires_at = _iso(now + timedelta(seconds=max(30, lease_seconds)))
        job.cooldown_reason = None
        job.manual_priority = False
        job.updated_at = _iso(now)
        _event(job, "leased", owner=owner_instance_id)
        _write_job(root, job)
        return job
    return None


@_serialized
def update_launch_job(
    job_id: str,
    update: BrowserLaunchJobUpdateRequest,
    *,
    spool_path: Path | None = None,
) -> BrowserLaunchJob | None:
    root = launch_job_root(spool_path)
    job = _read_job(_job_path(root, job_id))
    if job is None:
        return None
    if job.lease_owner != update.owner_instance_id:
        raise BrowserLaunchLeaseError("launch job lease owner mismatch")
    now = _now()
    was_submitted = job.status == "submitted"
    job.phase = update.phase
    job.updated_at = _iso(now)
    job.tab_id = update.tab_id if update.tab_id is not None else job.tab_id
    job.conversation_id = update.conversation_id or job.conversation_id
    job.conversation_url = update.conversation_url or job.conversation_url
    job.handoff_attachment_id = update.handoff_attachment_id or job.handoff_attachment_id
    job.retry_after_seconds = update.retry_after_seconds
    _event(job, update.outcome, detail=update.detail, owner=update.owner_instance_id)

    if update.outcome == "progress":
        if job.status == "submitted":
            job.lease_expires_at = _iso(now + timedelta(hours=6))
            job.cooldown_reason = None
            job.last_error = None
        elif update.phase == "submit_intent":
            job.status = "submitting"
            job.lease_expires_at = _iso(now + timedelta(hours=6))
        else:
            job.status = "uploading" if update.phase == "uploading" else "submitting"
            job.lease_expires_at = _iso(now + timedelta(seconds=LAUNCH_LEASE_SECONDS))
    elif update.outcome == "submitted":
        job.status = "submitted"
        job.attempts += 1
        job.last_error = None
        job.lease_expires_at = _iso(now + timedelta(hours=6))
    elif update.outcome == "submission_unknown":
        job.status = "submission_unknown"
        job.cooldown_reason = "submission_unknown"
        job.last_error = update.detail or "submission outcome is unknown; operator inspection required"
        job.lease_owner = None
        job.lease_expires_at = None
    elif update.outcome in {"auth_challenge", "protocol_mismatch"}:
        job.status = "paused"
        job.cooldown_reason = update.outcome
        job.last_error = update.detail or update.outcome
        job.lease_owner = None
        job.lease_expires_at = None
    elif update.outcome in {"rate_limited", "safety_locked", "network_error"}:
        job.attempts += 1
        if update.outcome == "safety_locked":
            default_delay = _backoff_with_jitter(job, 3600, cap_seconds=8 * 3600)
        elif update.outcome == "rate_limited":
            default_delay = _backoff_with_jitter(job, 900, cap_seconds=4 * 3600)
        else:
            default_delay = _backoff_with_jitter(job, 60, cap_seconds=1800)
        delay = update.retry_after_seconds or default_delay
        job.cooldown_reason = update.outcome
        job.last_error = update.detail or update.outcome
        job.next_attempt_at = _iso(now + timedelta(seconds=delay))
        if was_submitted and update.outcome in {"rate_limited", "safety_locked"}:
            # The provider may already have accepted the conversation. Never
            # turn a post-submit safety/rate response into an automatic new
            # conversation; pause this job while its circuit protects all
            # subsequent submissions.
            job.status = "paused"
            job.lease_owner = None
            job.lease_expires_at = None
        elif was_submitted:
            job.status = "submitted"
            job.lease_expires_at = _iso(now + timedelta(hours=6))
        else:
            job.status = "cooldown"
            job.lease_owner = None
            job.lease_expires_at = None
    else:
        job.status = "failed"
        job.last_error = update.detail or update.outcome
        job.lease_owner = None
        job.lease_expires_at = None
    _write_job(root, job)
    return job


@_serialized
def control_launch_job(
    job_id: str,
    request: BrowserLaunchJobControlRequest,
    *,
    spool_path: Path | None = None,
) -> BrowserLaunchJob | None:
    root = launch_job_root(spool_path)
    job = _read_job(_job_path(root, job_id))
    if job is None:
        return None
    now = _now()
    if request.action == "pause":
        if job.status in {"completed", "cancelled"}:
            raise BrowserLaunchConflictError(f"cannot pause launch job in {job.status}")
        if job.status in {"leased", "uploading", "submitting", "submitted"}:
            raise BrowserLaunchConflictError("cannot pause an active launch; cancel it or let it finish")
        job.status = "paused"
        job.phase = "operator_paused"
        job.manual_priority = False
    elif request.action == "cancel":
        if job.status == "completed":
            raise BrowserLaunchConflictError("cannot cancel completed launch job")
        job.status = "cancelled"
        job.phase = "cancelled"
        job.manual_priority = False
    elif request.action == "launch_now":
        if job.status in {"completed", "cancelled", "leased", "uploading", "submitting", "submitted"}:
            raise BrowserLaunchConflictError(f"cannot launch now from {job.status}")
        if job.cooldown_reason in {"rate_limited", "safety_locked"}:
            raise BrowserLaunchConflictError("cannot bypass a provider rate or safety circuit")
        job.status = "queued"
        job.phase = "operator_launch_now"
        job.next_attempt_at = _iso(now)
        job.last_error = None
        job.cooldown_reason = None
        job.manual_priority = True
    elif request.action == "confirm_no_conversation":
        if job.status != "submission_unknown":
            raise BrowserLaunchConflictError(f"cannot confirm absent conversation for launch job in {job.status}")
        job.status = "queued"
        job.phase = "operator_confirmed_no_conversation"
        job.next_attempt_at = _iso(now)
        job.last_error = None
        job.cooldown_reason = None
        job.manual_priority = True
    else:
        if job.status not in {"paused", "failed", "cooldown"}:
            raise BrowserLaunchConflictError(f"cannot {request.action} launch job in {job.status}")
        protected_until = _parse(job.next_attempt_at)
        if (
            job.cooldown_reason in {"rate_limited", "safety_locked"}
            and protected_until is not None
            and protected_until > now
        ):
            raise BrowserLaunchConflictError("cannot bypass a provider rate or safety circuit")
        job.status = "queued"
        job.phase = "queued"
        job.next_attempt_at = _iso(now)
        job.last_error = None
        job.cooldown_reason = None
        job.manual_priority = False
    job.lease_owner = None
    job.lease_expires_at = None
    job.updated_at = _iso(now)
    _event(job, f"operator_{request.action}", detail=request.inspection_receipt)
    _write_job(root, job)
    return job


def read_launch_attachment(
    job_id: str, attachment_id: str, *, spool_path: Path | None = None
) -> tuple[BrowserLaunchAttachment, bytes] | None:
    root = launch_job_root(spool_path)
    job = _read_job(_job_path(root, job_id))
    if job is None:
        return None
    attachment = next((item for item in job.attachments if item.attachment_id == attachment_id), None)
    if attachment is None:
        return None
    path = _job_dir(root, job_id) / "attachments" / _safe_token(attachment_id)
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    if len(content) != attachment.size_bytes or hashlib.sha256(content).hexdigest() != attachment.sha256:
        raise BrowserLaunchConflictError("launch attachment integrity mismatch")
    return attachment, content


def _validate_handoff_zip(content: bytes, *, expected_prompt_profile: str) -> int:
    try:
        archive = zipfile.ZipFile(BytesIO(content))
    except (OSError, zipfile.BadZipFile) as exc:
        raise BrowserLaunchHandoffError("handoff is not a readable ZIP") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > LAUNCH_HANDOFF_MAX_FILES:
            raise BrowserLaunchHandoffError("handoff ZIP contains too many entries")
        if sum(info.file_size for info in infos) > LAUNCH_HANDOFF_MAX_UNCOMPRESSED_BYTES:
            raise BrowserLaunchHandoffError("handoff ZIP uncompressed size exceeds quota")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise BrowserLaunchHandoffError("handoff ZIP contains duplicate paths")
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise BrowserLaunchHandoffError("handoff ZIP contains an unsafe path")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise BrowserLaunchHandoffError("handoff ZIP contains a symlink")
        files = {info.filename: info for info in infos if not info.is_dir()}
        required_files = {"MANIFEST.json", "README.md", "SUMMARY.md", "VERIFICATION-LIMITS.md"}
        if not required_files.issubset(files):
            raise BrowserLaunchHandoffError("handoff ZIP is missing required root files")
        for required_dir in ("PATCHES/", "DESIGN/", "TESTS/"):
            if not any(name.startswith(required_dir) and name != required_dir for name in files):
                raise BrowserLaunchHandoffError(f"handoff ZIP has no deliverable under {required_dir}")
        try:
            manifest = json.loads(archive.read("MANIFEST.json"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BrowserLaunchHandoffError("handoff manifest is not valid JSON") from exc
        records = manifest.get("files") if isinstance(manifest, dict) else None
        prompt_profile = manifest.get("prompt_profile") if isinstance(manifest, dict) else None
        if prompt_profile != expected_prompt_profile:
            raise BrowserLaunchHandoffError("handoff manifest prompt profile mismatch")
        if not isinstance(records, list):
            raise BrowserLaunchHandoffError("handoff manifest must contain a files list")
        by_path = {
            str(record.get("path")): record
            for record in records
            if isinstance(record, dict) and isinstance(record.get("path"), str)
        }
        for name, info in files.items():
            if name == "MANIFEST.json":
                continue
            record = by_path.get(name)
            if record is None:
                raise BrowserLaunchHandoffError(f"handoff manifest omits {name}")
            if "purpose" not in record or "apply_order" not in record:
                raise BrowserLaunchHandoffError(f"handoff manifest lacks purpose/apply order for {name}")
            expected_size = record.get("size_bytes", record.get("byte_size"))
            if expected_size != info.file_size:
                raise BrowserLaunchHandoffError(f"handoff manifest size mismatch for {name}")
            expected_hash = record.get("sha256")
            if expected_hash != hashlib.sha256(archive.read(name)).hexdigest():
                raise BrowserLaunchHandoffError(f"handoff manifest checksum mismatch for {name}")
        return len(files)


@_serialized
def accept_launch_handoff(
    job_id: str,
    request: BrowserLaunchHandoffRequest,
    *,
    spool_path: Path | None = None,
) -> BrowserLaunchJob | None:
    root = launch_job_root(spool_path)
    job = _read_job(_job_path(root, job_id))
    if job is None:
        return None
    if job.lease_owner != request.owner_instance_id:
        raise BrowserLaunchLeaseError("launch job lease owner mismatch")
    if job.status != "submitted":
        raise BrowserLaunchConflictError(f"cannot accept handoff for launch job in {job.status}")
    try:
        content = base64.b64decode(request.content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BrowserLaunchHandoffError("handoff content is not valid base64") from exc
    if len(content) > LAUNCH_HANDOFF_MAX_BYTES:
        raise BrowserLaunchQuotaError("launch handoff byte quota exceeded")
    file_count = _validate_handoff_zip(content, expected_prompt_profile=job.prompt_profile)
    path = _job_dir(root, job_id) / "handoff" / LAUNCH_HANDOFF_NAME
    _atomic_write(path, content)
    now = _now()
    job.status = "completed"
    job.phase = "handoff_validated"
    job.updated_at = _iso(now)
    job.handoff_artifact_ref = path.relative_to(root.parent).as_posix()
    job.handoff_sha256 = hashlib.sha256(content).hexdigest()
    job.handoff_size_bytes = len(content)
    job.handoff_file_count = file_count
    job.handoff_validated_at = _iso(now)
    job.lease_owner = None
    job.lease_expires_at = None
    job.last_error = None
    _event(job, "completed", detail=f"validated {file_count} handoff files", owner=request.owner_instance_id)
    _write_job(root, job)
    return job


__all__ = [
    "BrowserLaunchConflictError",
    "BrowserLaunchHandoffError",
    "BrowserLaunchLeaseError",
    "BrowserLaunchQuotaError",
    "BrowserLaunchStateError",
    "accept_launch_handoff",
    "claim_due_launch_job",
    "control_launch_job",
    "enqueue_launch_job",
    "launch_job_root",
    "list_launch_jobs",
    "read_launch_attachment",
    "update_launch_job",
]
