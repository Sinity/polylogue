"""Pure redaction + fail-closed leak-check gate for sanitized exports (#2381).

The design has two independent layers:

* ``sanitize_rows`` / the ``_Sanitizer`` engine redact each string field value
  (absolute paths → ``<redacted-path>/{basename}``, known secrets and opaque
  high-entropy tokens → ``<redacted-secret>``) and record every decision in a
  :class:`~polylogue.schemas.redaction_report.SchemaReport`.
* ``verify_sanitized_export`` independently re-scans the *written* bundle files
  for any surviving absolute path, ``$HOME``-relative path, or known-secret
  pattern. This is the authoritative fail-closed gate: ``write_sanitized_bundle``
  runs it before publishing and refuses (cleaning up the temp dir, raising
  :class:`SanitizedExportError`) if anything leaks — even if per-field redaction
  missed it.

The manifest never stores an original secret/path value: rejected decisions
carry the redacted form only, so the manifest itself cannot leak.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from polylogue.core.json import dumps
from polylogue.errors import PolylogueError
from polylogue.insights.archive_models import ArchiveInsightModel
from polylogue.schemas.privacy import _looks_high_entropy_token
from polylogue.schemas.privacy_config import PrivacyConfig, PrivacyLevel
from polylogue.schemas.redaction_report import FieldReport, RedactionDecision, SchemaReport
from polylogue.version import VERSION_INFO

SANITIZED_EXPORT_BUNDLE_VERSION = 1

REDACTED_PATH_PREFIX = "<redacted-path>"
REDACTED_SECRET = "<redacted-secret>"
REDACTED_EMAIL = "<redacted-email>"

DATASET_FILENAME = "dataset.jsonl"
MANIFEST_FILENAME = "redaction-manifest.json"
README_FILENAME = "README.md"
POSTMORTEM_FILENAME = "postmortem.json"

# Absolute filesystem path: a leading slash followed by at least two path
# segments. The two-segment minimum avoids flagging a lone ``/word`` token in
# prose while still catching every real ``/a/b/...`` path. Segment characters
# include spaces so a path like ``/home/me/Secret Project/notes.md`` is matched
# as a whole — without the space, the directory and filename after the space
# would leak past both the redactor and the (same-pattern) gate. Over-matching
# prose that interleaves several slash-runs is the *safe* direction for a
# sanitizer: it redacts more, never less.
_ABS_PATH_RE = re.compile(r"/(?:[A-Za-z0-9._\- ]+/)+[A-Za-z0-9._\- ]*[A-Za-z0-9._\-]")
# ``$HOME``-relative path written with a tilde (``~/a/b`` or bare ``~/x``).
_TILDE_PATH_RE = re.compile(r"~/(?:[A-Za-z0-9._\- ]+/)*[A-Za-z0-9._\- ]*[A-Za-z0-9._\-]")
# Single-segment absolute path to a well-known system/home root (``/etc``,
# ``/tmp``, ``/home`` …). ``_ABS_PATH_RE`` requires two segments to avoid
# flagging ``/word`` prose, so these roots are caught explicitly by both the
# redactor and the gate. The negative lookahead excludes ``/`` so multi-segment
# paths stay owned by ``_ABS_PATH_RE`` and ``/etcetera`` is not matched.
_KNOWN_ROOT_PATH_RE = re.compile(
    r"/(?:home|Users|root|tmp|var|etc|opt|mnt|media|srv|usr|private|data|realm|persist)(?![A-Za-z0-9._\-/])"
)
# Windows-style filesystem paths: a drive-letter root (``C:\Users\me\...``) or a
# UNC share (``\\server\share\...``). Backslash-separated, so there is no overlap
# with the POSIX ``/``-based path regexes above. Matching runs of non-space,
# non-quote characters keeps a whole path together; over-matching is the safe
# direction for a sanitizer.
_WINDOWS_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s\"'<>|]*")
# Email addresses (PII). Conservative shape: local@domain.tld.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Candidate opaque token for high-entropy detection.
_HIGH_ENTROPY_CANDIDATE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{9,}")

# Known secret value shapes. Conservative, prefix-anchored patterns so opaque
# but benign tokens (UUID session ids, content hashes) are not flagged by the
# authoritative gate. Broader opaque-token redaction happens via
# ``_looks_high_entropy_token`` on the redaction side only.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"gh[posru]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


class SanitizedExportError(PolylogueError):
    """Raised when a sanitized export cannot be produced or fails the gate."""


class SanitizedExportRequest(ArchiveInsightModel):
    """Typed request for a sanitized shareable export bundle."""

    output_path: Path
    privacy_level: PrivacyLevel = "standard"
    with_postmortem: bool = False
    overwrite: bool = False
    redact: bool = True
    acknowledge_unredacted: bool = False


class SanitizedExportResult(ArchiveInsightModel):
    """Result of writing a sanitized export bundle."""

    output_path: Path
    dataset_path: Path
    manifest_path: Path
    readme_path: Path
    postmortem_path: Path | None = None
    row_count: int = 0
    redacted: bool = True
    total_included: int = 0
    total_rejected: int = 0
    verify_ok: bool = True


class SanitizedExportVerifyResult(ArchiveInsightModel):
    """Outcome of the independent leak-check gate over a written bundle."""

    bundle_dir: Path
    ok: bool
    files_scanned: tuple[str, ...] = ()
    absolute_path_leaks: tuple[str, ...] = ()
    home_path_leaks: tuple[str, ...] = ()
    secret_leaks: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Redaction engine
# ---------------------------------------------------------------------------


def _redacted_path(segment: str) -> str:
    base = segment.rstrip("/").rsplit("/", 1)[-1] or "path"
    return f"{REDACTED_PATH_PREFIX}/{base}"


def _redact_string(value: str) -> tuple[str, list[tuple[str, str]]]:
    """Return ``(redacted, decisions)`` for one string value.

    ``decisions`` is a list of ``(reason, risk)`` pairs, one per replacement.
    """

    reasons: list[tuple[str, str]] = []

    def _path_repl(match: re.Match[str]) -> str:
        reasons.append(("absolute_path", "high"))
        return _redacted_path(match.group(0))

    def _home_repl(match: re.Match[str]) -> str:
        reasons.append(("home_relative_path", "high"))
        return _redacted_path(match.group(0))

    def _known_root_repl(match: re.Match[str]) -> str:
        # The root name itself is the sensitive token, so drop it entirely (no
        # retained basename) — leaving it would re-trip the gate.
        reasons.append(("absolute_path", "high"))
        return REDACTED_PATH_PREFIX

    def _winpath_repl(match: re.Match[str]) -> str:
        reasons.append(("windows_path", "high"))
        return REDACTED_PATH_PREFIX

    def _secret_repl(match: re.Match[str]) -> str:
        reasons.append(("secret_pattern", "high"))
        return REDACTED_SECRET

    def _email_repl(match: re.Match[str]) -> str:
        reasons.append(("email", "high"))
        return REDACTED_EMAIL

    def _entropy_repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if _looks_high_entropy_token(token):
            reasons.append(("high_entropy_token", "high"))
            return REDACTED_SECRET
        return token

    result = _ABS_PATH_RE.sub(_path_repl, value)
    result = _TILDE_PATH_RE.sub(_home_repl, result)
    result = _KNOWN_ROOT_PATH_RE.sub(_known_root_repl, result)
    result = _WINDOWS_PATH_RE.sub(_winpath_repl, result)
    # Emails before secret/entropy passes so the address is replaced as one unit
    # rather than partly mangled by the opaque-token detector.
    result = _EMAIL_RE.sub(_email_repl, result)
    for pattern in _SECRET_VALUE_PATTERNS:
        result = pattern.sub(_secret_repl, result)
    result = _HIGH_ENTROPY_CANDIDATE_RE.sub(_entropy_repl, result)
    return result, reasons


class _Sanitizer:
    """Stateful redaction engine that records decisions into a SchemaReport."""

    def __init__(self, *, config: PrivacyConfig) -> None:
        self.config = config
        self.report = SchemaReport(provider="sanitized_export", privacy_level=config.level)
        self._field_reports: dict[str, FieldReport] = {}

    def value(self, value: object, path: str) -> object:
        if isinstance(value, str):
            return self._string(value, path)
        if isinstance(value, Mapping):
            return {str(key): self.value(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(item, f"{path}[]") for item in value]
        return value

    def mapping(self, row: Mapping[str, object]) -> dict[str, object]:
        return {key: self.value(item, key) for key, item in row.items()}

    def _string(self, value: str, path: str) -> str:
        redacted, decisions = _redact_string(value)
        if not decisions:
            self.report.add_decision(RedactionDecision(path=path, value="", action="included"))
            return redacted
        field_report = self._field_reports.setdefault(path, FieldReport(path=path))
        for reason, risk in decisions:
            decision = RedactionDecision(
                # Store the redacted form only — never the original value.
                path=path,
                value=redacted[:160],
                action="rejected",
                reason=reason,
                risk=risk,  # type: ignore[arg-type]
            )
            self.report.add_decision(decision)
            field_report.rejected.append(decision)
        return redacted

    def finalize(self) -> SchemaReport:
        self.report.field_reports = list(self._field_reports.values())
        self.report.total_fields = len(self._field_reports)
        return self.report


def sanitize_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    config: PrivacyConfig,
) -> tuple[list[dict[str, object]], SchemaReport]:
    """Redact every string field in ``rows`` and return the redaction report.

    Pure: no I/O. Each absolute path becomes ``<redacted-path>/{basename}`` and
    each known secret / opaque high-entropy token becomes ``<redacted-secret>``.
    Every decision is recorded in the returned :class:`SchemaReport`.
    """

    sanitizer = _Sanitizer(config=config)
    sanitized = [sanitizer.mapping(row) for row in rows]
    sanitizer.finalize()
    return sanitized, sanitizer.report


# ---------------------------------------------------------------------------
# Leak-check gate
# ---------------------------------------------------------------------------


def _home_dir(home: str | None) -> str:
    return home if home is not None else os.path.expanduser("~")


def _scan_text(text: str, *, home: str) -> tuple[list[str], list[str], list[str]]:
    abs_leaks = sorted(
        {match.group(0) for match in _ABS_PATH_RE.finditer(text)}
        | {match.group(0) for match in _KNOWN_ROOT_PATH_RE.finditer(text)}
        | {match.group(0) for match in _WINDOWS_PATH_RE.finditer(text)}
    )
    home_leaks: list[str] = [match.group(0) for match in _TILDE_PATH_RE.finditer(text)]
    normalized_home = home.rstrip("/")
    if normalized_home and len(normalized_home) > 1 and normalized_home in text:
        home_leaks.append(normalized_home)
    secret_leaks: list[str] = []
    for pattern in _SECRET_VALUE_PATTERNS:
        secret_leaks.extend(match.group(0) for match in pattern.finditer(text))
    secret_leaks.extend(match.group(0) for match in _EMAIL_RE.finditer(text))
    return abs_leaks, sorted(set(home_leaks)), sorted(set(secret_leaks))


def verify_sanitized_export(
    bundle_dir: Path,
    *,
    home: str | None = None,
) -> SanitizedExportVerifyResult:
    """Independently scan a written bundle for surviving private data.

    Reads ``dataset.jsonl``, ``redaction-manifest.json``, ``README.md`` and an
    optional ``postmortem.json`` as raw text and flags any absolute path,
    ``$HOME``-relative path, or known-secret pattern. ``ok`` is True only when
    no leak of any class is found.
    """

    resolved_home = _home_dir(home)
    files = (DATASET_FILENAME, MANIFEST_FILENAME, README_FILENAME, POSTMORTEM_FILENAME)
    scanned: list[str] = []
    abs_leaks: list[str] = []
    home_leaks: list[str] = []
    secret_leaks: list[str] = []
    for name in files:
        candidate = bundle_dir / name
        if not candidate.exists():
            continue
        scanned.append(name)
        text = candidate.read_text(encoding="utf-8")
        file_abs, file_home, file_secret = _scan_text(text, home=resolved_home)
        abs_leaks.extend(f"{name}: {leak}" for leak in file_abs)
        home_leaks.extend(f"{name}: {leak}" for leak in file_home)
        secret_leaks.extend(f"{name}: {leak}" for leak in file_secret)
    ok = not (abs_leaks or home_leaks or secret_leaks)
    return SanitizedExportVerifyResult(
        bundle_dir=bundle_dir,
        ok=ok,
        files_scanned=tuple(scanned),
        absolute_path_leaks=tuple(abs_leaks),
        home_path_leaks=tuple(home_leaks),
        secret_leaks=tuple(secret_leaks),
    )


# ---------------------------------------------------------------------------
# Bundle writer
# ---------------------------------------------------------------------------


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _dataset_text(rows: Sequence[Mapping[str, object]]) -> str:
    return "".join(f"{dumps(dict(row))}\n" for row in rows)


def _readme_text(*, redacted: bool, row_count: int) -> str:
    posture = "Redacted (fail-closed)" if redacted else "UNSANITIZED — explicitly acknowledged"
    lines = [
        "# Sanitized Polylogue Export",
        "",
        f"- Bundle version: {SANITIZED_EXPORT_BUNDLE_VERSION}",
        f"- Privacy posture: {posture}",
        f"- Dataset rows: {row_count}",
        "",
        "## Contents",
        "",
        f"- `{DATASET_FILENAME}` — one session-level metadata row per line.",
        f"- `{MANIFEST_FILENAME}` — every redaction decision (what was removed, by which rule).",
        f"- `{POSTMORTEM_FILENAME}` — sanitized postmortem report (only when requested).",
        "",
        "## What is included",
        "",
        "Session-level metadata only: id, display title, origin, first/last",
        "message timestamps, message and word counts, repo names, cost totals",
        "with labelled token lanes, and wall-clock duration.",
        "",
        "## What is deliberately excluded (v0)",
        "",
        "Raw message text is NOT exported in this version — message bodies are",
        "the highest-leak surface and are deferred to a follow-up. Absolute",
        "filesystem paths (repo paths, working directories, touched files) and",
        "known secrets are scrubbed; the bundle is refused if any survive.",
        "",
    ]
    if not redacted:
        lines.extend(
            [
                "## WARNING",
                "",
                "This bundle was produced with redaction DISABLED via an explicit",
                "acknowledgement. It may contain private absolute paths and secrets.",
                "Do not share it.",
                "",
            ]
        )
    return "\n".join(lines)


def _manifest_document(
    *,
    report: SchemaReport,
    scope: Mapping[str, object],
    request: SanitizedExportRequest,
    row_count: int,
) -> dict[str, object]:
    return {
        "bundle_version": SANITIZED_EXPORT_BUNDLE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "polylogue_version": VERSION_INFO.full,
        "git_revision": VERSION_INFO.commit,
        "git_dirty": VERSION_INFO.dirty,
        "privacy_level": request.privacy_level,
        "redacted": request.redact,
        "row_count": row_count,
        "dataset_file": DATASET_FILENAME,
        "scope": dict(scope),
        "redaction": report.to_json(),
    }


def write_sanitized_bundle(
    *,
    rows: Sequence[Mapping[str, object]],
    report: SchemaReport,
    scope: Mapping[str, object],
    request: SanitizedExportRequest,
    postmortem: Mapping[str, object] | None = None,
    home: str | None = None,
    run_gate: bool = True,
) -> SanitizedExportResult:
    """Atomically write a bundle to a temp dir, gate it, then publish.

    The leak-check gate runs against the *written* files when ``run_gate`` is
    True. If it finds any leak, the temp dir is removed and
    :class:`SanitizedExportError` is raised — nothing is published.
    """

    target = request.output_path
    if target.exists() and not request.overwrite:
        raise SanitizedExportError(f"Export target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    tmp_target.mkdir(parents=False)

    postmortem_path: Path | None = None
    try:
        _write_text(tmp_target / DATASET_FILENAME, _dataset_text(rows))
        manifest = _manifest_document(report=report, scope=scope, request=request, row_count=len(rows))
        _write_text(tmp_target / MANIFEST_FILENAME, dumps(manifest))
        _write_text(tmp_target / README_FILENAME, _readme_text(redacted=request.redact, row_count=len(rows)))
        if postmortem is not None:
            postmortem_path = target / POSTMORTEM_FILENAME
            _write_text(tmp_target / POSTMORTEM_FILENAME, dumps(dict(postmortem)))

        verify_ok = True
        if run_gate:
            verdict = verify_sanitized_export(tmp_target, home=home)
            verify_ok = verdict.ok
            if not verdict.ok:
                raise SanitizedExportError(
                    "sanitized export refused: leak-check gate found surviving private data "
                    f"(absolute_paths={list(verdict.absolute_path_leaks)}, "
                    f"home_paths={list(verdict.home_path_leaks)}, "
                    f"secrets={list(verdict.secret_leaks)})"
                )

        # Atomic publish: move any existing bundle aside first, swap the new one
        # in, then drop the backup. If the swap fails the previous bundle is
        # restored, so an overwrite never loses the last good bundle.
        backup: Path | None = None
        if target.exists():
            backup = target.parent / f".{target.name}.bak-{uuid.uuid4().hex}"
            target.replace(backup)
        try:
            tmp_target.replace(target)
        except BaseException:
            if backup is not None and not target.exists():
                backup.replace(target)
            raise
        if backup is not None:
            if backup.is_dir():
                shutil.rmtree(backup, ignore_errors=True)
            else:
                backup.unlink(missing_ok=True)
    except BaseException:
        shutil.rmtree(tmp_target, ignore_errors=True)
        raise

    return SanitizedExportResult(
        output_path=target,
        dataset_path=target / DATASET_FILENAME,
        manifest_path=target / MANIFEST_FILENAME,
        readme_path=target / README_FILENAME,
        postmortem_path=postmortem_path,
        row_count=len(rows),
        redacted=request.redact,
        total_included=report.total_included,
        total_rejected=report.total_rejected,
        verify_ok=verify_ok,
    )


def produce_sanitized_export(
    *,
    rows: Sequence[Mapping[str, object]],
    scope: Mapping[str, object],
    request: SanitizedExportRequest,
    postmortem: Mapping[str, object] | None = None,
    home: str | None = None,
) -> SanitizedExportResult:
    """Sanitize ``rows`` (+ scope/postmortem) and write the gated bundle.

    Fail-closed: when ``request.redact`` is False the writer refuses unless
    ``request.acknowledge_unredacted`` is also set. The unredacted path skips
    redaction and the gate by design (loud warning in the README/manifest); the
    redacted path always runs the gate.
    """

    if not request.redact:
        if not request.acknowledge_unredacted:
            raise SanitizedExportError(
                "refusing to write an unredacted export: redaction is disabled but the "
                "explicit acknowledge_unredacted confirmation was not given"
            )
        empty_report = SchemaReport(provider="sanitized_export", privacy_level=request.privacy_level)
        return write_sanitized_bundle(
            rows=[dict(row) for row in rows],
            report=empty_report,
            scope=dict(scope),
            request=request,
            postmortem=None if postmortem is None else dict(postmortem),
            home=home,
            run_gate=False,
        )

    config = PrivacyConfig(level=request.privacy_level)
    sanitizer = _Sanitizer(config=config)
    sanitized_rows = [sanitizer.mapping(row) for row in rows]
    sanitized_scope_value = sanitizer.value(dict(scope), "scope")
    sanitized_scope = sanitized_scope_value if isinstance(sanitized_scope_value, dict) else {}
    sanitized_postmortem: dict[str, object] | None = None
    if postmortem is not None:
        sanitized_pm_value = sanitizer.value(dict(postmortem), "postmortem")
        if isinstance(sanitized_pm_value, dict):
            sanitized_postmortem = sanitized_pm_value
    report = sanitizer.finalize()
    return write_sanitized_bundle(
        rows=sanitized_rows,
        report=report,
        scope=sanitized_scope,
        request=request,
        postmortem=sanitized_postmortem,
        home=home,
        run_gate=True,
    )


def assert_text_sanitized(text: str, *, home: str | None = None) -> None:
    """Fail-closed leak gate for rendered text (#2437).

    Re-scans an already-rendered report string with the same detector
    :func:`verify_sanitized_export` uses (absolute/known-root/Windows paths,
    ``$HOME``-relative paths, known secrets, emails). Raises
    :class:`SanitizedExportError` if any class of leak survives. Callers MUST
    emit nothing when this raises — that is the fail-closed contract.
    """

    abs_leaks, home_leaks, secret_leaks = _scan_text(text, home=_home_dir(home))
    if abs_leaks or home_leaks or secret_leaks:
        # Report counts by class only — never the leaked values themselves. The
        # CLI echoes this message, so embedding the surviving paths/secrets would
        # re-leak the exact private data the gate exists to suppress.
        raise SanitizedExportError(
            "rendered report failed the sanitizer leak gate: "
            f"absolute_paths={len(abs_leaks)} home_paths={len(home_leaks)} secrets={len(secret_leaks)} "
            "(values withheld)"
        )


__all__ = [
    "DATASET_FILENAME",
    "MANIFEST_FILENAME",
    "POSTMORTEM_FILENAME",
    "README_FILENAME",
    "REDACTED_PATH_PREFIX",
    "REDACTED_SECRET",
    "SANITIZED_EXPORT_BUNDLE_VERSION",
    "SanitizedExportError",
    "SanitizedExportRequest",
    "SanitizedExportResult",
    "SanitizedExportVerifyResult",
    "assert_text_sanitized",
    "produce_sanitized_export",
    "sanitize_rows",
    "verify_sanitized_export",
    "write_sanitized_bundle",
]
