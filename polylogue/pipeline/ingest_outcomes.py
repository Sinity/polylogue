"""Typed ingest-attempt disposition classification (polylogue-cnu3).

``ingest_attempts.error_message`` used to be the only durable record of why
an ingest attempt failed: free-form text that could not be counted, grouped,
or consumed by a retry policy without re-grepping logs and source. This
module is the single place that maps a real production failure -- an
exception object, a structural validation result, or an explicit "this
artifact isn't admitted" decision -- to a closed :class:`IngestOutcome` plus
retryability, evidence, and remediation, *without ever text-matching against
an error string*. Every classification below keys off the caught exception's
*type* or a structural boolean the pipeline already computed.

Two call sites use this:

- ``polylogue.pipeline.services.ingest_worker`` classifies each raw record at
  the acquire/detect/parse/materialize boundary (subprocess-safe, no DB
  access) -- see ``_record_result`` call sites there.
- ``polylogue.sources.live.batch`` classifies each ingest-attempt's terminal
  outcome at the archive-write boundary (schema mismatch, database error,
  transient SQLite lock, or clean completion) before it lands in the
  ``ingest_attempts`` ops-tier row via ``CursorStore``.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from polylogue.core.enums import INGEST_OUTCOME_RETRYABLE, IngestOutcome

#: Bounded diagnostic length -- long enough to keep the failing detail
#: legible, short enough that a pathological payload can never make the
#: disposable ops.db tier grow without limit from one attempt's diagnostic.
_MAX_DIAGNOSTIC_LEN = 2000

#: Coarse secret-shaped substring patterns redacted before a diagnostic is
#: persisted. This is deliberately conservative (false positives redact
#: harmless text; false negatives are the real risk) -- it is a bounded
#: best-effort net for the ops-tier disposable diagnostic column, not the
#: full archive-wide secret scan (``polylogue.security.secret_scan``).
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(sk|pk|api|secret|token|bearer)[-_ ]?[a-z0-9]{16,}\b"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._-]{8,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),  # long base64-shaped runs
)
_REDACTED = "[redacted]"


def bounded_diagnostic(text: str | None, *, max_len: int = _MAX_DIAGNOSTIC_LEN) -> str | None:
    """Return a length-bounded, secret-redacted copy of ``text`` for storage.

    ``None`` in, ``None`` out -- callers should not fabricate a diagnostic
    when there is nothing to say (e.g. a clean success).
    """
    if text is None:
        return None
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    if len(redacted) > max_len:
        redacted = redacted[:max_len] + "...[truncated]"
    return redacted


@dataclass(frozen=True, slots=True)
class IngestAttemptDisposition:
    """One typed, queryable classification of an ingest attempt's outcome."""

    outcome: IngestOutcome
    evidence_ref: str | None = None
    diagnostic: str | None = None
    remediation: str | None = None

    @property
    def retryable(self) -> bool | None:
        return INGEST_OUTCOME_RETRYABLE[self.outcome]

    @property
    def outcome_code(self) -> str:
        return self.outcome.value


def success_disposition(evidence_ref: str | None = None) -> IngestAttemptDisposition:
    """The disposition for a clean completion (including a no-op re-ingest)."""
    return IngestAttemptDisposition(outcome=IngestOutcome.SUCCESS, evidence_ref=evidence_ref)


def corrupt_input_disposition(*, evidence_ref: str, diagnostic: str | None) -> IngestAttemptDisposition:
    return IngestAttemptDisposition(
        outcome=IngestOutcome.CORRUPT_INPUT,
        evidence_ref=evidence_ref,
        diagnostic=bounded_diagnostic(diagnostic),
        remediation="verify the source export produced well-formed, non-empty content",
    )


def unsupported_shape_disposition(*, evidence_ref: str, diagnostic: str | None = None) -> IngestAttemptDisposition:
    return IngestAttemptDisposition(
        outcome=IngestOutcome.UNSUPPORTED_SHAPE,
        evidence_ref=evidence_ref,
        diagnostic=bounded_diagnostic(diagnostic),
        remediation="this artifact shape/kind is not admitted for session parsing; open a source-support issue",
    )


def validation_rejected_disposition(*, evidence_ref: str, diagnostic: str | None) -> IngestAttemptDisposition:
    return IngestAttemptDisposition(
        outcome=IngestOutcome.VALIDATION_REJECTED,
        evidence_ref=evidence_ref,
        diagnostic=bounded_diagnostic(diagnostic),
        remediation="fix the source input to satisfy strict schema validation, or relax validation mode",
    )


def transient_error_disposition(*, evidence_ref: str, diagnostic: str | None) -> IngestAttemptDisposition:
    return IngestAttemptDisposition(
        outcome=IngestOutcome.TRANSIENT_ERROR,
        evidence_ref=evidence_ref,
        diagnostic=bounded_diagnostic(diagnostic),
        remediation="transient infrastructure contention; the daemon will retry automatically",
    )


def parser_defect_disposition(*, evidence_ref: str, diagnostic: str | None) -> IngestAttemptDisposition:
    return IngestAttemptDisposition(
        outcome=IngestOutcome.PARSER_DEFECT,
        evidence_ref=evidence_ref,
        diagnostic=bounded_diagnostic(diagnostic),
        remediation="unexpected parser/transform exception; file a bug with the attached diagnostic",
    )


def legacy_unknown_disposition(diagnostic: str | None = None) -> IngestAttemptDisposition:
    return IngestAttemptDisposition(
        outcome=IngestOutcome.LEGACY_UNKNOWN,
        diagnostic=bounded_diagnostic(diagnostic),
        remediation=None,
    )


def classify_decode_exception(exc: BaseException) -> IngestAttemptDisposition:
    """Classify a raw-payload decode-stage failure by the exception's type.

    ``UnicodeDecodeError``/``ValueError`` (which ``json.JSONDecodeError``
    subclasses) mean the bytes themselves are not the expected shape --
    :data:`IngestOutcome.CORRUPT_INPUT`. Anything else at the decode
    boundary is an unexpected defect in the decoder itself, not corrupt
    input.
    """
    evidence_ref = f"decode:{type(exc).__name__}"
    if isinstance(exc, UnicodeDecodeError | ValueError):
        return corrupt_input_disposition(evidence_ref=evidence_ref, diagnostic=str(exc))
    return parser_defect_disposition(evidence_ref=evidence_ref, diagnostic=str(exc))


def classify_parse_exception(exc: BaseException) -> IngestAttemptDisposition:
    """Classify a per-record parse-stage failure by the exception's type.

    A ``pydantic.ValidationError`` is a structural validation rejection, not
    a parser bug -- it means the input itself failed the provider's own
    strict record validation. Everything else at this boundary is treated
    as a genuine parser defect (the honest "we didn't expect this" bucket).
    """
    from pydantic import ValidationError

    evidence_ref = f"parse:{type(exc).__name__}"
    if isinstance(exc, ValidationError):
        return validation_rejected_disposition(evidence_ref=evidence_ref, diagnostic=str(exc))
    return parser_defect_disposition(evidence_ref=evidence_ref, diagnostic=str(exc))


def classify_archive_write_exception(exc: BaseException) -> IngestAttemptDisposition:
    """Classify a batch-level archive-write failure (the daemon writer boundary).

    A ``sqlite3.OperationalError`` recognized by
    :func:`polylogue.sources.live.sqlite_locking.is_transient_sqlite_lock` is
    retryable infrastructure contention, never a poisoned payload. Any other
    exception escaping the archive-write boundary is treated as a parser
    defect (see AC2: materialization/index-failure is deliberately deferred
    to follow-up work, so it also lands here today rather than silently
    vanishing as ``LEGACY_UNKNOWN``).
    """
    from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock

    evidence_ref = f"archive_write:{type(exc).__name__}"
    if isinstance(exc, sqlite3.OperationalError) and is_transient_sqlite_lock(exc):
        return transient_error_disposition(evidence_ref=evidence_ref, diagnostic=str(exc))
    return parser_defect_disposition(evidence_ref=evidence_ref, diagnostic=str(exc))


__all__ = [
    "IngestAttemptDisposition",
    "bounded_diagnostic",
    "classify_archive_write_exception",
    "classify_decode_exception",
    "classify_parse_exception",
    "corrupt_input_disposition",
    "legacy_unknown_disposition",
    "parser_defect_disposition",
    "success_disposition",
    "transient_error_disposition",
    "unsupported_shape_disposition",
    "validation_rejected_disposition",
]
