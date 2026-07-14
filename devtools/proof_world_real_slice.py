"""Real-archive candidate-slice extraction and privacy screening.

Support for polylogue-212.11 (shared deterministic proof world / Incident
14:32): the deterministic demo corpus should eventually be extended with a
*representative slice of real archive data*, not synthetic fixtures alone.
That extension is deliberately a two-step, human-gated process:

1. This tool runs **read-only** queries against a real Polylogue archive,
   flattens each candidate session to plain text, and screens that text for
   secrets/credentials and personal-information patterns. It writes a report
   plus rendered transcripts to an arbitrary output directory. It never
   mutates the source archive (``Polylogue.get_session`` opens the archive
   tiers with ``read_only=True``) and never writes into the product fixture
   tree (``polylogue/scenarios/``) on its own. The per-session transcript
   files under ``<out>/transcripts/`` are full, unredacted flattened text —
   they exist for a human to read the real session content. The
   *report/manifest* (``SCREENING_REPORT.md``, ``manifest.json``) are a
   different, narrower surface: any matched **secret** value is redacted
   before it is written there (see ``scan_text``/``_snippet``), so a report
   that later gets shared or accidentally committed doesn't itself become a
   secret-leak vector. Matched **PII** text is kept verbatim in the report
   since a reviewer needs the real value to judge placeholder vs. genuine
   data. Point ``--out`` at a location outside version control (e.g. a
   gitignored scratch directory) — this tool applies no guard against
   writing into a tracked path.
2. An operator reviews the report and transcripts and decides, session by
   session, whether the slice is safe to fold into the shared proof-world
   corpus. Only after that explicit approval should the slice move into a
   real fixture path.

The screening pass is a best-effort heuristic layer, not a certification.
"Clean" means "no configured pattern fired" — an operator still has to read
the transcripts before promoting anything to a shared fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from polylogue.api import Polylogue


class _MessageLike(Protocol):
    """Structural shape ``_flatten_session_text`` reads from a message.

    A ``Protocol`` (not the concrete ``polylogue.archive.session.domain_models
    .Session``/``Message`` classes) so the real session objects returned by
    ``Polylogue.get_session`` and the lightweight duck-typed test doubles in
    ``tests/unit/devtools/test_proof_world_real_slice.py`` both satisfy the
    parameter type structurally, without the tests needing to construct or
    subclass the full domain model. Declared as read-only ``@property``
    members rather than plain attributes: mypy checks plain-attribute
    Protocol members *invariantly* (both read and write), which the real
    ``Message.blocks: list[dict[str, object]]`` fails against a plain
    ``blocks: object`` attribute even though every value it can hold is
    assignable to ``object``. Properties are read-only, so the check is
    covariant instead and both the real model and the test doubles conform.
    """

    @property
    def text(self) -> str | None: ...

    @property
    def blocks(self) -> object: ...


class _SessionLike(Protocol):
    """Structural shape ``_flatten_session_text`` reads from a session.

    ``Iterable``, not ``Sequence`` — the real ``Session.messages`` is a
    ``MessageCollection`` that supports iteration but is not a nominal
    ``collections.abc.Sequence`` subclass, and ``Sequence`` is a concrete ABC
    in typeshed (not a structural ``Protocol``) so mypy would reject it here
    even though the object is sequence-*shaped*. Only iteration is needed.
    """

    @property
    def messages(self) -> Iterable[_MessageLike]: ...


# Patterns that indicate a live secret/credential shape. Kept intentionally
# narrow (favor false negatives over drowning the report in noise) — this is
# a triage aid, not a DLP product.
_SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key_id": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic_credential_assignment": re.compile(
        r"(?i)\b(api[_-]?key|secret|password|passwd|access[_-]?token)\b\s*[:=]\s*"
        r"['\"]?[A-Za-z0-9_\-/+=.]{12,}"
    ),
    "bearer_token": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-.=]{16,}"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "ssh_public_key": re.compile(r"\bssh-(?:rsa|ed25519) [A-Za-z0-9+/]{20,}"),
    "jwt_like": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
}

# Patterns that may indicate personal information. These fire far more often
# on ordinary dev-work text (localhost IPs, placeholder emails), so callers
# should read the samples rather than treat any hit as disqualifying.
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "home_path": re.compile(r"/home/[a-zA-Z0-9_-]+"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Values that are conventionally placeholders/loopback addresses, not real
# personal data. A hit is downgraded only when the *entire* matched string
# equals one of these exactly (never a substring check — "notuser@example.com"
# and "127.0.0.123" must NOT be treated as allowlisted merely because an
# allowlisted value happens to appear inside them). A downgraded hit is still
# reported (never silently dropped).
_ALLOWLIST_VALUES: frozenset[str] = frozenset(
    (
        "test@example.com",
        "user@example.com",
        "127.0.0.1",
        "0.0.0.0",
        "255.255.255.255",
    )
)

_SNIPPET_RADIUS = 40
_MAX_SAMPLES_PER_PATTERN = 5


@dataclass(slots=True)
class PatternHit:
    pattern: str
    kind: str  # "secret" | "pii"
    count: int
    samples: list[str] = field(default_factory=list)
    all_allowlisted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "kind": self.kind,
            "count": self.count,
            "samples": self.samples,
            "all_allowlisted": self.all_allowlisted,
        }


@dataclass(slots=True)
class SessionScreeningResult:
    session_id: str
    origin: str
    title: str | None
    created_at: str | None
    message_count: int
    word_count: int
    hits: list[PatternHit]

    @property
    def verdict(self) -> str:
        secret_hits = [h for h in self.hits if h.kind == "secret"]
        if secret_hits:
            return "flagged"
        pii_hits = [h for h in self.hits if h.kind == "pii" and not h.all_allowlisted]
        if pii_hits:
            return "review"
        return "clean"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "origin": self.origin,
            "title": self.title,
            "created_at": self.created_at,
            "message_count": self.message_count,
            "word_count": self.word_count,
            "verdict": self.verdict,
            "hits": [h.to_dict() for h in self.hits],
        }


def _snippet(text: str, match: re.Match[str], *, redact: bool) -> str:
    """Render a bounded context window around ``match``.

    When ``redact`` is true (secret-kind hits), the matched substring itself
    is replaced with a placeholder — the *surrounding* context is still
    useful for triage (which pattern fired, roughly where), but the actual
    secret value never reaches the report/manifest on disk. PII hits are not
    redacted: a human reviewer needs the real matched text (e.g. the actual
    email/IP) to judge whether it is a placeholder or genuine personal data.
    """

    start = max(0, match.start() - _SNIPPET_RADIUS)
    end = min(len(text), match.end() + _SNIPPET_RADIUS)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    window = text[start:end]
    if redact:
        rel_start = match.start() - start
        rel_end = match.end() - start
        window = f"{window[:rel_start]}<redacted:{match.end() - match.start()}ch>{window[rel_end:]}"
    return f"{prefix}{window!r}{suffix}"


def scan_text(text: str) -> list[PatternHit]:
    """Run every configured secret/PII pattern over ``text``.

    Returns one :class:`PatternHit` per pattern that matched at least once,
    each carrying up to ``_MAX_SAMPLES_PER_PATTERN`` samples for human
    review. Secret-kind samples have the actual matched value redacted (see
    :func:`_snippet`) — never the raw secret — to keep the report itself
    from becoming a leak surface. PII-kind samples keep the real matched
    text, which a reviewer needs to judge placeholder vs. genuine data.
    """

    hits: list[PatternHit] = []
    for kind, patterns in (("secret", _SECRET_PATTERNS), ("pii", _PII_PATTERNS)):
        redact = kind == "secret"
        for name, pattern in patterns.items():
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            samples = [_snippet(text, m, redact=redact) for m in matches[:_MAX_SAMPLES_PER_PATTERN]]
            all_allowlisted = all(m.group(0) in _ALLOWLIST_VALUES for m in matches)
            hits.append(
                PatternHit(
                    pattern=name,
                    kind=kind,
                    count=len(matches),
                    samples=samples,
                    all_allowlisted=all_allowlisted,
                )
            )
    return hits


def _flatten_session_text(session: _SessionLike) -> str:
    """Flatten every message's text and structured blocks to one string."""

    parts: list[str] = []
    for message in session.messages:
        if message.text:
            parts.append(message.text)
        if message.blocks:
            parts.append(json.dumps(message.blocks, default=str))
    return "\n".join(parts)


async def _screen_session_with(poly: Polylogue, session_id: str) -> tuple[SessionScreeningResult, str]:
    """Screen one session through an already-open ``Polylogue`` instance."""

    session = await poly.get_session(session_id)
    if session is None:
        raise ValueError(f"session not found in archive: {session_id}")
    text = _flatten_session_text(session)
    word_count = len(text.split())
    result = SessionScreeningResult(
        session_id=str(session.id),
        origin=str(session.origin),
        title=session.title,
        created_at=str(session.created_at) if session.created_at else None,
        message_count=len(session.messages),
        word_count=word_count,
        hits=scan_text(text),
    )
    return result, text


async def screen_session(archive_root: Path, session_id: str) -> tuple[SessionScreeningResult, str]:
    """Load one session read-only and screen it. Returns (result, transcript_text).

    Opens and closes its own scoped ``Polylogue`` instance — the convenient
    single-session entry point used by tests and one-off callers. Batch
    callers should use :func:`screen_sessions`, which opens the archive once
    and reuses the same instance across every session id instead of paying
    the open/close cost per id.
    """

    from polylogue.api import Polylogue

    async with Polylogue(archive_root=archive_root) as pl:
        return await _screen_session_with(pl, session_id)


async def screen_sessions(archive_root: Path, session_ids: list[str]) -> list[tuple[SessionScreeningResult, str]]:
    """Screen every id in ``session_ids`` through one shared archive open."""

    from polylogue.api import Polylogue

    async with Polylogue(archive_root=archive_root) as pl:
        return [await _screen_session_with(pl, session_id) for session_id in session_ids]


def render_report_markdown(results: list[SessionScreeningResult], *, archive_root: Path) -> str:
    lines = [
        "# Real-archive candidate slice — privacy screening report",
        "",
        f"Archive root: `{archive_root}`",
        f"Sessions screened: {len(results)}",
        "",
        "This is an automated triage pass (pattern matching only). It is not a",
        "certification. An operator must read the transcripts before any of",
        "this content is folded into the shared demo proof-world fixture.",
        "",
        "| session_id | origin | verdict | messages | words | flags |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in results:
        flags = ", ".join(f"{h.pattern}×{h.count}" for h in r.hits) or "—"
        lines.append(
            f"| `{r.session_id}` | {r.origin} | **{r.verdict}** | {r.message_count} | {r.word_count} | {flags} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## `{r.session_id}`")
        lines.append("")
        lines.append(f"- title: {r.title!r}")
        lines.append(f"- created_at: {r.created_at}")
        lines.append(f"- verdict: **{r.verdict}**")
        if not r.hits:
            lines.append("- no pattern hits")
        for h in r.hits:
            lines.append(f"- `{h.pattern}` ({h.kind}) × {h.count}{' (all allowlisted)' if h.all_allowlisted else ''}")
            for sample in h.samples:
                lines.append(f"  - {sample}")
        lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools demo real-slice-screen",
        description=(
            "Read-only extraction and privacy screening of a candidate real-archive "
            "session slice, for later human-gated inclusion in the shared demo "
            "proof-world corpus (polylogue-212.11)."
        ),
    )
    parser.add_argument("--archive-root", type=Path, required=True, help="Real archive root to read (read-only).")
    parser.add_argument(
        "--session",
        dest="sessions",
        action="append",
        default=[],
        help="Session id to screen (repeatable).",
    )
    parser.add_argument(
        "--refs-file",
        type=Path,
        default=None,
        help="Optional file with one session id per line (blank lines and '#' comments ignored).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the report + transcripts.")
    return parser


def _safe_transcript_filename(session_id: str) -> str:
    """Filesystem-safe, collision-free filename stem for a session id.

    Path-unsafe characters are replaced for readability, but readability
    alone is not collision-safe: distinct session ids that differ only in
    which punctuation character separates otherwise-identical characters
    (e.g. ``origin:a:b`` vs. ``origin:a_b``) would sanitize to the same
    stem. A short content hash of the *original, unsanitized* session id is
    appended so two distinct session ids can never produce the same
    filename, guaranteeing no transcript is silently overwritten.
    """

    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    return f"{sanitized}__{digest}"


def _read_refs_file(path: Path) -> list[str]:
    refs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        refs.append(stripped)
    return refs


def main(argv: list[str] | None = None) -> int:
    import asyncio

    args = _parser().parse_args(argv)
    session_ids = list(args.sessions)
    if args.refs_file is not None:
        session_ids.extend(_read_refs_file(args.refs_file))
    session_ids = list(dict.fromkeys(session_ids))  # de-dupe, preserve order
    if not session_ids:
        print("no session ids given (use --session or --refs-file)", file=sys.stderr)
        return 2

    pairs = asyncio.run(screen_sessions(args.archive_root, session_ids))
    results = [r for r, _ in pairs]

    args.out.mkdir(parents=True, exist_ok=True)
    transcripts_dir = args.out / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    seen_filenames: dict[str, str] = {}
    for result, text in pairs:
        safe_name = _safe_transcript_filename(result.session_id)
        prior = seen_filenames.get(safe_name)
        if prior is not None and prior != result.session_id:
            # Should be unreachable (sha256 collision on distinct inputs),
            # but fail loudly rather than silently overwrite a transcript.
            raise RuntimeError(
                f"transcript filename collision: {safe_name!r} claimed by both {prior!r} and {result.session_id!r}"
            )
        seen_filenames[safe_name] = result.session_id
        (transcripts_dir / f"{safe_name}.txt").write_text(text, encoding="utf-8")

    manifest = {
        "archive_root": str(args.archive_root),
        "sessions": [r.to_dict() for r in results],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report_md = render_report_markdown(results, archive_root=args.archive_root)
    (args.out / "SCREENING_REPORT.md").write_text(report_md + "\n", encoding="utf-8")

    flagged = [r for r in results if r.verdict == "flagged"]
    review = [r for r in results if r.verdict == "review"]
    print(
        f"screened {len(results)} sessions: {len(flagged)} flagged, {len(review)} need review, "
        f"{len(results) - len(flagged) - len(review)} clean"
    )
    print(f"report: {args.out / 'SCREENING_REPORT.md'}")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
