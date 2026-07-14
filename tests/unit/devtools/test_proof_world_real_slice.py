"""Tests for the real-archive candidate-slice screening harness (polylogue-212.11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import proof_world_real_slice as m
from polylogue.core.enums import Provider
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.storage_records import SessionBuilder, db_setup

_CLEAN_ID = native_session_id_for("claude-ai", "clean-session")
_SECRET_ID = native_session_id_for("claude-ai", "secret-session")


class _FakeMessage:
    def __init__(self, text: str | None, blocks: object = None) -> None:
        self.text = text
        self.blocks = blocks


class _FakeSession:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self.messages = messages


# --- pure scan_text() behavior ---------------------------------------------


def test_scan_text_finds_no_hits_on_clean_text() -> None:
    hits = m.scan_text("just an ordinary sentence about pytest and git diffs")
    assert hits == []


def test_scan_text_flags_a_credential_shaped_string() -> None:
    hits = m.scan_text("aws key: AKIAABCDEFGHIJKLMNOP")
    names = {h.pattern for h in hits}
    assert "aws_access_key_id" in names
    secret_hit = next(h for h in hits if h.pattern == "aws_access_key_id")
    assert secret_hit.kind == "secret"
    assert secret_hit.count == 1
    assert not secret_hit.all_allowlisted


def test_scan_text_allowlists_placeholder_email_and_loopback() -> None:
    hits = m.scan_text("git config user.email test@example.com; server on 127.0.0.1")
    email_hit = next(h for h in hits if h.pattern == "email")
    ipv4_hit = next(h for h in hits if h.pattern == "ipv4")
    assert email_hit.all_allowlisted
    assert ipv4_hit.all_allowlisted


def test_scan_text_does_not_allowlist_a_real_looking_email() -> None:
    hits = m.scan_text("contact jane.doe@personalmail.example for details")
    email_hit = next(h for h in hits if h.pattern == "email")
    assert not email_hit.all_allowlisted


def test_scan_text_does_not_allowlist_by_substring_containment() -> None:
    """A match that merely *contains* an allowlisted value must not be
    downgraded — only an exact full-match equals check counts. Regression
    for a bug where `notuser@example.com` and `127.0.0.123` were both
    treated as fully allowlisted (and thus 'clean') purely because
    `user@example.com`/`127.0.0.1` occur as substrings."""

    hits = m.scan_text("contact notuser@example.com and reach server at 127.0.0.123 for details")
    email_hit = next(h for h in hits if h.pattern == "email")
    ipv4_hit = next(h for h in hits if h.pattern == "ipv4")
    assert not email_hit.all_allowlisted
    assert not ipv4_hit.all_allowlisted

    result = m.SessionScreeningResult(
        session_id="x",
        origin="claude-code-session",
        title=None,
        created_at=None,
        message_count=1,
        word_count=1,
        hits=[email_hit, ipv4_hit],
    )
    assert result.verdict == "review"


# --- verdict computation -----------------------------------------------------


def test_verdict_clean_when_no_hits() -> None:
    result = m.SessionScreeningResult(
        session_id="x",
        origin="claude-code-session",
        title=None,
        created_at=None,
        message_count=1,
        word_count=1,
        hits=[],
    )
    assert result.verdict == "clean"


def test_verdict_flagged_beats_review_when_a_secret_hits() -> None:
    result = m.SessionScreeningResult(
        session_id="x",
        origin="claude-code-session",
        title=None,
        created_at=None,
        message_count=1,
        word_count=1,
        hits=[
            m.PatternHit(pattern="email", kind="pii", count=1, samples=["s"], all_allowlisted=False),
            m.PatternHit(pattern="openai_style_key", kind="secret", count=1, samples=["s"], all_allowlisted=False),
        ],
    )
    assert result.verdict == "flagged"


def test_verdict_review_when_only_non_allowlisted_pii_hits() -> None:
    result = m.SessionScreeningResult(
        session_id="x",
        origin="claude-code-session",
        title=None,
        created_at=None,
        message_count=1,
        word_count=1,
        hits=[m.PatternHit(pattern="home_path", kind="pii", count=1, samples=["s"], all_allowlisted=False)],
    )
    assert result.verdict == "review"


def test_verdict_clean_when_pii_hits_are_fully_allowlisted() -> None:
    result = m.SessionScreeningResult(
        session_id="x",
        origin="claude-code-session",
        title=None,
        created_at=None,
        message_count=1,
        word_count=1,
        hits=[m.PatternHit(pattern="email", kind="pii", count=1, samples=["s"], all_allowlisted=True)],
    )
    assert result.verdict == "clean"


# --- secret redaction in samples ----------------------------------------------


def test_scan_text_redacts_the_actual_secret_value_in_samples() -> None:
    """The report/manifest must never embed a raw secret value verbatim —
    only PII context does that. Regression for a bug where the snippet
    window always fully contained the matched secret text despite the
    docstring's claim that samples 'never' leak the full match."""

    text = "the real key is AKIAABCDEFGHIJKLMNOP and it must stay secret"
    hits = m.scan_text(text)
    secret_hit = next(h for h in hits if h.pattern == "aws_access_key_id")
    joined = " ".join(secret_hit.samples)
    assert "AKIAABCDEFGHIJKLMNOP" not in joined
    assert "redacted" in joined
    # surrounding context should still be present for triage
    assert "real key" in joined


def test_scan_text_keeps_real_pii_text_in_samples_for_human_judgment() -> None:
    hits = m.scan_text("contact jane.doe@personalmail.example for details")
    email_hit = next(h for h in hits if h.pattern == "email")
    joined = " ".join(email_hit.samples)
    assert "jane.doe@personalmail.example" in joined


# --- transcript filename collision safety --------------------------------------


def test_safe_transcript_filename_disambiguates_punctuation_variants() -> None:
    """Two distinct session ids that differ only in which punctuation
    character separates otherwise-identical characters must never collide
    on the sanitized filename stem."""

    a = m._safe_transcript_filename("origin:a:b")
    b = m._safe_transcript_filename("origin:a_b")
    assert a != b


def test_safe_transcript_filename_is_deterministic() -> None:
    assert m._safe_transcript_filename("claude-code-session:abc") == m._safe_transcript_filename(
        "claude-code-session:abc"
    )


# --- flatten helper -----------------------------------------------------------


def test_flatten_session_text_includes_message_text_and_block_json() -> None:
    session = _FakeSession(
        [
            _FakeMessage(text="hello world"),
            _FakeMessage(text=None, blocks=[{"kind": "tool_use", "input": {"cmd": "ls"}}]),
        ]
    )
    flat = m._flatten_session_text(session)
    assert "hello world" in flat
    assert "tool_use" in flat
    assert '"cmd": "ls"' in flat


# --- report rendering -----------------------------------------------------


def test_render_report_markdown_includes_verdict_and_samples() -> None:
    result = m.SessionScreeningResult(
        session_id="claude-code-session:abc",
        origin="claude-code-session",
        title="Some session",
        created_at="2026-01-01T00:00:00Z",
        message_count=3,
        word_count=42,
        hits=[m.PatternHit(pattern="home_path", kind="pii", count=2, samples=["…/home/x…"], all_allowlisted=False)],
    )
    md = m.render_report_markdown([result], archive_root=Path("/fake/archive"))
    assert "claude-code-session:abc" in md
    assert "**review**" in md
    assert "home_path" in md
    assert "/home/x" in md


# --- end-to-end read path against a seeded archive --------------------------


async def _seed(db_path: Path) -> None:
    await (
        SessionBuilder(db_path, "clean-session")
        .provider(Provider.CLAUDE_AI.value)
        .title("Clean session")
        .add_message(text="just discussing pytest fixtures, nothing sensitive")
        .build()
    )
    await (
        SessionBuilder(db_path, "secret-session")
        .provider(Provider.CLAUDE_AI.value)
        .title("Session with a planted secret")
        .add_message(text="here is my key: AKIAABCDEFGHIJKLMNOP please rotate it")
        .build()
    )


async def test_screen_session_reads_real_archive_and_flags_planted_secret(
    workspace_env: dict[str, Path],
) -> None:
    db_path = db_setup(workspace_env)
    await _seed(db_path)
    archive_root = db_path.parent

    clean_result, clean_text = await m.screen_session(archive_root, _CLEAN_ID)
    assert clean_result.verdict == "clean"
    assert "pytest fixtures" in clean_text

    secret_result, secret_text = await m.screen_session(archive_root, _SECRET_ID)
    assert secret_result.verdict == "flagged"
    assert any(h.pattern == "aws_access_key_id" for h in secret_result.hits)
    # the raw transcript text is unredacted (it exists for full human review)...
    assert "AKIAABCDEFGHIJKLMNOP" in secret_text
    # ...but the report-facing samples must never carry the raw secret value
    all_samples = [s for h in secret_result.hits for s in h.samples]
    assert not any("AKIAABCDEFGHIJKLMNOP" in s for s in all_samples)


async def test_screen_session_raises_for_unknown_session(workspace_env: dict[str, Path]) -> None:
    db_path = db_setup(workspace_env)
    await _seed(db_path)

    with pytest.raises(ValueError, match="not found"):
        await m.screen_session(db_path.parent, "claude-code-session:does-not-exist")


# --- CLI argument handling ---------------------------------------------------


def test_main_exits_nonzero_with_no_session_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = m.main(["--archive-root", str(tmp_path), "--out", str(tmp_path / "out")])
    assert code == 2
    captured = capsys.readouterr()
    assert "no session ids given" in captured.err


def test_read_refs_file_skips_blanks_and_comments(tmp_path: Path) -> None:
    refs_path = tmp_path / "refs.txt"
    refs_path.write_text("\n# a comment\nclaude-code-session:a:b\n\nclaude-code-session:c:d\n", encoding="utf-8")
    assert m._read_refs_file(refs_path) == [
        "claude-code-session:a:b",
        "claude-code-session:c:d",
    ]
