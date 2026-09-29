"""Sidecar absence is an observation, not permission to sample a later tree."""

from pathlib import Path

import pytest

from polylogue.sources.live.sidecar_resolution import FencedFilesystemSidecarResolver
from polylogue.sources.live.tool_result_sidecars import resolve_tool_results_dir


def _scope(tmp_path: Path) -> tuple[Path, Path]:
    transcript = tmp_path / "synthetic-session.jsonl"
    transcript.write_text("{}\n")
    directory = resolve_tool_results_dir(transcript)
    assert directory is not None
    return transcript, directory


def test_absent_sidecar_directory_remains_part_of_the_parse_fence(tmp_path: Path) -> None:
    transcript, directory = _scope(tmp_path)
    parsed = FencedFilesystemSidecarResolver()
    assert not parsed.claude_code_scope(transcript).available

    # The parser has already observed absence. Publishing against this newer
    # tree must refuse rather than attach its identity to the old parse.
    directory.mkdir(parents=True)
    (directory / "tool-X.txt").write_text("arrived after parsing")
    current = FencedFilesystemSidecarResolver()
    assert current.claude_code_scope(transcript).available
    assert parsed.claude_code_fence(transcript) != current.claude_code_fence(transcript)


def test_scope_membership_aba_cannot_seal_a_mixed_parse(tmp_path: Path) -> None:
    transcript, directory = _scope(tmp_path)
    directory.mkdir(parents=True)
    parsed = FencedFilesystemSidecarResolver()
    assert parsed.claude_code_scope(transcript).files == ()
    sidecar = directory / "tool-X.txt"
    sidecar.write_text("temporary expansion")
    changed = parsed.claude_code_scope(transcript)
    assert len(changed.files) == 1
    assert changed.files[0].read_text() == "temporary expansion"
    sidecar.unlink()

    current = FencedFilesystemSidecarResolver()
    assert current.claude_code_scope(transcript).files == ()
    assert parsed.claude_code_fence(transcript) != current.claude_code_fence(transcript)


def test_unread_sidecars_are_hashed_without_materializing_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript, directory = _scope(tmp_path)
    directory.mkdir(parents=True)
    sidecar = directory / "tool-X.txt"
    sidecar.write_text("unread synthetic sidecar\n" * 1024)
    expected = FencedFilesystemSidecarResolver()
    expected.claude_code_scope(transcript)
    expected_fence = expected.claude_code_fence(transcript)

    original_read_bytes = Path.read_bytes

    def refuse_materialization(path: Path) -> bytes:
        if path == sidecar:
            raise AssertionError("An unread sidecar must be hashed as a stream")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", refuse_materialization)
    parsed = FencedFilesystemSidecarResolver()
    parsed.claude_code_scope(transcript)
    assert parsed.claude_code_fence(transcript) == expected_fence
