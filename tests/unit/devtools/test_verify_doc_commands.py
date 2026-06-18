"""Tests for devtools/verify_doc_commands.py.

Covers the doc-command lint that closes #1262: every command mentioned
in shipped documentation must resolve against the live ``polylogued``
or ``devtools`` subcommand inventory, and the explicit stale-command
denylist must trip when re-introduced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools.verify_doc_commands import (
    STALE_INVOCATIONS,
    check_docs,
    main,
)


def _write_docs(root: Path, files: dict[str, str]) -> None:
    """Materialise an in-memory file map under root, creating dirs."""
    for relpath, content in files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


class TestCheckDocsRepoBaseline:
    """The committed README and docs/ tree must pass the lint."""

    def test_repo_docs_pass(self) -> None:
        errors, files_checked = check_docs()
        assert errors == [], "\n".join(errors)
        assert files_checked > 0


class TestCheckDocsTmpFixtures:
    def test_known_devtools_command_passes(self, tmp_path: Path) -> None:
        _write_docs(
            tmp_path,
            {
                "README.md": "```bash\ndevtools render all\n```\n",
            },
        )
        errors, files_checked = check_docs(root=tmp_path)
        assert errors == []
        assert files_checked == 1

    def test_known_polylogued_command_passes(self, tmp_path: Path) -> None:
        _write_docs(
            tmp_path,
            {
                "README.md": "```bash\npolylogued run\n```\n",
            },
        )
        errors, files_checked = check_docs(root=tmp_path)
        assert errors == []

    def test_unknown_devtools_command_blocks(self, tmp_path: Path) -> None:
        _write_docs(
            tmp_path,
            {
                "README.md": "```bash\ndevtools not-a-real-command\n```\n",
            },
        )
        errors, _ = check_docs(root=tmp_path)
        assert any("not-a-real-command" in e for e in errors)

    def test_unknown_nested_devtools_command_blocks(self, tmp_path: Path) -> None:
        _write_docs(
            tmp_path,
            {
                "README.md": "```bash\ndevtools render imaginary-surface\n```\n",
            },
        )
        errors, _ = check_docs(root=tmp_path)
        assert any("render imaginary-surface" in e for e in errors)

    def test_unknown_polylogued_command_blocks(self, tmp_path: Path) -> None:
        _write_docs(
            tmp_path,
            {
                "README.md": "```bash\npolylogued imaginary-subcommand\n```\n",
            },
        )
        errors, _ = check_docs(root=tmp_path)
        assert any("imaginary-subcommand" in e for e in errors)

    def test_stale_enable_api_blocks(self, tmp_path: Path) -> None:
        _write_docs(
            tmp_path,
            {
                "README.md": "```bash\npolylogued run --enable-api\n```\n",
            },
        )
        errors, _ = check_docs(root=tmp_path)
        assert any("--enable-api" in e for e in errors)

    def test_stale_polylogue_run_source_blocks(self, tmp_path: Path) -> None:
        _write_docs(
            tmp_path,
            {
                "README.md": "```bash\npolylogue run --source claude-code\n```\n",
            },
        )
        errors, _ = check_docs(root=tmp_path)
        assert any("polylogue run --source" in e for e in errors)

    def test_prose_mention_not_flagged(self, tmp_path: Path) -> None:
        """Prose ('polylogue and devtools share a flow') must be ignored."""
        _write_docs(
            tmp_path,
            {
                "README.md": (
                    "Polylogue ships polylogue, polylogued, and devtools binaries.\n"
                    "The polylogued daemon and the devtools control plane share a workflow.\n"
                ),
            },
        )
        errors, _ = check_docs(root=tmp_path)
        assert errors == []

    def test_systemd_unit_filename_not_flagged(self, tmp_path: Path) -> None:
        _write_docs(
            tmp_path,
            {
                "docs/note.md": ("```bash\nsystemctl --user start polylogued.service\n```\n"),
            },
        )
        errors, _ = check_docs(root=tmp_path)
        assert errors == []

    def test_inline_code_span_checked(self, tmp_path: Path) -> None:
        _write_docs(
            tmp_path,
            {
                "README.md": "Use `polylogued totally-fake` to ingest.\n",
            },
        )
        errors, _ = check_docs(root=tmp_path)
        assert any("totally-fake" in e for e in errors)

    def test_bash_comment_skipped(self, tmp_path: Path) -> None:
        """A '# ... polylogued runs ...' comment is prose, not invocation."""
        _write_docs(
            tmp_path,
            {
                "docs/x.md": ("```bash\n# example convergence work (polylogued runs, ingest)\npolylogued run\n```\n"),
            },
        )
        errors, _ = check_docs(root=tmp_path)
        assert errors == []


class TestStaleInvocationCoverage:
    @pytest.mark.parametrize("needle,_hint", STALE_INVOCATIONS)
    def test_each_stale_invocation_blocks(self, tmp_path: Path, needle: str, _hint: str) -> None:
        _write_docs(
            tmp_path,
            {
                "README.md": f"```bash\n{needle.rstrip()} extra-arg\n```\n",
            },
        )
        errors, _ = check_docs(root=tmp_path)
        # Both the substring-match denylist and the subcommand check may
        # surface the issue; the denylist message is the targeted one.
        assert any(needle.rstrip() in e for e in errors), errors


class TestMainEntrypoint:
    def test_exit_zero_on_clean_tree(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Run against the real repo: must currently be clean.
        rc = main([])
        assert rc == 0

    def test_json_mode_emits_blocking_field(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["--json"])
        captured = capsys.readouterr()
        assert rc == 0
        assert "blocking" in captured.out
