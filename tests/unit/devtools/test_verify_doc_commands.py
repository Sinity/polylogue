"""Tests for devtools/verify_doc_commands.py.

Covers the doc-command lint: executable examples resolve against the live
``polylogue``, ``polylogued``, or ``devtools`` command inventories.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools.verify_doc_commands import check_docs, main
from polylogue.daemon import blob_gc_periodic
from polylogue.daemon import cli as daemon_cli


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

    def test_blob_gc_recovery_documents_gated_bounded_daemon_route(self) -> None:
        """Keep the recovery runbook aligned with the daemon's actual schedule.

        This contract checks the cleanup ownership boundary as well as timing.
        It uses the production registry and handler maps, so a stale document
        cannot pass merely because a forbidden token was removed from prose.
        """
        root = Path(__file__).parents[3]
        text = (root / "docs" / "maintenance.md").read_text()
        cli_reference = (root / "docs" / "cli-reference.md").read_text()
        heading = "### Recovering a corrupt blob store"
        start = text.index(heading)
        next_heading = text.find("\n### ", start + len(heading))
        section = text[start:] if next_heading == -1 else text[start:next_heading]
        lower_section = section.casefold()

        restart = section.index("systemctl --user start polylogued.service")
        gate_timeout = f"{daemon_cli._CATCH_UP_GATE_TIMEOUT_SECONDS:g}-second gate timeout"
        interval = f"{blob_gc_periodic.BLOB_GC_INTERVAL_SECONDS:g}-second interval"
        max_batch = f"at most {blob_gc_periodic.BLOB_GC_MAX_BATCH} blobs"
        first_wait = section.index(interval)
        gate_release = lower_section.index("after the catch-up event or timeout")

        from polylogue.maintenance.targets import build_maintenance_target_catalog
        from polylogue.storage import repair

        target_name = "orphaned_blobs"
        catalog = build_maintenance_target_catalog()
        assert catalog.resolve_name(target_name) is None
        assert target_name not in repair.PREVIEW_HANDLERS
        assert target_name not in repair.REPAIR_HANDLERS
        assert target_name not in text
        assert target_name not in cli_reference

        assert gate_timeout in section
        assert "daemon-owned blob-gc loop" in lower_section
        assert restart < gate_release < first_wait, (
            "restart must precede the catch-up gate and the first periodic wait"
        )
        assert max_batch in section
        assert "eligible leftovers are handled by later passes" in lower_section
        assert "manual orphaned-blob repair is not a supported route" in lower_section
        assert "reservation ttls must not be inferred" in lower_section


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


class TestPolylogueCommandRecognition:
    """Query-first examples validate live commands without a stale-name registry."""

    def test_recognized_verb_with_valid_flag_passes(self, tmp_path: Path) -> None:
        _write_docs(
            tmp_path,
            {"README.md": "```bash\npolylogue --origin claude-code-session analyze --count\n```\n"},
        )
        errors, _ = check_docs(root=tmp_path)
        assert errors == [], errors

    def test_unknown_flag_on_recognized_verb_fails(self, tmp_path: Path) -> None:
        _write_docs(tmp_path, {"README.md": "```bash\npolylogue analyze --bogus-flag\n```\n"})
        errors, _ = check_docs(root=tmp_path)
        assert any("--bogus-flag" in e for e in errors), errors

    def test_quoted_free_text_query_passes(self, tmp_path: Path) -> None:
        _write_docs(tmp_path, {"README.md": '```bash\npolylogue "rate limiting retries"\n```\n'})
        errors, _ = check_docs(root=tmp_path)
        assert errors == [], errors

    @pytest.mark.parametrize(
        "invocation",
        (
            "polylogue find rate limiting retries",
            "polylogue repo:polylogue",
        ),
    )
    def test_explicit_query_intent_passes(self, tmp_path: Path, invocation: str) -> None:
        _write_docs(tmp_path, {"README.md": f"```bash\n{invocation}\n```\n"})
        errors, _ = check_docs(root=tmp_path)
        assert errors == [], errors

    @pytest.mark.parametrize(
        "invocation",
        (
            "polylogue rate limiting retries",
            "polylogue list",
            "polylogue show abc",
        ),
    )
    def test_unsignalled_query_root_fails(self, tmp_path: Path, invocation: str) -> None:
        _write_docs(tmp_path, {"README.md": f"```bash\n{invocation}\n```\n"})
        errors, _ = check_docs(root=tmp_path)
        assert any("does not signal query intent" in error for error in errors), errors

    def test_leaf_subcommand_flag_resolves(self, tmp_path: Path) -> None:
        # The flag lives on the ``analyze insights profiles`` leaf, not the
        # ``analyze`` group — full-path resolution must accept it.
        _write_docs(
            tmp_path,
            {"README.md": "```bash\npolylogue analyze insights profiles --tier merged\n```\n"},
        )
        errors, _ = check_docs(root=tmp_path)
        assert errors == [], errors

    def test_renamed_flag_fails(self, tmp_path: Path) -> None:
        # ``--provider`` was renamed to ``--origin``.
        _write_docs(tmp_path, {"README.md": "```bash\npolylogue read --provider claude-code\n```\n"})
        errors, _ = check_docs(root=tmp_path)
        assert any("--provider" in e for e in errors), errors

    def test_then_chain_is_left_alone(self, tmp_path: Path) -> None:
        # ``then`` chains attribute flags to different verbs; skip flag checks.
        _write_docs(
            tmp_path,
            {"README.md": "```bash\npolylogue find id:abc then read --view messages\n```\n"},
        )
        errors, _ = check_docs(root=tmp_path)
        assert errors == [], errors

    def test_dated_audit_docs_are_excluded(self, tmp_path: Path) -> None:
        # Point-in-time audits assert past command state; not held to current.
        _write_docs(
            tmp_path,
            {"docs/audits/2020-01-01-x.md": "```bash\npolylogue list\n```\n"},
        )
        errors, _ = check_docs(root=tmp_path)
        assert errors == [], errors
