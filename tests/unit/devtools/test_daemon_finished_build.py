"""Synthetic stop-predicate checks for the ordinary-daemon scratch probe."""

from dataclasses import replace
from pathlib import Path

import pytest

from devtools.daemon_finished_build import REQUIRED_READINESS_DOMAINS, BuildEvidence


def _ready_evidence() -> BuildEvidence:
    return BuildEvidence(
        input_bytes=419_476_000,
        input_sha256="a" * 64,
        accepted_input_files=1,
        accepted_raw_rows=3,
        raw_parse_failures=0,
        raw_parse_pending=0,
        cursor_complete=True,
        promoted_index_path="/scratch/archive/.index-generations/gen-a/index.db",
        schema_identity="index-schema-sha256:synthetic",
        index_session_count=3,
        index_message_count=30,
        index_block_count=42,
        fts_source_rows=42,
        fts_indexed_rows=42,
        open_convergence_debt=0,
        readiness_surfaces=dict.fromkeys(REQUIRED_READINESS_DOMAINS, True),
        input_cursor={"source_path": "/scratch/input.jsonl", "complete": True},
        input_dispositions=(
            {
                "raw_id": "raw-synthetic",
                "terminal_disposition": "materialized",
                "materialized_session_count": 1,
                "membership_count": 1,
                "membership_pending_count": 0,
            },
        ),
        canonical_logical_digest="b" * 64,
        schema_object_census=(("table", "sessions"),),
        output_equivalence_key="c" * 64,
    )


def test_finished_daemon_probe_requires_positive_terminal_evidence() -> None:
    assert _ready_evidence().ready

    rejected = (
        replace(_ready_evidence(), accepted_input_files=0),
        replace(_ready_evidence(), accepted_raw_rows=0),
        replace(_ready_evidence(), raw_parse_failures=1),
        replace(_ready_evidence(), raw_parse_pending=1),
        replace(_ready_evidence(), cursor_complete=False),
        replace(_ready_evidence(), promoted_index_path=None),
        replace(_ready_evidence(), schema_identity=None),
        replace(_ready_evidence(), index_session_count=0),
        replace(_ready_evidence(), fts_source_rows=0),
        replace(_ready_evidence(), fts_indexed_rows=41),
        replace(_ready_evidence(), open_convergence_debt=1),
        replace(_ready_evidence(), readiness_surfaces={}),
        replace(_ready_evidence(), readiness_surfaces={"archive_sessions": True}),
        replace(
            _ready_evidence(),
            readiness_surfaces={**_ready_evidence().readiness_surfaces, "raw_artifacts": False},
        ),
        replace(
            _ready_evidence(),
            readiness_surfaces={**_ready_evidence().readiness_surfaces, "new_domain": False},
        ),
        replace(_ready_evidence(), input_cursor=None),
        replace(_ready_evidence(), input_dispositions=()),
        replace(
            _ready_evidence(),
            input_dispositions=({"terminal_disposition": "unmaterialized_or_unclassified"},),
        ),
        replace(
            _ready_evidence(),
            input_dispositions=(
                {
                    "terminal_disposition": "materialized",
                    "materialized_session_count": 1,
                    "membership_count": 0,
                    "membership_pending_count": 0,
                },
            ),
        ),
        replace(
            _ready_evidence(),
            input_dispositions=(
                {
                    "terminal_disposition": "materialized",
                    "materialized_session_count": 1,
                    "membership_count": 1,
                    "membership_pending_count": 1,
                },
            ),
        ),
        replace(_ready_evidence(), canonical_logical_digest=None),
        replace(_ready_evidence(), schema_object_census=()),
        replace(_ready_evidence(), output_equivalence_key=None),
    )
    assert all(not evidence.ready for evidence in rejected)


def test_convergence_is_not_finished_output_acceptance() -> None:
    converged = replace(
        _ready_evidence(),
        input_dispositions=(),
        canonical_logical_digest=None,
        schema_object_census=(),
        output_equivalence_key=None,
    )
    assert converged.converged
    assert not converged.ready


def test_qualification_environment_isolates_every_discovery_root(tmp_path: Path) -> None:
    """Inherited XDG roots and path overrides never reach the qualification daemon.

    Anti-vacuity: set only ``HOME`` and the inherited ``XDG_CONFIG_HOME`` (and
    with it the operator's real config) is passed to the run.
    """
    from devtools.daemon_finished_build import qualification_environment

    inherited = {
        "HOME": "/home/operator",
        "XDG_CONFIG_HOME": "/home/operator/.config",
        "XDG_DATA_HOME": "/home/operator/.local/share",
        "POLYLOGUE_CONFIG": "/home/operator/.config/polylogue/polylogue.toml",
        "POLYLOGUE_HERMES_ROOT": "/home/operator/.hermes",
        "PATH": "/usr/bin",
    }
    home = tmp_path / "home"

    env = qualification_environment(inherited, home=home, archive=tmp_path / "archive", candidate=tmp_path / "c")

    assert env["HOME"] == str(home)
    assert env["XDG_CONFIG_HOME"] == str(home / ".config")
    assert env["XDG_DATA_HOME"] == str(home / ".local/share")
    # Not merely dropped: an explicit override, pointed at a nonexistent
    # path under the isolated home, disables the <cwd>/polylogue.toml
    # fallback that a bare removal would leave live.
    assert env["POLYLOGUE_CONFIG"] != "/home/operator/.config/polylogue/polylogue.toml"
    assert Path(env["POLYLOGUE_CONFIG"]).is_relative_to(home)
    assert "POLYLOGUE_HERMES_ROOT" not in env
    assert env["POLYLOGUE_SITE_CONFIG"] == ""
    assert env["PATH"] == "/usr/bin"


def test_qualification_home_admits_only_the_declared_input(tmp_path: Path) -> None:
    """Any other artifact in the isolated home is refused, whatever its suffix.

    Anti-vacuity: enumerate only ``.json``/``.jsonl`` siblings and the Codex
    state database the daemon would acquire goes unreported.
    """
    from devtools.daemon_finished_build import undeclared_source_entries

    home = tmp_path / "home"
    sessions = home / ".codex" / "sessions" / "2026"
    sessions.mkdir(parents=True)
    source = sessions / "rollout.jsonl"
    source.write_text("{}\n")
    assert undeclared_source_entries(home, source) == []

    (home / ".codex" / "goals_1.sqlite").write_bytes(b"SQLite format 3\x00")
    (home / ".claude").mkdir()
    (home / ".claude" / "projects").symlink_to(tmp_path)

    assert undeclared_source_entries(home, source) == [".claude/projects", ".codex/goals_1.sqlite"]


def test_verify_args_rejects_a_suffix_the_watcher_never_admits(tmp_path: Path) -> None:
    """An input the canonical watcher ignores must fail fast, not time out.

    A correctly sized and hashed file at a canonical location with the wrong
    suffix passes containment but produces no cursor/raw evidence, so the
    qualification consumed its full default timeout instead of rejecting the
    invocation immediately.

    Anti-vacuity: drop the suffix check and this call succeeds instead of
    raising ``ValueError``.
    """
    import argparse

    from devtools.daemon_finished_build import _verify_args

    # The suffix check runs before candidate/source_root are used for
    # anything but path resolution, so a real git checkout is not needed.
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    source_root = tmp_path / "source-home"
    sessions = source_root / ".codex" / "sessions"
    sessions.mkdir(parents=True)
    source = sessions / "input.txt"
    payload = b"not jsonl"
    source.write_bytes(payload)

    args = argparse.Namespace(
        receipt=tmp_path / "receipt.json",
        candidate=candidate,
        source_root=source_root,
        input=source,
        expected_bytes=len(payload),
        expected_sha256=None,
        candidate_sha=None,
    )
    with pytest.raises(ValueError, match="canonical watcher admits"):
        _verify_args(args)
