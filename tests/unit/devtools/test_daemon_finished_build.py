"""Synthetic stop-predicate checks for the ordinary-daemon scratch probe."""

from dataclasses import replace
from pathlib import Path

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
    assert "POLYLOGUE_CONFIG" not in env and "POLYLOGUE_HERMES_ROOT" not in env
    assert env["POLYLOGUE_SITE_CONFIG"] == ""
    assert env["PATH"] == "/usr/bin"
