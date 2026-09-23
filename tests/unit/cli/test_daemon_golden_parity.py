"""Golden parity for the daemon-backed CLI operation route.

polylogue-20d.1 acceptance criterion: "`--format json` output is byte-identical
between equivalent daemon-backed invocations for every read surface on the
demo corpus." A real production UDS daemon server is started against a seeded
archive and each invocation is run through :class:`click.testing.CliRunner`,
so this is an end-to-end regression test, not a mock of the daemon transport.

The daemon route renders one declared ``cli.query`` result through the shared
renderer, and its ``source`` provenance marker must name the UDS executor.
That marker is asserted explicitly, so a route that silently answers through a
retired local executor fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner

from tests.infra.daemon_operations import DaemonOperationStack, running_daemon_operations
from tests.infra.storage_records import SessionBuilder

pytestmark = pytest.mark.uses_real_clock(
    "polylogue-20d.1 golden-parity test starts the maintained production daemon operation stack; frozen_clock cannot substitute for its real writer/listener lifecycle."
)


@pytest.fixture
def golden_parity_workspace(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A real seeded archive for daemon-backed runs."""

    monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(cli_workspace["archive_root"]))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.delenv("POLYLOGUE_NO_DAEMON", raising=False)
    monkeypatch.delenv("POLYLOGUE_DAEMON", raising=False)

    return cli_workspace


def _seed_golden_archive(root: Path) -> None:
    """Seed the neutral corpus after the maintained daemon stack bootstraps."""

    (
        SessionBuilder(root / "index.db", "conv1")
        .provider("chatgpt")
        .title("Python Error Handling")
        .git_repository_url("polylogue")
        .add_message("m1", role="user", text="How to handle exceptions in Python?")
        .add_message("m2", role="assistant", text="Use try-except blocks.")
        .save()
    )
    (
        SessionBuilder(root / "index.db", "conv2")
        .provider("claude-code")
        .title("Rust Ownership")
        .git_repository_url("polylogue")
        .add_message("m3", role="user", text="What is ownership in Rust?")
        .add_message("m4", role="assistant", text="Rust ownership ensures memory safety.")
        .save()
    )


def _pin_cli_daemon_socket(monkeypatch: pytest.MonkeyPatch, stack: DaemonOperationStack) -> None:
    """Route CLI discovery to the fixture's independently-owned socket."""

    monkeypatch.setattr(
        "polylogue.daemon.socket_path.daemon_socket_path",
        lambda *_args, **_kwargs: stack.socket_path,
    )


def _run_find_json(args: list[str]) -> dict[str, object]:
    from polylogue.cli import cli

    runner = CliRunner()
    # `--repo` is a root option (`click_app.py::cli`), not
    # `find` verb options — they must precede `find` in argv. Passing `--repo`
    # here as the root option (rather than a `repo:polylogue` DSL query token,
    # which routes through a different rendering path with a distinct envelope
    # shape) is what exercises the plain `cli.query` list page this test
    # targets.
    root_flags = ["--plain", *args]
    result = runner.invoke(cli, [*root_flags, "find", "--format", "json", "--limit", "10"])
    assert result.exit_code == 0, result.output
    return dict(json.loads(result.output))


def _strip_provenance(envelope: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in envelope.items() if key != "source"}


def test_find_list_json_is_served_by_the_daemon(
    golden_parity_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = golden_parity_workspace["archive_root"]
    args = ["--repo", "polylogue"]

    with running_daemon_operations(archive_root, seed_archive=_seed_golden_archive) as stack:
        _pin_cli_daemon_socket(monkeypatch, stack)
        daemon_payload = _run_find_json(args)

    # Each leg must name the executor that actually answered, and everything
    # else must agree.  Anti-vacuity is the ``items`` assertion below -- two
    # empty pages would agree trivially.
    assert daemon_payload["source"] == "daemon"
    assert daemon_payload["items"], "fixture query must actually match rows"


def test_find_daemon_proxied_path_authenticates_with_auto_minted_token(
    golden_parity_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-n6pz: the daemon auto-mints and requires a bearer token by
    default (polylogue-rzve) when no ``daemon.api.auth_token`` is explicitly
    configured. The CLI fast path must resolve that same auto-minted token
    (``polylogue.daemon.api_auth.resolve_api_auth_token``) rather than only
    reading the unset config value -- otherwise every unauthenticated probe
    gets a 401, ``DaemonClient.probe`` returns ``None``, and the CLI refuses
    the reachable daemon instead of serving the request through its route.
    This test starts the real UDS server with the archive's actual
    auto-minted token (not an empty one) and asserts the daemon-proxied path
    is still reached."""
    from polylogue.daemon.api_auth import load_or_mint_api_auth_token

    archive_root = golden_parity_workspace["archive_root"]
    args = ["--repo", "polylogue"]

    with running_daemon_operations(archive_root, seed_archive=_seed_golden_archive) as stack:
        minted_token = load_or_mint_api_auth_token()
        assert minted_token
        stack.server.auth_token = minted_token
        _pin_cli_daemon_socket(monkeypatch, stack)
        daemon_payload = _run_find_json(args)

    assert daemon_payload["source"] == "daemon"
    assert daemon_payload["items"], "fixture query must actually match rows, or parity is vacuous"


def _run_read_messages_json(session_id: str) -> tuple[dict[str, object], str]:
    from polylogue.cli import cli

    runner = CliRunner()
    # `--no-daemon` is a root option (`click_app.py::cli`) and must precede the
    # verb; `--verbose` is what makes the route name its own executor on
    # stderr, which is the marker this parity test reads.
    result = runner.invoke(
        cli,
        [
            "--plain",
            "--verbose",
            "read",
            f"session:{session_id}",
            "--view",
            "messages",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    return dict(json.loads(result.stdout)), result.stderr


def test_read_messages_json_is_served_by_the_daemon(
    golden_parity_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-fko9.3: ``read --view messages`` is served by a reachable daemon.

    The messages view was the one read view that still computed its answer in
    this process, so a healthy daemon paid for a local archive open on every
    invocation.  Both legs now run one declared ``session.read`` window: the
    rendered document must agree field for field except the authority
    envelope, which names the executor that actually answered.

    Anti-vacuity: the ``messages`` assertion below -- two empty documents would
    agree trivially -- plus the per-leg ``served-by``/``server_identity``
    assertions, which a route that silently answered as the other one fails.
    """

    archive_root = golden_parity_workspace["archive_root"]
    session_id = "chatgpt-export:ext-conv1"

    with running_daemon_operations(archive_root, seed_archive=_seed_golden_archive) as stack:
        _pin_cli_daemon_socket(monkeypatch, stack)
        daemon_payload, daemon_err = _run_read_messages_json(session_id)

    assert "served-by: daemon (uds," in daemon_err, daemon_err
    assert cast("dict[str, object]", daemon_payload["authority"])["server_identity"] == "daemon"
    assert daemon_payload["messages"], "fixture session must actually hold rows"
    assert [message["text"] for message in cast("list[dict[str, object]]", daemon_payload["messages"])] == [
        "How to handle exceptions in Python?",
        "Use try-except blocks.",
    ]


def _strip_read_provenance(payload: dict[str, object]) -> dict[str, object]:
    """Drop the per-call authority envelope, which is provenance, not content.

    It carries the serving executor and that call's own wall-clock elapsed
    time; both necessarily differ between two independent invocations, and the
    executor identity is asserted explicitly on each leg above rather than
    merely stripped here.
    """

    return {key: value for key, value in payload.items() if key != "authority"}


def _seed_evidence_archive(root: Path) -> None:
    """Seed one session that actually carries per-session evidence rows.

    ``SessionBuilder`` writes the session tree only, so a file-edits parity
    over it would compare two empty bodies.  This writes a real Edit tool call
    and its result through the production writer, which is what populates the
    ``file_edits`` relation the view reads.
    """

    from polylogue.core.enums import BlockType, Provider, Role
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedFileEdit, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(root) as archive_db:
        archive_db.write_raw_and_parsed(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="evidence-parity",
                title="Evidence parity",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        position=0,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_USE,
                                tool_name="Edit",
                                tool_id="edit-parity-1",
                                tool_input={"file_path": "/tmp/parity.py"},
                            )
                        ],
                    ),
                    ParsedMessage(
                        provider_message_id="m2",
                        role=Role.USER,
                        position=1,
                        blocks=[
                            ParsedContentBlock(
                                type=BlockType.TOOL_RESULT,
                                outcome_unknown_reason="not_reported",
                                tool_id="edit-parity-1",
                                text="applied",
                                file_edit=ParsedFileEdit(
                                    file_path="/tmp/parity.py",
                                    structured_patch=[
                                        {"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}
                                    ],
                                    original_file="old contents\n",
                                    old_string="old",
                                    new_string="new",
                                    replace_all=False,
                                    user_modified=True,
                                ),
                            )
                        ],
                    ),
                ],
            ),
            payload=b'{"raw": "claude payload"}',
            source_path="/tmp/evidence-parity.jsonl",
            acquired_at_ms=1735689600000,
        )


def _run_read_evidence_json(session_id: str, view: str) -> tuple[dict[str, object], str]:
    from polylogue.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "--verbose",
            "read",
            f"session:{session_id}",
            "--view",
            view,
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    return dict(json.loads(result.stdout)), result.stderr


def test_read_file_edits_json_is_served_by_the_daemon(
    golden_parity_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-r3cuz: ``read --view file-edits`` is served by a reachable daemon.

    ``file-edits`` was one of the ``IN_PROCESS_READ_VIEWS``: it opened the
    archive through the Python API facade in this process, so a healthy daemon
    paid for a local archive open on every invocation. The declared
    ``session.read`` evidence kind now runs through the daemon route, and the
    evidence body is the whole payload here.

    Anti-vacuity: the ``file_edits`` row assertions below -- two empty bodies
    would agree trivially -- plus the ``served-by`` assertion, which a route
    that silently answered through a local executor fails.
    """

    archive_root = golden_parity_workspace["archive_root"]
    session_id = "claude-code-session:evidence-parity"

    with running_daemon_operations(archive_root, seed_archive=_seed_evidence_archive) as stack:
        _pin_cli_daemon_socket(monkeypatch, stack)
        daemon_payload, daemon_err = _run_read_evidence_json(session_id, "file-edits")

    assert "served-by: daemon (uds," in daemon_err, daemon_err
    assert daemon_payload["total"] == 1, daemon_payload
    edit = cast("list[dict[str, object]]", daemon_payload["file_edits"])[0]
    assert edit["file_path"] == "/tmp/parity.py"
    assert edit["original_file"] == "old contents\n"
    assert edit["structured_patch"] == [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}]


def test_read_agent_policies_and_web_content_are_served_by_the_daemon(
    golden_parity_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two sibling evidence kinds reach the daemon on the same route.

    These relations are empty for this fixture session, which is exactly why
    the assertion is the *executor*, not the rows: an empty body served in
    process and an empty body served by the daemon are indistinguishable by
    content, and the thing that moved is which one answered.  The row-bearing
    parity for this route is the ``file-edits`` test above.
    """

    archive_root = golden_parity_workspace["archive_root"]
    session_id = "claude-code-session:evidence-parity"

    with running_daemon_operations(archive_root, seed_archive=_seed_evidence_archive) as stack:
        _pin_cli_daemon_socket(monkeypatch, stack)
        for view, rows_key in (("agent-policies", "agent_policies"), ("web-content", "web_content_constructs")):
            daemon_payload, daemon_err = _run_read_evidence_json(session_id, view)
            assert "served-by: daemon (uds," in daemon_err, (view, daemon_err)
            assert rows_key in daemon_payload, (view, sorted(daemon_payload))


def test_facets_json_is_served_by_the_daemon(
    golden_parity_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The facets surface has its own daemon fast path (`_fetch_daemon_facets`)."""
    archive_root = golden_parity_workspace["archive_root"]
    from polylogue.cli import cli

    runner = CliRunner()

    with running_daemon_operations(archive_root, seed_archive=_seed_golden_archive) as stack:
        _pin_cli_daemon_socket(monkeypatch, stack)
        daemon_result = runner.invoke(cli, ["--plain", "facets", "--format", "json"])
        assert daemon_result.exit_code == 0, daemon_result.output
        daemon_payload = json.loads(daemon_result.output)

    # `generated_at` is a genuine wall-clock timestamp stamped independently
    # by each call, not a parity signal. `elapsed_s` (both top-level and
    # nested under `availability` -- FacetsResponse duplicates it,
    # polylogue/api/archive.py) is likewise a real per-call wall-clock
    # measurement of how long the facets projection took, not a parity
    # signal -- it necessarily differs between two independent invocations.
    for payload in (daemon_payload,):
        payload.pop("generated_at", None)
        payload.pop("elapsed_s", None)
        availability = payload.get("availability")
        if isinstance(availability, dict):
            availability.pop("elapsed_s", None)
    # The daemon serves the canonical FacetsResponse envelope (global/scoped
    # families), rather than the retired direct-route ``facets`` alias.
    assert daemon_payload.get("global") or daemon_payload.get("scoped")


def test_find_then_read_transcript_survives_daemon_proxied_keyword_search(
    golden_parity_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for polylogue-ajmu.

    ``find QUERY then read --view transcript`` (default plain-text rendering,
    the "summary"/"transcript" views' shared query-set renderer) used to crash
    with ``KeyError: 'rank'`` once a daemon was reachable and the query was a
    keyword FTS search rather than an exact session ref -- even with exactly
    one matching session, no disambiguation involved. Root cause: the daemon's
    ``_archive_search_hit_payload`` (``daemon/http.py``) put ``rank`` as a
    top-level sibling of ``session``/``match``, while the shared text renderer
    ``_hit_line`` (``cli/archive_query.py``) -- and every other search-hit
    producer (direct CLI ``_hit_payload``, MCP ``archive_search_hit_payload``)
    -- reads ``match["rank"]``.
    """
    archive_root = golden_parity_workspace["archive_root"]
    from polylogue.cli import cli

    with running_daemon_operations(archive_root, seed_archive=_seed_golden_archive) as stack:
        _pin_cli_daemon_socket(monkeypatch, stack)
        runner = CliRunner()
        # "exceptions" only appears in conv1's seeded message text, so this is
        # a single-hit keyword search -- the exact shape the bead reported.
        result = runner.invoke(cli, ["--plain", "find", "exceptions", "then", "read", "--view", "transcript"])

    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "Python Error Handling" in result.output
