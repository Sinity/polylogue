"""Golden parity: direct CLI execution vs config-matched daemon-proxied execution.

polylogue-20d.1 acceptance criterion: "`--format json` output is byte-identical
between direct and daemon-proxied execution for every read surface on the demo
corpus." A real production UDS daemon server is started against the same
seeded archive the direct path reads, and the same `find` invocation is run
through :class:`click.testing.CliRunner` twice — once with no daemon socket
present (direct path) and once with the daemon reachable (proxied path) — so
this is an end-to-end regression test, not a mock of the daemon transport.

The two envelopes are compared field-for-field rather than as raw text: the
daemon envelope carries an explicit ``"source": "daemon"`` provenance marker
that the direct envelope does not (`archive_query.py::_emit_daemon_list_payload`
vs `_emit_list`) — that is the one intentional, documented difference. Every
other field (`items`, `total`, `limit`, `offset`, `origin`, `next_offset`,
`next_cursor`) must match exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.infra.daemon_operations import DaemonOperationStack, running_daemon_operations
from tests.infra.storage_records import SessionBuilder

pytestmark = pytest.mark.uses_real_clock(
    "polylogue-20d.1 golden-parity test starts the maintained production daemon operation stack; frozen_clock cannot substitute for its real writer/listener lifecycle."
)


@pytest.fixture
def golden_parity_workspace(cli_workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A real seeded archive, reused for both the direct and daemon-proxied runs."""

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


def _run_find_json(args: list[str], *, no_daemon: bool = False) -> dict[str, object]:
    from polylogue.cli import cli

    runner = CliRunner()
    # `--no-daemon` and `--repo` are root options (`click_app.py::cli`), not
    # `find` verb options — they must precede `find` in argv. Passing `--repo`
    # here as the root option (rather than a `repo:polylogue` DSL query token,
    # which routes through a different, older rendering path with a distinct
    # envelope shape — see the module docstring follow-up note) is what
    # actually exercises `_try_emit_daemon_session_page` /
    # `_daemon_session_page_supported`, the code this test targets.
    root_flags = ["--plain", *args, *(["--no-daemon"] if no_daemon else [])]
    result = runner.invoke(cli, [*root_flags, "find", "--format", "json", "--limit", "10"])
    assert result.exit_code == 0, result.output
    return dict(json.loads(result.output))


def _strip_provenance(envelope: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in envelope.items() if key != "source"}


def test_find_list_json_parity_between_direct_and_daemon(
    golden_parity_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = golden_parity_workspace["archive_root"]
    args = ["--repo", "polylogue"]

    with running_daemon_operations(archive_root, seed_archive=_seed_golden_archive) as stack:
        _pin_cli_daemon_socket(monkeypatch, stack)
        direct_payload = _run_find_json(args, no_daemon=True)
        assert "source" not in direct_payload
        daemon_payload = _run_find_json(args)

    assert daemon_payload["source"] == "daemon"
    assert _strip_provenance(daemon_payload) == _strip_provenance(direct_payload)
    assert direct_payload["items"], "fixture query must actually match rows, or parity is vacuous"


def test_find_daemon_proxied_path_authenticates_with_auto_minted_token(
    golden_parity_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """polylogue-n6pz: the daemon auto-mints and requires a bearer token by
    default (polylogue-rzve) when no ``daemon.api.auth_token`` is explicitly
    configured. The CLI fast path must resolve that same auto-minted token
    (``polylogue.daemon.api_auth.resolve_api_auth_token``) rather than only
    reading the unset config value -- otherwise every unauthenticated probe
    gets a 401, ``DaemonClient.probe`` returns ``None``, and the CLI silently
    falls back to the direct path even though a real daemon is reachable.
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


def test_facets_json_parity_between_direct_and_daemon(
    golden_parity_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The facets surface has its own daemon fast path (`_fetch_daemon_facets`)."""
    archive_root = golden_parity_workspace["archive_root"]
    from polylogue.cli import cli

    runner = CliRunner()

    with running_daemon_operations(archive_root, seed_archive=_seed_golden_archive) as stack:
        _pin_cli_daemon_socket(monkeypatch, stack)
        direct_result = runner.invoke(cli, ["--plain", "--no-daemon", "facets", "--format", "json"])
        assert direct_result.exit_code == 0, direct_result.output
        direct_payload = json.loads(direct_result.output)
        daemon_result = runner.invoke(cli, ["--plain", "facets", "--format", "json"])
        assert daemon_result.exit_code == 0, daemon_result.output
        daemon_payload = json.loads(daemon_result.output)

    # `generated_at` is a genuine wall-clock timestamp stamped independently
    # by each call, not a parity signal. `elapsed_s` (both top-level and
    # nested under `availability` -- FacetsResponse duplicates it,
    # polylogue/api/archive.py) is likewise a real per-call wall-clock
    # measurement of how long the facets projection took, not a parity
    # signal -- it necessarily differs between two independent invocations.
    for payload in (direct_payload, daemon_payload):
        payload.pop("generated_at", None)
        payload.pop("elapsed_s", None)
        availability = payload.get("availability")
        if isinstance(availability, dict):
            availability.pop("elapsed_s", None)
    assert daemon_payload == direct_payload


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
