"""``session.read`` answers the same thing over every route that serves it.

The design's S8 gap is that a read view could be migrated onto an operation
whose daemon and direct executors disagree, or whose answer differs from the
API's, and nothing would notice.  These tests run the same read over all three
routes on one seeded archive and compare the bodies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.api import Polylogue
from polylogue.archive.query.transaction import QueryContinuationInvalidError
from polylogue.config import Config
from polylogue.core.enums import Origin, Provider
from polylogue.operations.daemon_reads import execute_read_operation
from polylogue.operations.operation_context import open_operation_read
from polylogue.operations.session_contracts import SessionRead
from polylogue.operations.session_reads import execute_session_operation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent
from tests.infra.daemon_operations import running_daemon_operations
from tests.infra.storage_records import SessionBuilder

pytestmark = pytest.mark.uses_real_clock(
    "starts the real UDS listener and coordinator loop to compare the daemon route against the direct one"
)

_NATIVE_ID = "ext-parity-0"
_EVENT_TYPES = ("PreToolUse", "PostToolUse", "PostToolUse", "UserPromptSubmit")


def _seed(root: Path) -> str:
    builder = (
        SessionBuilder(root / "index.db", "parity-0")
        .provider("codex")
        .title("Parity session")
        .add_message(text="A seeded message for the parity corpus.")
    )
    builder.save()
    with ArchiveStore(root) as archive:
        for index, event_type in enumerate(_EVENT_TYPES):
            archive.write_hook_event(
                provider=Provider.CODEX,
                payload=b'{"event":"' + event_type.encode() + b'"}',
                source_path=f"/hooks/parity-{index}.json",
                acquired_at_ms=2000 + index,
                hook_event=ArchiveHookEvent(
                    hook_event_id=f"hook:parity:{index}",
                    origin=Origin.CODEX_SESSION,
                    source_path=f"/hooks/parity-{index}.json",
                    event_type=event_type,
                    payload={"event": event_type},
                    observed_at_ms=2000 + index,
                    native_id=f"parity-native-{index}",
                    session_native_id=_NATIVE_ID,
                ),
                carrier_source_id="primary",
                carrier_relative_path=f"parity-{index}.json",
            )
    return builder.native_session_id()


def test_hooks_evidence_agrees_across_daemon_direct_and_the_api(tmp_path: Path) -> None:
    """Mutation: answer the daemon route from a different reader than the direct
    one -- or let either drift from the facade -- and these bodies diverge.

    The three routes are genuinely distinct: the daemon body crosses the UDS
    transport and its envelope validation, the direct body is produced by the
    in-process handler over a pinned reader, and the API body is the facade
    route that MCP and the Python API already use.
    """

    session_id: str = ""

    def seed(root: Path) -> None:
        nonlocal session_id
        session_id = _seed(root)

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        envelope = stack.client.operation(
            "session.read",
            {"ref": session_id, "kind": "hooks"},
            archive_root=str(stack.archive_root),
        )
        assert envelope is not None
        daemon_body = envelope["result"]
        archive_root = Path(stack.archive_root)

        with open_operation_read(archive_root) as pinned:
            direct_body = execute_read_operation(
                "session.read",
                {"ref": session_id, "kind": "hooks"},
                archive=pinned.archive,
                serving_identity="direct",
            )

        async def _facade() -> dict[str, object] | None:
            config = Config(
                archive_root=archive_root,
                render_root=archive_root / "render",
                sources=[],
                db_path=archive_root / "index.db",
            )
            async with Polylogue.open(config=config) as api:
                return await api.get_hook_event_summary_for_session(session_id)

        api_body = asyncio.run(_facade())

    assert daemon_body["evidence"] == direct_body["evidence"]
    assert daemon_body["evidence"] == api_body
    assert daemon_body["session_id"] == direct_body["session_id"] == session_id

    evidence = daemon_body["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["total"] == len(_EVENT_TYPES)
    assert evidence["by_event_type"] == {"PostToolUse": 2, "PreToolUse": 1, "UserPromptSubmit": 1}


def test_transcript_window_agrees_across_daemon_and_direct(tmp_path: Path) -> None:
    """The kind discriminator did not change what a transcript window returns.

    Mutation: route the default kind through the evidence branch and the
    transcript window loses its messages on one route but not the other.
    """

    session_id: str = ""

    def seed(root: Path) -> None:
        nonlocal session_id
        builder = SessionBuilder(root / "index.db", "parity-window").provider("codex").title("Window parity")
        for index in range(5):
            builder = builder.add_message(text=f"Transcript row {index}.")
        builder.save()
        session_id = builder.native_session_id()

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        envelope = stack.client.operation(
            "session.read",
            {"ref": session_id, "limit": 1},
            archive_root=str(stack.archive_root),
        )
        assert envelope is not None
        daemon_body = envelope["result"]

        with open_operation_read(Path(stack.archive_root)) as pinned:
            direct_body = execute_read_operation(
                "session.read",
                {"ref": session_id, "limit": 1},
                archive=pinned.archive,
                serving_identity="direct",
            )
            direct_next = cast(
                "dict[str, Any]",
                execute_read_operation(
                    "session.read",
                    {"ref": session_id, "continuation": daemon_body["continuation"]},
                    archive=pinned.archive,
                    serving_identity="direct",
                ),
            )
        daemon_next_envelope = stack.client.operation(
            "session.read",
            {"ref": session_id, "continuation": direct_body["continuation"]},
            archive_root=str(stack.archive_root),
        )
        assert daemon_next_envelope is not None
        daemon_next = daemon_next_envelope["result"]

    assert daemon_body["session"] == direct_body["session"]
    assert daemon_body["total"] == direct_body["total"]
    assert daemon_body.get("evidence") is None
    assert direct_body.get("evidence") is None
    assert direct_next["offset"] == daemon_next["offset"] == 1
    assert direct_next["session"]["messages"] == daemon_next["session"]["messages"]


def test_session_owner_continuation_refusal_names_its_dialect(tmp_path: Path) -> None:
    """A token from the owner executor cannot silently select the legacy envelope."""
    session_id: str = ""

    def seed(root: Path) -> None:
        nonlocal session_id
        builder = SessionBuilder(root / "index.db", "dialect-parity").provider("codex").title("Dialect parity")
        for index in range(3):
            builder = builder.add_message(text=f"Transcript row {index}.")
        builder.save()
        session_id = builder.native_session_id()

    with running_daemon_operations(tmp_path / "archive", seed_archive=seed) as stack:
        archive_root = Path(stack.archive_root)
        config = Config(
            archive_root=archive_root,
            render_root=archive_root / "render",
            sources=[],
            db_path=archive_root / "index.db",
        )

        async def owner_page() -> Any:
            async with Polylogue.open(config=config) as api:
                return await execute_session_operation(
                    api,
                    SessionRead(ref=f"session:{session_id}", limit=1),
                )

        owner_result = asyncio.run(owner_page())
        owner_token = owner_result.continuation
        assert owner_token

        with open_operation_read(archive_root) as pinned:
            with pytest.raises(
                QueryContinuationInvalidError,
                match=r"continuation dialect mismatch: expected session.read/session-read-v1, got sessions.read/session-owner-v1",
            ):
                execute_read_operation(
                    "session.read",
                    {"ref": f"session:{session_id}", "continuation": owner_token},
                    archive=pinned.archive,
                    serving_identity="direct",
                )
