"""Every migrated CLI mutation is an adapter over the operation contract.

The command families migrated onto the canonical daemon operation contract are
the root query's ``delete`` and its ``--add-tag``/``--set`` mutations. These
laws bind that contract: the destructive path passes the operation's preview
and authorization choke point, an interrupted mutation reports a cancelled
outcome with a non-zero exit, and no adapter reintroduces a polling loop.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest

from polylogue.cli.archive_query import _emit_delete, _emit_user_mutations
from polylogue.cli.operation_kernel import (
    OperationCancelledError,
    OperationKernel,
    OperationRequest,
    OperationUnavailableError,
)
from polylogue.operations.daemon_protocol import (
    DAEMON_OPERATION_SPECS,
    MUTATION_OPERATION_NAMES,
    DaemonAuthority,
    DaemonFallback,
    daemon_operation_spec,
)


def _env(*, plain: bool = True) -> MagicMock:
    env = MagicMock()
    env.ui.plain = plain
    env.ui.confirm = MagicMock()
    return env


class TestDeclaredMutationAuthority:
    """A declared mutation can only ever be served by the daemon."""

    @pytest.mark.parametrize("operation", sorted(MUTATION_OPERATION_NAMES))
    def test_mutation_operations_declare_no_direct_fallback(self, operation: str) -> None:
        """Anti-vacuity: flipping a spec to ``DIRECT_READ`` turns this red."""
        spec = daemon_operation_spec(operation)
        assert spec is not None
        assert spec.authority is not DaemonAuthority.READ
        assert spec.fallback is DaemonFallback.NEVER
        assert spec.direct_allowed is False

    def test_mutation_names_cover_every_non_read_declaration(self) -> None:
        declared = {spec.name for spec in DAEMON_OPERATION_SPECS if spec.authority is not DaemonAuthority.READ}
        assert declared == MUTATION_OPERATION_NAMES

    @pytest.mark.parametrize("operation", sorted(MUTATION_OPERATION_NAMES))
    def test_absent_daemon_refuses_instead_of_executing_locally(self, operation: str) -> None:
        """A missing daemon must not reach any local executor."""
        executed = False

        def direct(_request: OperationRequest) -> object:
            nonlocal executed
            executed = True
            return {"written": True}

        with pytest.raises(OperationUnavailableError):
            OperationKernel(lambda _request: None).execute(OperationRequest(operation, {}))
        assert executed is False


class TestDeleteChokePoint:
    """The destructive path passes preview and authorization, or it refuses."""

    def test_forced_delete_passes_preview_authorize_execute_in_order(self, capsys: pytest.CaptureFixture[str]) -> None:
        issued: list[str] = []

        def _served(_config: object, operation: str, _payload: dict[str, object]) -> dict[str, object]:
            issued.append(operation)
            if operation.endswith(".preview"):
                return {"status": "prepared", "preview_ref": "preview:1", "session_ids": ["s1"]}
            if operation.endswith(".authorize"):
                return {"status": "authorized", "authorization_token": "token-1"}
            return {"status": "deleted", "affected_count": 1}

        with patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served):
            _emit_delete(_env(), ("s1",), params={"force": True, "dry_run": False})

        assert issued == [
            "mutation.session.delete.preview",
            "mutation.session.delete.authorize",
            "mutation.session.delete.execute",
        ]
        assert json.loads(capsys.readouterr().out)["status"] == "deleted"

    def test_execute_without_an_authorization_token_never_runs(self) -> None:
        """A preview the daemon did not authorize cannot reach the execute step."""
        issued: list[str] = []

        def _served(_config: object, operation: str, _payload: dict[str, object]) -> dict[str, object]:
            issued.append(operation)
            if operation.endswith(".preview"):
                return {"status": "prepared", "preview_ref": "preview:1", "session_ids": ["s1"]}
            return {"status": "authorized"}

        with (
            patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served),
            pytest.raises(click.ClickException, match="invalid delete authorization"),
        ):
            _emit_delete(_env(), ("s1",), params={"force": True, "dry_run": False})

        assert "mutation.session.delete.execute" not in issued

    def test_authorization_token_count_must_match_the_previewed_chunks(self) -> None:
        """One token cannot stand in for two previewed chunks."""
        issued: list[str] = []

        def _served(_config: object, operation: str, _payload: dict[str, object]) -> dict[str, object]:
            issued.append(operation)
            if operation.endswith(".preview"):
                return {
                    "status": "prepared",
                    "preview_refs": ["preview:1", "preview:2"],
                    "session_ids": ["s1", "s2"],
                }
            return {"status": "authorized", "authorization_token": "token-1"}

        with (
            patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served),
            pytest.raises(click.ClickException, match="invalid delete authorization"),
        ):
            _emit_delete(_env(), ("s1", "s2"), params={"force": True, "dry_run": False})

        assert "mutation.session.delete.execute" not in issued


class TestCancellation:
    """An interrupted mutation reports cancelled, never a half-rendered success."""

    def test_interrupted_confirmation_cancels_the_preview_and_exits_non_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        issued: list[str] = []
        env = _env(plain=False)
        env.ui.confirm.side_effect = KeyboardInterrupt

        def _served(_config: object, operation: str, _payload: dict[str, object]) -> dict[str, object]:
            issued.append(operation)
            if operation.endswith(".preview"):
                return {"status": "prepared", "preview_ref": "preview:1", "session_ids": ["s1"]}
            return {"status": "cancelled", "preview_ref": "preview:1"}

        with (
            patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served),
            pytest.raises(click.exceptions.Exit) as exit_info,
        ):
            _emit_delete(env, ("s1",), params={"force": False, "dry_run": False})

        assert exit_info.value.exit_code != 0
        assert issued == ["mutation.session.delete.preview", "mutation.session.delete.cancel"]
        assert json.loads(capsys.readouterr().out)["status"] == "aborted"

    def test_interrupt_before_authorization_cancels_and_exits_non_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        issued: list[str] = []
        env = _env(plain=False)
        env.ui.confirm.return_value = True

        def _served(_config: object, operation: str, _payload: dict[str, object]) -> dict[str, object]:
            issued.append(operation)
            if operation.endswith(".preview"):
                return {"status": "prepared", "preview_ref": "preview:1", "session_ids": ["s1"]}
            if operation.endswith(".authorize"):
                raise KeyboardInterrupt
            return {"status": "cancelled", "preview_ref": "preview:1"}

        with (
            patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served),
            pytest.raises(click.exceptions.Exit) as exit_info,
        ):
            _emit_delete(env, ("s1",), params={"force": False, "dry_run": False})

        assert exit_info.value.exit_code != 0
        assert issued[-1] == "mutation.session.delete.cancel"
        assert json.loads(capsys.readouterr().out)["status"] == "aborted"

    def test_kernel_reports_a_cancelled_envelope_as_a_typed_cancellation(self) -> None:
        """An interrupted daemon-side operation is cancelled, not failed."""
        with pytest.raises(OperationCancelledError):
            OperationKernel(
                lambda _request: {"outcome": "interrupted", "result": None},
            ).execute(OperationRequest("mutation.session.delete.execute", {}))


class TestNoPolling:
    """A migrated adapter waits on the operation, never on a sleep loop."""

    _ADAPTERS = (
        Path("polylogue/cli/archive_query.py"),
        Path("polylogue/cli/operation_kernel.py"),
        Path("polylogue/daemon_client.py"),
    )

    @pytest.mark.parametrize("module", _ADAPTERS, ids=lambda path: path.name)
    def test_adapter_has_no_sleep_or_retry_loop(self, module: Path) -> None:
        """Anti-vacuity: adding ``time.sleep`` to any adapter turns this red."""
        tree = ast.parse(module.read_text(encoding="utf-8"))
        sleeps = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sleep"
        ]
        assert sleeps == [], f"{module} polls instead of waiting on the operation"

    def test_no_adapter_names_a_poll_interval(self) -> None:
        for module in self._ADAPTERS:
            source = module.read_text(encoding="utf-8")
            assert "poll_after_ms" not in source, f"{module} reintroduced a polling hint"


class TestUserMutationRefusal:
    """The matched-page tag/metadata route has no offline write path."""

    def test_absent_daemon_refuses_the_tag_write(self, tmp_path: Path) -> None:
        archive = MagicMock()
        env = _env()
        with (
            patch(
                "polylogue.cli.archive_query._submit_mutation_operation",
                side_effect=OperationUnavailableError("daemon is unavailable"),
            ),
            pytest.raises(click.ClickException, match="daemon is unavailable"),
        ):
            _emit_user_mutations(
                env,
                archive,
                ("s1",),
                tags_to_add=("triage",),
                metadata_to_set=(),
            )
        archive.add_user_tags.assert_not_called()
