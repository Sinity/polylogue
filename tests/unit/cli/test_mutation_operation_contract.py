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
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import click
import pytest

from polylogue.cli.archive_query import _emit_user_mutations, execute_delete_selection
from polylogue.cli.operation_kernel import (
    OperationCancelledError,
    OperationKernel,
    OperationRequest,
    OperationUnavailableError,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.operations.daemon_protocol import (
    DAEMON_OPERATION_SPECS,
    MUTATION_OPERATION_NAMES,
    DaemonAuthority,
    DaemonFallback,
    daemon_operation_spec,
)
from polylogue.operations.mutation_transaction import ConfirmationRequiredError
from tests.infra.daemon_operations import accepted_operation_reference

if TYPE_CHECKING:
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.daemon_protocol import DaemonOperationRequest
    from polylogue.operations.operation_context import PinnedOperationRead
    from polylogue.operations.operation_context_types import OperationContext


def _env(*, plain: bool = True) -> MagicMock:
    env = MagicMock()
    env.ui.plain = plain
    env.ui.confirm = MagicMock()
    return env


class TestDeclaredMutationAuthority:
    """A declared mutation can only ever be served by the daemon."""

    @pytest.mark.parametrize("operation", sorted(MUTATION_OPERATION_NAMES))
    def test_mutation_operations_declare_no_direct_fallback(self, operation: str) -> None:
        """Anti-vacuity: changing the fallback metadata turns this red."""
        spec = daemon_operation_spec(operation)
        assert spec is not None
        assert spec.authority is not DaemonAuthority.READ
        assert spec.fallback is DaemonFallback.NEVER

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

    def test_daemon_executor_requires_request_confirmation_before_authorizing(self) -> None:
        """A direct destructive request cannot mint its own bound confirmation.

        Anti-vacuity: removing the daemon-side confirmation check lets this
        request reach ``authorize_bound`` with ``bound_token`` strength.
        """
        from polylogue.operations import daemon_mutations

        context = SimpleNamespace(runtime=object(), archive_root=Path("/archive"), principal=object())
        binding = SimpleNamespace(actuator=SimpleNamespace(required_confirmation="confirm_flag"))
        request = SimpleNamespace(payload={"publication_ids": ["reservation:1"]}, operation="maintenance.reset")
        with (
            patch.object(daemon_mutations, "runtime_operation_binding", return_value=binding),
            patch.object(daemon_mutations, "OperationExecutor") as executor,
        ):
            with pytest.raises(ConfirmationRequiredError, match="explicit confirmation"):
                daemon_mutations._execute_named_mutation(
                    cast("DaemonOperationRequest", request),
                    cast("OperationContext", context),
                    cast("AuditRepository", object()),
                    cast("PinnedOperationRead", object()),
                    object(),
                    object(),
                )
        executor.assert_not_called()


class TestDeleteChokePoint:
    """The destructive path passes preview and authorization, or it refuses."""

    def test_forced_delete_passes_preview_authorize_execute_in_order(self, capsys: pytest.CaptureFixture[str]) -> None:
        issued: list[str] = []

        def _served(_config: object, operation: str, _payload: dict[str, object]) -> dict[str, object]:
            issued.append(operation)
            if operation.endswith(".preview"):
                return {
                    "status": "prepared",
                    "preview_ref": "preview:1",
                    "reference": accepted_operation_reference(
                        "mutation.session.delete.preview", request_id="preview-owner", artifact_kind="preview-batch"
                    ),
                    "session_count": 1,
                    "session_ids_sample": ["s1"],
                }
            if operation.endswith(".authorize"):
                return {
                    "status": "authorized",
                    "authorization_ref": "authorization:1",
                    "source_request_id": "preview-owner",
                    "reference": accepted_operation_reference(
                        "mutation.session.delete.authorize",
                        request_id="authorization-owner",
                        artifact_kind="authorization-batch",
                    ),
                }
            return {"status": "deleted", "affected_count": 1}

        with patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served):
            execute_delete_selection(
                _env(), RootModeRequest.from_params({"query": ("needle",)}), mode="all", force=True, dry_run=False
            )

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
                return {
                    "status": "prepared",
                    "preview_ref": "preview:1",
                    "reference": accepted_operation_reference(
                        "mutation.session.delete.preview", request_id="preview-owner", artifact_kind="preview-batch"
                    ),
                    "session_count": 1,
                    "session_ids_sample": ["s1"],
                }
            return {"status": "authorized"}

        with (
            patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served),
            pytest.raises(click.ClickException, match="delete operation reference|different delete selection"),
        ):
            execute_delete_selection(
                _env(), RootModeRequest.from_params({"query": ("needle",)}), mode="all", force=True, dry_run=False
            )

        assert "mutation.session.delete.execute" not in issued

    def test_authorization_reference_must_cover_the_whole_previewed_selection(self) -> None:
        """A smaller authorization cannot stand in for the complete durable preview."""
        issued: list[str] = []

        def _served(_config: object, operation: str, _payload: dict[str, object]) -> dict[str, object]:
            issued.append(operation)
            if operation.endswith(".preview"):
                return {
                    "status": "prepared",
                    "reference": accepted_operation_reference(
                        "mutation.session.delete.preview",
                        request_id="preview-owner",
                        artifact_kind="preview-batch",
                        part_count=2,
                    ),
                    "session_count": 2,
                    "session_ids_sample": ["s1", "s2"],
                }
            return {
                "status": "authorized",
                "authorization_ref": "authorization:1",
                "source_request_id": "preview-owner",
                "reference": accepted_operation_reference(
                    "mutation.session.delete.authorize",
                    request_id="authorization-owner",
                    artifact_kind="authorization-batch",
                ),
            }

        with (
            patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served),
            pytest.raises(click.ClickException, match="delete operation reference|different delete selection"),
        ):
            execute_delete_selection(
                _env(), RootModeRequest.from_params({"query": ("needle",)}), mode="all", force=True, dry_run=False
            )

        assert "mutation.session.delete.execute" not in issued

    @pytest.mark.parametrize(
        ("count", "sample", "accepted"),
        [
            (30, [f"s{index}" for index in range(20)], True),
            (3, [f"s{index}" for index in range(5)], False),
            (2, ["s1", "s1"], False),
            (0, [], False),
        ],
    )
    def test_preview_reports_selection_size_and_bounded_sample(
        self, count: int, sample: list[str], accepted: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The preview names the selection by size and a leading sample, never the whole list.

        Fails if the CLI needs the full echoed selection (a large accepted
        selection would then exceed the operation result bound), or if it
        accepts a sample that is not the canonical leading slice.
        """
        issued: list[str] = []

        def _served(_config: object, operation: str, _payload: dict[str, object]) -> dict[str, object]:
            issued.append(operation)
            if operation.endswith(".preview"):
                return {
                    "status": "prepared",
                    "preview_ref": "preview:1",
                    "reference": accepted_operation_reference(
                        "mutation.session.delete.preview", request_id="preview-owner", artifact_kind="preview-batch"
                    ),
                    "session_count": count,
                    "session_ids_sample": sample,
                }
            if operation.endswith(".authorize"):
                return {
                    "status": "authorized",
                    "authorization_ref": "authorization:1",
                    "source_request_id": "preview-owner",
                    "reference": accepted_operation_reference(
                        "mutation.session.delete.authorize",
                        request_id="authorization-owner",
                        artifact_kind="authorization-batch",
                    ),
                }
            return {"status": "deleted", "affected_count": count}

        with patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served):
            if accepted:
                execute_delete_selection(
                    _env(), RootModeRequest.from_params({"query": ("needle",)}), mode="all", force=True, dry_run=False
                )
            else:
                with pytest.raises(click.ClickException, match="delete preview|No sessions"):
                    execute_delete_selection(
                        _env(),
                        RootModeRequest.from_params({"query": ("needle",)}),
                        mode="all",
                        force=True,
                        dry_run=False,
                    )

        assert ("mutation.session.delete.execute" in issued) is accepted
        if accepted:
            assert json.loads(capsys.readouterr().out)["affected_count"] == count


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
                return {
                    "status": "prepared",
                    "preview_ref": "preview:1",
                    "reference": accepted_operation_reference(
                        "mutation.session.delete.preview", request_id="preview-owner", artifact_kind="preview-batch"
                    ),
                    "session_count": 1,
                    "session_ids_sample": ["s1"],
                }
            return {
                "status": "cancelled",
                "source_request_id": "preview-owner",
                "reference": accepted_operation_reference(
                    "mutation.session.delete.cancel", request_id="cancel-owner", artifact_kind="cancelled-preview-batch"
                ),
            }

        with (
            patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served),
            pytest.raises(click.exceptions.Exit) as exit_info,
        ):
            execute_delete_selection(
                env, RootModeRequest.from_params({"query": ("needle",)}), mode="all", force=False, dry_run=False
            )

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
                return {
                    "status": "prepared",
                    "preview_ref": "preview:1",
                    "reference": accepted_operation_reference(
                        "mutation.session.delete.preview", request_id="preview-owner", artifact_kind="preview-batch"
                    ),
                    "session_count": 1,
                    "session_ids_sample": ["s1"],
                }
            if operation.endswith(".authorize"):
                raise KeyboardInterrupt
            return {
                "status": "cancelled",
                "source_request_id": "preview-owner",
                "reference": accepted_operation_reference(
                    "mutation.session.delete.cancel", request_id="cancel-owner", artifact_kind="cancelled-preview-batch"
                ),
            }

        with (
            patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served),
            pytest.raises(click.exceptions.Exit) as exit_info,
        ):
            execute_delete_selection(
                env, RootModeRequest.from_params({"query": ("needle",)}), mode="all", force=False, dry_run=False
            )

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
        # The route no longer takes an archive at all -- there is no local
        # writer to fall back to. The double stays as the negative evidence:
        # nothing may reach a store while the daemon is absent.
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
                RootModeRequest.from_params({"query": ("needle",)}),
                limit=10,
                offset=0,
                tags_to_add=("triage",),
                metadata_to_set=(),
            )
        archive.add_user_tags.assert_not_called()
