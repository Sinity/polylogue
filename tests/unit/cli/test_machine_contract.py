"""Property tests for the machine error and success envelope contract.

Validates the JSON envelope shapes and builder functions used by the CLI
when JSON machine output is requested — via Hypothesis property tests instead of
hand-enumerated cases.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from polylogue.cli.shared.machine_errors import (
    DEPENDENCY_MISSING,
    INVALID_ARGUMENTS,
    INVALID_PATH,
    RUNTIME_ERROR,
    UNSUPPORTED_ENVIRONMENT,
    MachineError,
    MachineSuccess,
    error_dependency_missing,
    error_invalid_arguments,
    error_invalid_path,
    error_runtime,
    error_unsupported_environment,
    extract_command,
    success,
    wants_json,
)

pytestmark = pytest.mark.machine_contract

ErrorBuilder = Callable[..., MachineError]
ErrorBuilderSpec = tuple[ErrorBuilder, str, str | None]


class TestMachineErrorEnvelope:
    """Property: MachineError.to_dict() always has status=error, code, message;
    optional command/details appear iff non-empty."""

    @given(
        code=st.text(min_size=1, max_size=50),
        message=st.text(min_size=1, max_size=200),
        command=st.one_of(
            st.just([]),
            st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=5),
        ),
        details=st.one_of(
            st.just({}),
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.text(max_size=50),
                max_size=3,
            ),
        ),
    )
    def test_machine_error_envelope_contract(
        self, code: str, message: str, command: list[str], details: dict[str, str]
    ) -> None:
        err = MachineError(code=code, message=message, command=command, details=details)
        result = err.to_dict()

        assert result["status"] == "error"
        assert result["code"] == code
        assert result["message"] == message

        if command:
            assert result["command"] == command
        else:
            assert "command" not in result

        if details:
            assert result["details"] == details
        else:
            assert "details" not in result


class TestMachineSuccessEnvelope:
    """Property: MachineSuccess.to_dict() always has status=ok and result matches input."""

    @given(
        result_data=st.one_of(
            st.just({}),
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(
                    st.text(max_size=50),
                    st.integers(-1000, 1000),
                    st.booleans(),
                    st.lists(
                        st.dictionaries(
                            st.text(min_size=1, max_size=10),
                            st.text(max_size=20),
                            max_size=2,
                        ),
                        max_size=3,
                    ),
                ),
                max_size=5,
            ),
        ),
    )
    def test_machine_success_envelope_contract(self, result_data: dict[str, object]) -> None:
        success_obj = MachineSuccess(result=result_data)
        result = success_obj.to_dict()

        assert result["status"] == "ok"
        assert result["result"] == result_data


# --- Error builder specs ---

# Each spec: (builder_fn, expected_code, optional_kwarg_name)
_ERROR_BUILDER_SPECS: list[ErrorBuilderSpec] = [
    (error_invalid_arguments, INVALID_ARGUMENTS, "option"),
    (error_invalid_path, INVALID_PATH, "path"),
    (error_runtime, RUNTIME_ERROR, "exception_type"),
    (error_dependency_missing, DEPENDENCY_MISSING, "dependency"),
    (error_unsupported_environment, UNSUPPORTED_ENVIRONMENT, None),
]


class TestErrorBuilders:
    """Property: every error builder produces the correct code and populates
    details only when the optional kwarg is provided."""

    @given(
        spec=st.sampled_from(_ERROR_BUILDER_SPECS),
        message=st.text(min_size=1, max_size=100),
        kwarg_value=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    )
    @example(
        spec=(error_unsupported_environment, UNSUPPORTED_ENVIRONMENT, None),
        message="Platform not supported",
        kwarg_value=None,
    )
    def test_error_builder_contract(self, spec: ErrorBuilderSpec, message: str, kwarg_value: str | None) -> None:
        builder_fn, expected_code, kwarg_name = spec

        kwargs: dict[str, str] = {}
        if kwarg_name is not None and kwarg_value is not None:
            kwargs[kwarg_name] = kwarg_value

        err = builder_fn(message, **kwargs)

        assert err.code == expected_code
        assert err.message == message

        if kwarg_name is not None and kwarg_value is not None:
            assert err.details == {kwarg_name: kwarg_value}
        else:
            assert err.details == {}


class TestSuccessBuilder:
    """Property: success() always returns MachineSuccess with result as dict."""

    @given(
        result_data=st.one_of(
            st.none(),
            st.just({}),
            st.dictionaries(
                st.text(min_size=1, max_size=20),
                st.one_of(st.text(max_size=50), st.integers(-100, 100), st.booleans()),
                max_size=5,
            ),
        ),
    )
    @example(result_data=None)
    def test_success_builder_contract(self, result_data: dict[str, object] | None) -> None:
        success_obj = success(result_data)
        assert isinstance(success_obj, MachineSuccess)
        assert isinstance(success_obj.result, dict)

        if result_data:
            assert success_obj.result == result_data
        else:
            assert success_obj.result == {}


class TestWantsJsonDetection:
    """Property: wants_json detects explicit JSON machine-output intent."""

    @given(argv=st.lists(st.text(max_size=30), max_size=10))
    @example(argv=[])
    @example(argv=["list", "--format", "json"])
    @example(argv=["--format=json", "list"])
    @example(argv=["-f", "json", "list"])
    def test_wants_json_only_matches_exact_flag(self, argv: list[str]) -> None:
        result = wants_json(argv)
        has_json_flag = False
        for index, arg in enumerate(argv):
            if arg == "--format" and index + 1 < len(argv) and argv[index + 1] == "json":
                has_json_flag = True
                break
            if arg.startswith("--format=") and arg.split("=", 1)[1] == "json":
                has_json_flag = True
                break
            if arg == "-f" and index + 1 < len(argv) and argv[index + 1] == "json":
                has_json_flag = True
                break
        assert result == has_json_flag


class TestExtractCommand:
    """Property: extract_command output is a subset of input in original order,
    containing no flags (strings starting with -)."""

    @given(argv=st.lists(st.text(max_size=30), max_size=10))
    @example(argv=[])
    def test_extract_command_contract(self, argv: list[str]) -> None:
        result = extract_command(argv)

        # Every element in the result must come from argv
        for item in result:
            assert item in argv

        # No element starts with -
        for item in result:
            assert not item.startswith("-"), f"Flag leaked: {item!r}"

        # Order is preserved (result is a subsequence of argv)
        argv_iter = iter(argv)
        for item in result:
            while True:
                try:
                    candidate = next(argv_iter)
                    if candidate == item:
                        break
                except StopIteration:
                    raise AssertionError(f"Order violation: {item!r} not in expected position") from None

    def test_extract_command_skips_option_values(self) -> None:
        argv = ["--format", "json", "--limit", "1", "list"]
        assert extract_command(argv) == ["list"]
