"""FastMCP registration adapter backed by the executable declaration inventory.

The adapter validates the compatibility surface while deliberately passing the
original handler object to FastMCP.  It is not an executor and does not acquire
runtime, authorization, storage, or workflow authority.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast

from polylogue.mcp.declarations.models import MCPRole, MCPToolDeclaration, mcp_role_allows
from polylogue.mcp.declarations.registry import declaration_for_tool, declared_tool_names

HandlerT = TypeVar("HandlerT", bound=Callable[..., Any])
AnyHandler = Callable[..., Any]


class ToolRegistrar(Protocol):
    """The FastMCP decorator shape used by current registration owners."""

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
    ) -> Callable[[AnyHandler], AnyHandler]: ...


class MCPRegistrationError(RuntimeError):
    """Actionable declaration/registration mismatch."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        repair_command: str = "devtools render mcp-equivalence",
    ) -> None:
        self.message = message
        self.tool_name = tool_name
        self.repair_command = repair_command
        super().__init__(f"{message}; repair with `{repair_command}`")


class DeclaredToolRegistrar:
    """Validate one role-filtered live tool surface against declarations.

    Attribute access other than :meth:`tool` is delegated to the wrapped
    FastMCP instance so existing registration helpers remain usable.  The
    decorator returned by :meth:`tool` receives and forwards the exact handler
    object; behavior, signatures, cancellation, and error handling therefore
    remain owned by the existing production implementation.
    """

    def __init__(self, delegate: ToolRegistrar, *, role: MCPRole) -> None:
        self._delegate = delegate
        self._role = role
        self._registered: dict[str, Callable[..., Any]] = {}

    @property
    def role(self) -> MCPRole:
        return self._role

    @property
    def registered_names(self) -> frozenset[str]:
        return frozenset(self._registered)

    def _declaration(self, name: str) -> MCPToolDeclaration:
        try:
            declaration = declaration_for_tool(name)
        except KeyError as exc:
            raise MCPRegistrationError(
                f"live MCP handler {name!r} has no executable declaration",
                tool_name=name,
            ) from exc
        if not mcp_role_allows(self._role, declaration.minimum_role):
            raise MCPRegistrationError(
                f"MCP role {self._role!r} registered {name!r}, which requires {declaration.minimum_role!r}",
                tool_name=name,
            )
        return declaration

    @staticmethod
    def _resolved_name(handler: Callable[..., Any], explicit_name: str | None) -> str:
        if explicit_name is not None:
            if not explicit_name:
                raise MCPRegistrationError("FastMCP tool name must be a non-empty string")
            return explicit_name
        name = getattr(handler, "__name__", None)
        if not isinstance(name, str) or not name:
            raise MCPRegistrationError("MCP handler has no stable __name__ for declaration lookup")
        return name

    @staticmethod
    def _validate_binding(declaration: MCPToolDeclaration, handler: Callable[..., Any]) -> None:
        module = getattr(handler, "__module__", None)
        if module != declaration.registration.module:
            raise MCPRegistrationError(
                f"MCP handler {declaration.name!r} is implemented by {module!r}, not declared module "
                f"{declaration.registration.module!r}",
                tool_name=declaration.name,
            )

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
    ) -> Callable[[AnyHandler], AnyHandler]:
        """Return the validating decorator used by every live tool registrar."""

        def register(handler: AnyHandler) -> AnyHandler:
            resolved_name = self._resolved_name(handler, name)
            declaration = self._declaration(resolved_name)
            self._validate_binding(declaration, handler)
            if description is not None and description != declaration.description:
                raise MCPRegistrationError(
                    f"MCP tool {resolved_name!r} discovery text differs from its declaration",
                    tool_name=resolved_name,
                )
            if resolved_name in self._registered:
                raise MCPRegistrationError(
                    f"MCP tool {resolved_name!r} was registered more than once",
                    tool_name=resolved_name,
                )
            registered = self._delegate.tool(name=name, description=description)(handler)
            self._registered[resolved_name] = handler
            return registered

        return register

    def finalize(self) -> None:
        """Require the exact role-visible declaration set after registration."""

        expected = declared_tool_names(self._role)
        actual = self.registered_names
        if expected == actual:
            return
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        parts: list[str] = [f"MCP {self._role!r} registration does not match executable declarations"]
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        raise MCPRegistrationError("; ".join(parts))

    def __getattr__(self, name: str) -> object:
        """Delegate non-registration behavior without creating a parallel server."""

        return getattr(self._delegate, name)


def register_declared_handler(
    mcp: ToolRegistrar,
    handler: HandlerT,
    *,
    name: str,
) -> object:
    """Register one compatibility handler using declaration-owned metadata."""

    declaration = declaration_for_tool(name)
    handler.__name__ = declaration.name
    handler.__doc__ = declaration.description
    return cast(HandlerT, mcp.tool()(handler))


__all__ = [
    "DeclaredToolRegistrar",
    "MCPRegistrationError",
    "ToolRegistrar",
    "register_declared_handler",
]
