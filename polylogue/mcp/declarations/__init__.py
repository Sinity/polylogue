"""Executable MCP declaration models and inventory.

The package initializer stays dependency-light so renderers and static checks can
load the contract without importing the MCP runtime or archive services.
"""

from polylogue.mcp.declarations.models import (
    MCPContinuationContract,
    MCPDeprecationState,
    MCPHandlerBinding,
    MCPInputContract,
    MCPOutputContract,
    MCPPromptDeclaration,
    MCPResourceDeclaration,
    MCPResultSemantics,
    MCPRole,
    MCPToolDeclaration,
    MCPTransactionDeclaration,
    MCPVerb,
    PythonParityExpectation,
)
from polylogue.mcp.declarations.registry import (
    MCP_KERNEL_REGISTRY,
    MCP_TOOL_DECLARATION_BY_NAME,
    MCP_TOOL_DECLARATIONS,
    PRIVILEGED_ALGEBRA,
    TARGET_DEFAULT_READ_ALGEBRA,
    TARGET_PROMPTS,
    TARGET_RESOURCES,
    declaration_for_tool,
    declared_tool_names,
)

__all__ = [
    "MCPContinuationContract",
    "MCPDeprecationState",
    "MCPHandlerBinding",
    "MCPInputContract",
    "MCP_KERNEL_REGISTRY",
    "MCPOutputContract",
    "MCPPromptDeclaration",
    "MCPResourceDeclaration",
    "MCPResultSemantics",
    "MCPRole",
    "MCPToolDeclaration",
    "MCPTransactionDeclaration",
    "MCPVerb",
    "MCP_TOOL_DECLARATIONS",
    "MCP_TOOL_DECLARATION_BY_NAME",
    "PRIVILEGED_ALGEBRA",
    "PythonParityExpectation",
    "TARGET_DEFAULT_READ_ALGEBRA",
    "TARGET_PROMPTS",
    "TARGET_RESOURCES",
    "declaration_for_tool",
    "declared_tool_names",
]
