# Machine output and action contract

Purpose: every public command and action declares executable metadata beside its registration. This metadata drives help, completion, output tests, API parity, and safety guards.

Fields: path, effect, input kind, supported formats, default format, schema, cardinality, daemon requirement, guards, completion context.

Rules: finite machine mode emits one document. Streaming machine mode emits one object per line. Human progress and warnings use stderr.

Slices: add contract type and registry; attach contracts to retained commands and actions; generate inventory; convert output tests to read the registry; validate outputs; share envelope vocabulary across CLI, daemon, MCP, and Python surfaces.

Acceptance: every public command/action has a contract; tests fail when metadata is missing; machine stdout is pure; guarded actions declare behavior; old assurance code is not an independent truth source.