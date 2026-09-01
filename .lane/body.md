Summary

Enforce a complete, declaration-generated topology capability census and retain the production parser fix that removes fabricated Codex and Hermes message-parent links.

Problem

The census projection accepted a caller-supplied subset of OriginSpec declarations, allowing an omitted current Origin or unknown capability to appear complete. Codex and Hermes message records carry no message-parent evidence, so positional chaining fabricated topology.

Solution

The census now requires exactly one declaration for every current Origin, all five dimensions, a completed capability state, evidence, and a reason for structural absence. Mutation tests cover omitted origins and unknown states. Codex and Hermes retain explicit session-parent references while leaving message parents unset when the wire format has no such field. Their parser fingerprints now cover the affected origins: codex-session and hermes-session.

Verification

- `nix develop --accept-flake-config --command devtools test tests/unit/sources/test_origin_specs.py tests/unit/sources/test_parsers_codex.py tests/unit/sources/test_parsers_local_agent.py`: 163 passed.
- `nix develop --accept-flake-config --command devtools render all --check`: generated surfaces synchronized.
- `nix develop --accept-flake-config --command devtools verify --quick`: exit 0; static and repository gates passed.

Residual risk

The full corpus was not run. No durable schema migration is required. The parser fingerprint change affects codex-session and hermes-session.

LANE-BRANCH: feature/packet/polylogue-ksgg.3
LANE-COMMIT: 6d54e1a
LANE-QUICK: green
LANE-CLASSIFICATION: parser semantics changed for codex-session and hermes-session; no schema migration
