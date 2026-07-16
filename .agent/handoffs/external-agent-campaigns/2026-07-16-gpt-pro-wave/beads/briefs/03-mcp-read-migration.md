Title: "[beads 03] MCP read-surface algebra migration"

Job ID: `beads-03`
Result ZIP: `beads-03-mcp-read-migration-r01.zip`
Dependency: the accepted algebra/equivalence kernel from `beads-02`.

## Mission

Migrate the disjoint MCP read/search/insight/topology/timeline tool families
onto the accepted shared algebra. Preserve exact discovery names, schemas,
role gates, ObjectRef semantics, query behavior, pagination, and error/absence
contracts. Remove superseded per-tool glue in the same patch rather than
keeping two authorities. Update the generated Python parity matrix and MCP
tool/contract fixtures required by current source. Prove representative tools
through the real MCP registration and operations route, not by testing the
descriptor in isolation.
