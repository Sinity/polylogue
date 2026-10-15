# Contract snapshot

These are reviewed design documents copied in their original relative package layout for isolated foundation preparation. Current implementation support is listed in the top-level README. Cross-project source links not included in this snapshot remain upstream design references. This branch has not replaced the current specification authority or any deployed source.

The retained-WAL startup contract follows specification commit `360813f` (owning storage README/blobs and law 123). Canonical DDL retains its separately recorded basis in `crates/store/schema/provenance.json`; no SQL changed with this policy.

The import wait binds source-row settlement to the admitted immutable basis and
required canonical effects (owning CLI/law clarification `b508a1e`). Its typed
`source_rows` result does not settle Session/global/search coverage. Standalone
Session JSON carries its actual degraded result metadata under the export
qualification contract `a9947de`; current import/export roundtrip propagation
and complete ordinary clients remain unqualified. These prose changes do not
change the separately pinned index18 schema.
