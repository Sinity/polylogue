# Wave 2 result custody

This directory is the immutable custody record for returned external-agent
artifacts.  A package being present here means its bytes, member layout, and
mission association were checked; it does **not** mean its patch was applied.

Each `rNN/receipt.json` records one package revision. `index.json` is a
rebuildable, human-readable reception projection. Duplicated browser downloads
with identical SHA-256 values are deliberately not copied twice. The latest
reception pass used current master `1e60ef0bb` for the explicitly recorded
apply checks.

The three Clipboard reports matching `mandate-02`, `support-b`, and `support-d`
were corroborated against their package `HANDOFF.md`; their source reports are
not duplicated because the package is the durable, checksum-addressed source.

## Adjudication vocabulary

- `snapshot_mismatch`: the patch did not apply to the snapshot it claimed.
- `needs_rebase_review`: the patch was coherent on its supplied baseline but
  conflicts with current master; it must not be replayed blindly.
- `ready_for_review`: the patch applies cleanly to current master, but still
  requires ordinary source review and tests before admission.
- `analysis_only`: durable design/research material, with no code admission.
- `incomplete_delivery`: a returned artifact that is structurally valid but
  cannot satisfy its mission.
- `superseded`: current master already contains the relevant completed slice.
