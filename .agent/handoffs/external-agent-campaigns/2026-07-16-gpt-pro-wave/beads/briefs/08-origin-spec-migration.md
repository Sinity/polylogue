Title: "[beads 08] OriginSpec source-family migration"

Job ID: `beads-08`
Result ZIP: `beads-08-origin-spec-migration-r01.zip`
Primary Bead: `polylogue-2qx.1.2`; launch only after the OriginSpec kernel is
present in the attached snapshot.

## Mission

Migrate one cohesive set of source families onto the landed OriginSpec kernel,
partitioned by actual detector/parser/fixture write footprints. Preserve
non-injective provider-to-origin behavior, strictness/tightness ordering,
parser fingerprint, raw and normalized fixture authority, fidelity notes, and
deterministic ambiguity. Remove migrated parallel declarations and update real
dispatch/validation consumers. If all families overlap through one hotspot,
deliver one carefully ordered patch rather than pretending the work is
parallel; otherwise identify clean follow-up family slices.
