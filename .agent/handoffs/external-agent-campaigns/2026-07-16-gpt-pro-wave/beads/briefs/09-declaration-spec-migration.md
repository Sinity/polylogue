Title: "[beads 09] DeclarationSpec family migration"

Job ID: `beads-09`
Result ZIP: `beads-09-declaration-spec-migration-r01.zip`
Primary Bead: `polylogue-o21.3`; launch only after the DeclarationSpec kernel
is present in the attached snapshot.

## Mission

Migrate a coherent declaration family onto the landed DeclarationSpec
authority, preserving its public/generated/validation semantics and removing
the old duplicated registry in the same patch. Trace every writer and consumer
before choosing the family boundary. Exercise real generation/discovery and
runtime consumers so the new descriptor cannot authorize itself through a
self-referential test. Report remaining families, shared hotspots, ordering,
and any declaration whose semantics do not fit the kernel rather than forcing
it into a false abstraction.
