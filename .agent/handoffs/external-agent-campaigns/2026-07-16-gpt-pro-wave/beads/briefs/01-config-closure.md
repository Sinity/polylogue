Title: "[beads 01] Configuration precedence closure"

Job ID: `beads-01`
Result ZIP: `beads-01-config-closure-r01.zip`
Primary Beads: `polylogue-9gh1`, `polylogue-fd2s`, `polylogue-cxlk`; treat
`polylogue-uu8r` as a separately partitioned migration unless current source
proves the write footprint is cohesive.

## Mission

Close the actual configuration precedence/merge slice across config files,
environment, explicit arguments, runtime inventory, archive tier roots, and
diagnostics. Read every later note and current call site. Implement one typed
authority and migrate the relevant consumers rather than adding another
normalization wrapper. Preserve provider/runtime-specific direct-environment
behavior only where it is a real boundary, and make conflicting/split roots
fail honestly. Include real-route tests through at least the config facade and
daemon/CLI consumer paths plus a precise residual matrix for `uu8r`.
