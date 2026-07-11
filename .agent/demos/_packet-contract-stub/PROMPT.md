# Packet Contract Stub

Predeclaration receipt: `artifact:packet-contract-stub-predeclaration`.

This is a deliberately minimal, hand-authored fixture packet proving the
Demo Finding Packet contract (polylogue-212.7) end to end — it is NOT one of
the real 212 demos (D1/D2/D4/D5/D8/post-hoc-Q&A/anti-demo/foreman-rhetoric).
Those still need their own implementation (each is its own bead); this stub
exists only so `devtools lab policy demo-packet-registry` has one real,
conforming registry entry to validate, and so a future contributor building
a real demo has a concrete worked example of every required file.

## What a real prompt would say here

A real demo's PROMPT.md instructs a coding agent to run specific `polylogue`
commands (product primitives — the 212 compositionality rule: shell/python
is glue only) against the seeded demo corpus (`polylogue demo seed`, seed
1843), then package the results into this same packet shape.

This stub's "claim" is trivial by design: count sessions in the seeded
corpus. Run: `polylogue --format json find` against the seeded corpus and
report the total.
