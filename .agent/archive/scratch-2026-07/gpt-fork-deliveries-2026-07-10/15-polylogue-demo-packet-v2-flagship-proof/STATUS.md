# Status — Polylogue Demo Packet v2 + Flagship Proof

**Context:** Branch of the shared Polylogue/Sinex overview root (see
`../00-shared-branch-root-polylogue-sinex-overview/`). Diverges attaching
`03-demo-packet-v2-and-flagships.md` as the mission spec (confirmed via tool output listing
`/mnt/data/03-demo-packet-v2-and-flagships.md`).

**Delivered:** `# Demo Packet v2 + flagship proof package` — "I treated the attached mission as
the newer execution authority, including its explicit requirement that th[e package be
independently verifiable]." Rebuilt Polylogue's demo-packet doctrine and flagship proof
artifacts against a clean git clone of the working tree. `inline-artifacts.md` captures one
complete recovered source file: `tests/unit/devtools/test_demo_packet.py` — real contract tests
for "Demo Packet v2 and the committed packet registry," printed inline via heredoc before the
final delivery.

**Recoverable vs LOST:** The delivery prose and the one complete test file are fully recovered
verbatim. LOST: the actual demo-packet-v2 patch/package itself (apply-patches.sh,
verify-package.sh, and the underlying implementation) — referenced only as sandbox paths, not
printed inline beyond the one test file and short script snippets that didn't meet the inclusion
bar.

**Regeneration value:** Medium — the recovered test file gives a concrete contract to
reimplement against; the rest of the patch would need to be re-derived.
