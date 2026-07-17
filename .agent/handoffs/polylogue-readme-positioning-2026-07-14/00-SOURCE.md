# Source

External design study conducted in ChatGPT (o-series Pro, "Great GitHub READMEs" session,
2026-07-13, chatgpt.com/c/6a54dd7c-756c-83eb-88b6-66cc8f61f0d4), NOT authoritative — verify
every claim against live source before acting, per repo convention. Two turns:

1. Research pass on great open-source READMEs (10 example repos + synthesis: 4-question
   viewport contract, one dominant proof/CTA not five, show outputs not just inputs).
2. User uploaded full source tarballs of both Sinex and Polylogue (`sinex-all.tar(82).gz`,
   `polylogue-all.tar(98).gz`) and asked for an open-ended, no-time-limit "library to compose
   from" rather than a single proposal. ChatGPT worked 86m21s and produced ~30 components,
   7 README compositions, coding-agent packets, a reader-comprehension test harness, patches,
   and validation reports for both repos.

Polylogue-relevant conclusion: **receipts-first** ("Know what the agents actually did") is the
recommended repository front door, built around the deterministic `polylogue demo receipts
--compact` proof — VERIFIED live 2026-07-14, this command already exists and produces exactly
the fixture ChatGPT described (claim-time pytest exit=1, later exit=0 repair, anti-grep control
with 2 prose "error" hits / 0 structural failures). This converges independently with Fable's
2026-07-10 legibility-kit (`.agent/scratch/legibility-kit-2026-07-10/`,
`.agent/scratch/legibility-kit-v2-2026-07-10/`, which also has a `polylogue-demo-receipts/`
folder) — two independent passes landed on the same positioning.

Files here are candidate/draft material only, reconcile against current master before use.
Not copied: the Sinex-side material (out of scope for Polylogue beads), the HTML
playground/gallery prototypes, the 20 rendered images, or the 7 unified diffs (only the
text/markdown packets relevant to Polylogue beads were extracted).
