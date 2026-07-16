# Status — GitHub Publication Feasibility (sinnix / lynchpin / sinex)

**Asked:** Multi-part conversation. First: "I meant to attach these ~current bundles. Do look
into these now, in depth. I'm more interested not in whether these are currently ready to
publish, but rather what makes them not so and how to make it different" (sinnix + lynchpin
bundles). Then a follow-up requesting git history cleanup for both repos with preserved
author/committer identity ("do preserve useful author and committer dates, but otherwise
rewrite timestamps to make history clear without the noise... make commit messages rich,
semantic"). Then, separately: "review sinex publication feasibility, in depth, thoroughly,
think about many angles" (sinex-all.tar.gz). Finally: "review publication feasibility for all
of these [sinnix, lynchpin, sinex] ... Make the analysis standalone".

**Delivered:** (1) a v5 git-history-cleanup pass for sinnix (~300 commits) + lynchpin
(~1150 commits) preserving author/committer identity while scrubbing secrets/private
material and squashing noisy churn (turn 9, in `inline-artifacts.md`); (2) a standalone deep
Sinex-only publication feasibility review (turn 12, in `inline-artifacts.md`); (3) the final
consolidated cross-repo review (turn 20, `delivery.md`) recommending: publish all three, but
differently — Sinnix as public dotfiles after light cleanup (highest feasibility), Lynchpin as
a clean repo with synthetic fixtures (medium-high), Sinex as a cleaned pre-alpha dogfood repo
(medium, due to a messier public/private boundary).

**Recoverable vs LOST:** All narrative/report text is fully recovered verbatim (delivery.md +
inline-artifacts.md). LOST (sandbox-only, referenced by name but content never printed to
chat): `Full output tarball`, `Sinnix cleaned bundle`, `Lynchpin cleaned bundle`,
`make_clean_git_exports_v5.py`, `git-cleanup-v5-report.md`, `audit.json`, `SHA256SUMS.txt` —
i.e. the actual rewritten git bundles and the cleanup script itself are gone; only the prose
description of what they contain survives.

**Regeneration value:** Low-to-medium. The prose analysis (what's exposed, what needs
cleanup, publish/don't-publish verdict per repo) is fully captured and directly actionable.
The actual git-rewrite script and bundles are gone and would need to be re-run from scratch
in a fresh session/sandbox if the cleanup itself (not just the analysis) is wanted — the
analysis alone doesn't require regeneration.
