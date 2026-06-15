# Release Readiness Gate

Issue: #1827.

This gate decides whether Polylogue is externally presentable enough to merge a
release PR. A green release workflow is not sufficient: the release must be
installable, demoable without private data, and truthful about shipped command,
read, import, and output surfaces.

## Gate Rule

The release PR remains held unless every required item below is either:

- `satisfied`, with a PR/check/document citation; or
- `scoped out`, with release notes that explicitly avoid claiming the missing
  capability.

Do not use release pressure to force unrelated architecture work. Design-frontier
issues can stay open when the README and release notes do not advertise them as
shipped product behavior.

## Required Local Commands

Run these from a clean checkout inside the devshell:

```bash
devtools release-readiness
devtools verify --quick
devtools verify --lab
devtools build-package
devtools render-pages
devtools verify-doc-commands
```

Add focused commands for changed release surfaces:

```bash
devtools test tests/unit/cli/test_query_verbs_runtime.py
devtools test tests/unit/storage/test_blackboard_facade.py
devtools lab-scenario verify-baselines
```

If packaging, Nix, or dependency metadata changed, also run:

```bash
nix flake check
```

## Automated Gate Matrix

| Area | Required Evidence | Current Owner |
| --- | --- | --- |
| Command floor | Final public command tree has no stale command aliases or undocumented examples. | #1842 |
| Machine output | JSON is finite, NDJSON is streaming, mutation/error envelopes are typed, and machine-output prompts do not block. | #1818, #1816 |
| README commands | README examples resolve against live commands and do not cite stale APIs. | #1841 |
| Import/demo | Public import vocabulary is stable and demo mode has deterministic private-data-free fixtures. | #1815, #1843 |
| Recovery/digest | Agent-session recovery views are available only to the extent claimed by README/release notes. | #1880 |
| Assertion/user state | User-tier note/assertion data is preserved and reset/delete flows are guarded. | #1883, #1839 |
| Public vocabulary | Public surfaces use session/origin vocabulary; provider/conversation terms are raw-source or historical only. | #1810 |
| Web/API | Any advertised web/API surface has stable DTOs, auth posture, and route vocabulary. | #1847, #1846 |
| Static/docs surface | Generated docs, pages, media, and topology status are current. | #1848, #1849 |
| CI reliability | Known benchmark/test flakes are resolved or explicitly excluded from the release gate with rationale. | #1878 |
| Packaging | Wheel/sdist/Nix package expose only supported runtime entrypoints. | `devtools build-package`, `nix flake check` |

## Manual Release Review

Before merging a release PR, record the answers in the PR body:

| Question | Required Answer |
| --- | --- |
| What can a new user run first? | Exact command sequence, including archive root/demo root. |
| Does the sequence touch private archives? | No, unless the user supplied an explicit source path. |
| Which origins are advertised? | Only origins with current parser/import/read evidence. |
| Which features are deliberately not shipped? | Listed in release notes, with open issue refs. |
| What irreversible publication happens? | PyPI/GHCR/tag behavior stated before merge. |
| Which local gates ran? | Exact commands and key pass/fail line. |

## Current Status

Satisfied:

- #1810 public session/origin vocabulary sweep is closed.
- #1818 machine-output concrete violations are closed.
- #1841 README cockpit is landed and command examples are checked by
  `verify-doc-commands`.
- #1878 benchmark convergence flake is closed.
- #1880 has recovery digest registry, extraction, CLI read view, Python API
  facade, and GitHub/check event extraction slices.
- #1883 has the assertions table, write-through adapters, delete/status
  transitions, reset user.db guard, and blackboard assertion metadata slices.
- #1843 has deterministic demo corpus specs and the `polylogue import --demo`
  scheduling surface.

Still blocking external release claims:

- #1843 still needs end-to-end demo archive convergence evidence and
  README-ready commands against the generated demo archive.
- #1816 generated action-contract replacement is not landed.
- #1847/#1846 web/API release scope is not settled.
- #1848/#1849 static/docs/proof pruning is not settled.
- #1880 persisted transform outputs and `continue`/`blame` report presets are
  not landed; do not advertise them as shipped.

## Release PR Body Requirements

Use this section in the release PR:

```text
Release gate:
- Command floor:
- Machine output:
- README/demo:
- Import/demo fixture:
- Recovery/digest:
- Web/API scope:
- Packaging:
- Known caveats scoped out:

Verification:
- <command> — <key output line>
```
