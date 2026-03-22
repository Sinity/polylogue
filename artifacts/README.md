# Local Generated Artifacts

This directory quarantines heavyweight local working-state outputs that should
not read like part of the source tree.

Typical subdirectories:

- `artifacts/qa/` - local QA runs, ad hoc transcripts, and operator scratch outputs
- `artifacts/mutants/` - local mutation-testing workspaces and raw mutmut state
- `artifacts/demos/` - locally generated VHS/GIF screencasts
- `artifacts/dist/` - optional local package/build outputs
- `artifacts/result/` - optional Nix build out-link if you use `nix build --out-link artifacts/result`

Rules:

- durable documentary evidence belongs under `docs/`
- runtime archive state belongs under Polylogue's XDG/archive roots
- local repo-working artifacts belong here
