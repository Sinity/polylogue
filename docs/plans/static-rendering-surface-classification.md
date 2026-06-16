# Static Rendering Surface Classification

Tracked under #1848. This ledger separates current product/read surfaces from
verification infrastructure so cleanup work can delete stale ceremony without
breaking required gates.

| Path | Classification | Retain / Change | Evidence |
| --- | --- | --- | --- |
| `polylogue/rendering/` | read/export format | Retain | Imported by CLI/API rendering paths; owns Markdown and HTML session output. |
| `polylogue/rendering/templates/session.html` | read/export format | Retain | HTML session renderer template, not a separate product UI. |
| `polylogue/templates/` | dead static-site product residue | Deleted | No live imports or package/runtime references were found; `tests/unit/architecture/test_static_rendering_prune.py` pins absence while retaining `polylogue/rendering/templates/session.html`. |
| `polylogue/publication/` | report manifest helper | Retain for lab reports | Used by report writers as an output manifest type. |
| `devtools/pages_builder.py` | documentation site build | Retain | `devtools render-pages` and site renderer tests exercise it. |
| `devtools/pages_templates.py` | documentation site build | Retain | Template source for `pages_builder.py`. |
| `devtools/render_pages.py` | documentation site build | Retain | Registered generated-surface command and covered by renderer tests. |
| `devtools/generate_readme_media.py` | public docs media generator | Retain and police | Generates README diagrams; stale product labels are blocked by the public-surface audit. |
| `docs/media/*.mmd` | public docs media source | Retain and police | Mermaid sources are public README assets and must avoid stale product wording. |
| `docs/media/*.svg` | public docs media output | Retain | Committed render output for environments without `mmdc`. |
| `polylogue/scenarios/` | fixture/demo and contract substrate | Retain | Used by demo import, validation lanes, benchmarks, and pipeline probes. |
| `polylogue/showcase/` | verification-lab runner substrate | Retain until #1849 replacement | Required by `devtools lab-scenario`, validation lanes, and command help baselines. |
| `tests/baselines/showcase/` | generated command contract fixtures | Retain until #1849 replacement | Required by `devtools lab-scenario verify-baselines`. |
| `polylogue/verification/` | manifest validation substrate | Retain | Used by manifest verification tooling. |

Deletion candidates after #1849 or a focused import graph proves disuse:

- any `polylogue/showcase/*` report/output adapters that are no longer used by
  `devtools lab-scenario`, validation lanes, or command help baselines.

Current rule: public docs and README media describe the web reader,
read/export output, demos, and verification-lab evidence. Internal devtools,
baseline, and historical docs may keep older terms only where they name
existing verification machinery.
