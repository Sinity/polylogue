# Static Rendering Surface Classification

Tracked under #1848. This ledger separates current product/read surfaces from
verification infrastructure so cleanup work can delete stale ceremony without
breaking required gates.

| Path | Classification | Retain / Change | Evidence |
| --- | --- | --- | --- |
| `polylogue/rendering/` | read/export format | Retain | Imported by CLI/API rendering paths; owns Markdown and HTML session output. |
| `polylogue/rendering/templates/session.html` | read/export format | Retain | HTML session renderer template, not a separate product UI. |
| `polylogue/templates/` | dead static-site product residue | Deleted | No live imports or package/runtime references were found; the retained HTML session renderer lives under `polylogue/rendering/templates/session.html`. |
| `polylogue/publication/` | report manifest helper | Retain for lab reports | Used by report writers as an output manifest type. |
| `devtools/pages_builder.py` | documentation site build | Retain | `devtools render pages` and site renderer tests exercise it. |
| `devtools/pages_templates.py` | documentation site build | Retain | Template source for `pages_builder.py`. |
| `devtools/render_pages.py` | documentation site build | Retain | Registered generated-surface command and covered by renderer tests. |
| `devtools/generate_readme_media.py` | public docs media generator | Retain | Generates README diagrams; normal render/docs review owns accuracy. |
| `docs/media/*.mmd` | public docs media source | Retain | Mermaid sources are public README assets reviewed with the README and generated media. |
| `docs/media/*.svg` | public docs media output | Retain | Committed render output for environments without `mmdc`. |
| `polylogue/scenarios/` | fixture/demo and contract substrate | Retain | Used by demo import, validation lanes, benchmarks, and pipeline probes. |
| `polylogue/showcase/` | verification-lab runner substrate | Retain until #1849 replacement | Required by `devtools lab-scenario`, validation lanes, and command help baselines. |
| `polylogue/showcase/qa_report.py` | obsolete report aggregator | Deleted | Callers now import the owned QA payload, Markdown, and summary modules directly; the lab runner/report persistence path remains intact. |
| `tests/baselines/showcase/` | generated command contract fixtures | Retain until #1849 replacement | Required by `devtools lab-scenario verify-baselines`. |
| `polylogue/verification/` | manifest validation substrate | Retain | Used by manifest verification tooling. |

Deletion candidates after #1849 or a focused import graph proves disuse:

- any `polylogue/showcase/*` report/output adapters that are no longer used by
  `devtools lab-scenario`, validation lanes, or command help baselines.

Current rule: public docs and README media describe the current web reader,
read/export output, demos, and verification-lab evidence directly. Internal
devtools and historical docs should name the current machinery plainly.
