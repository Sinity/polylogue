# Import Pipeline Overview

Polylogue treats every provider through the same pipeline: adapters normalise raw exports into a shared chunk schema, the renderer materialises Markdown/HTML, and state tracking keeps reruns idempotent. This page captures how those pieces fit together.

## Adapter Architecture

- Each provider module under `polylogue/importers/` parses its source format (ZIP bundles, JSONL streams, Drive payloads) and emits a list of normalised chunks.
- Chunks include role, text, attachments, citations, and any tool metadata so downstream steps can render them uniformly.
- Payloads go through schema validation (Pydantic and optional `jsonschema`) so format drift surfaces as actionable errors rather than silent corruption.
- The shared `MarkdownDocument` builder handles collapse thresholds, attachment summaries, optional HTML previews, and diff snapshots, keeping visual output and telemetry consistent across providers.

## Conversation State & Re-imports

- Polylogue persists per-conversation metadata inside `$XDG_STATE_HOME/polylogue/polylogue.db` (SQLite), recording the slug, content hash, timestamps, run settings, and output paths written during the last run.
- Slugs come from `assign_conversation_slug()`, which reuses previously stored names and appends numeric suffixes deterministically when collisions appear—keeping Git history tidy even if provider titles change.
- Before writing files, `conversation_is_current()` compares the provider’s `lastUpdated` value (when available) and the stored hash. If nothing changed, the importer returns an `ImportResult` tagged `skipped` so the CLI can report the outcome without touching disk.
- Stored HTML and `attachments/` directories are reused whenever possible so reruns do not duplicate artefacts or churn timestamps unnecessarily.

## UX & Output Considerations

- **Provider parity**: Adapters preserve provider-specific metadata (tool calls, citations, attachment provenance) while still mapping onto the shared chunk schema.
- **Attachment extraction**: Default heuristics spill large payloads into `attachments/` with short inline previews. Provider-specific overrides can fine-tune thresholds without overwhelming the CLI surface with toggles.
- **Tool interactions**: When an export links tool calls to their results, Polylogue renders them as a single Markdown block and attaches oversized inputs/outputs for later inspection.
- **Interactive tuning**: CLI summaries surface attachment counts, extracted MiB, and token totals so you can spot outliers quickly during imports or watcher runs.
- **Formatting fidelity**: Tables, lists, code blocks, and JSON segments render exactly; Markdown and HTML carry provenance tags (provider, conversation IDs, timestamps) in the front matter.
- **Validation & safety**: Users can relax validation when working with experimental provider features, but the default path errs on the side of explicit failures.

## Supporting Libraries & Integrations

- `markdown-it-py` renders Markdown previews, while `python-frontmatter` keeps YAML headers round-trippable.
- `jinja2` powers the HTML shell today and can be extended for dashboards or richer templates.
- Terminal UX relies on `rich`, `questionary`, `pygments`, and other pure-Python helpers; watchers depend on `watchfiles`; clipboard helpers use `pyperclip`.
- Downstream analysis works well with `ripgrep`, `jq`, and `sqlite-utils`, and the bundled SQLite FTS index keeps metadata queryable without reparsing Markdown.
