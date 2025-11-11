# Polylogue Architecture (Nov 2024)

Polylogue now follows a modular structure that separates orchestration concerns from provider-specific logic. The key building blocks are:

## Command Registry

- `polylogue.cli.registry.CommandRegistry` maps CLI verbs to handlers.
- `polylogue.cli.app` registers commands and uses the registry for dispatch and for the interactive menu.
- Makes it easy to add new commands or reuse handlers across CLI/parsers/UI.

## Console Facade & UI

- `polylogue/ui/facade.py` wraps Rich/gum functionality (the devshell ships these by default) and only drops to the plain console when `--plain` is explicitly requested or CI disables the interactive stack.
- `polylogue.ui.UI` delegates to the facade but now provides interactive prompts even in plain mode (numeric selection, yes/no prompts, etc.).
- The interactive menu groups actions (“Render & Import”, “Sync & Inspect”, “Maintenance”), shows a status snapshot using `status_command`, and relies on the registry for help text.

## Persistence Layer

- `polylogue/persistence/state.py` exposes `ConversationStateRepository` backed entirely by SQLite metadata (no more JSON cache files).
- `polylogue/persistence/database.py` provides `ConversationDatabase`, a thin wrapper around the archive database as well as helper queries for doctor/status/search.
- `polylogue/services/conversation_registrar.ConversationRegistrar` coordinates writes across the SQLite metadata tables and the on-disk archive. Imports, sync, render, and branching now funnel persistence through the registrar so branch/message tables stay consistent.
- `polylogue/services/conversation_service.ConversationService` presents a read-only facade (state iterators, conversation listings, deletes) so doctor/status/search no longer poke at repositories directly.
- Doctor, status, and search use the service instead of touching JSON/SQLite directly.

## Pipeline Framework

- `polylogue/pipeline_runner.py` introduces a simple `Pipeline` + `PipelineContext` abstraction.
- Render and Drive sync flows now compose stages:
  - SourceReader (`RenderReadStage` / `DriveDownloadStage`)
  - Normaliser (`RenderNormalizeStage` / `DriveNormalizeStage`)
  - Transformer (`RenderDocumentStage`)
  - Persistence (`RenderPersistStage`)
- Pipelines handle aborts and context sharing, making it straightforward to extend or reuse stages across providers.
- Each run records per-stage telemetry (name, status, and duration) in the `PipelineContext.history`, so callers and tests can assert on the exact execution path and surface meaningful errors.

## Schema Validation

- `polylogue/validation.py` provides reusable helpers (currently `ensure_chunked_prompt`) that guard against malformed provider payloads.
- Normalisation stages call these helpers before sanitising chunks, so users see clear error messages instead of silent skips when exports are incomplete.

## Domain & Configuration

- `polylogue/domain/models.py` defines provider-independent dataclasses (Conversation, Branch, Message, Attachment).
- `polylogue/core/configuration.py` loads layered configuration (defaults, file, env). `polylogue/config.py` keeps the previous API but delegates to the new loader.
- `polylogue/archive/service.py` centralises archive path resolution via the configuration defaults.

## Testing

- Integration tests cover the CLI (including the new plain-mode prompts).
- Dedicated tests exercise the pipeline runner and the new facade logic.

## Next Steps

- Extend the pipeline stages with richer validation (e.g., schema checks for provider exports) and tighten error reporting.
- Build end-to-end snapshots for `sync`/`render` flows that assert both Markdown output and metadata side effects (state + SQLite) to guard against regressions.
- Explore packaging the console menu as a reusable service for the forthcoming TUI revamp (faceted filters, richer status dashboards).

This structure makes it easier to add providers, surface richer UI, and build new automation on top of Polylogue without duplicating plumbing code.
