# Polylogue Architecture (Nov 2024)

Polylogue now follows a modular structure that separates orchestration concerns from provider-specific logic. The key building blocks are:

## Command Registry

- `polylogue.cli.registry.CommandRegistry` maps CLI verbs to handlers.
- `polylogue.cli.app` registers commands and uses the registry for dispatch.
- Makes it easy to add new commands or reuse handlers across CLI/parsers/UI.

## Console Facade & UI

- `polylogue/ui/facade.py` wraps Rich/gum functionality (the devshell ships these by default) and automatically drops to the plain console whenever stdout/stderr aren’t TTYs (or when `POLYLOGUE_FORCE_PLAIN=1` is set).
- `polylogue.ui.UI` delegates to the facade but still provides interactive prompts (pickers, yes/no prompts, etc.) whenever the session is interactive (use `--interactive` to force prompts even in headless shells).

## Provider Sessions

- `polylogue/providers/registry.ProviderRegistry` registers instantiated provider SDKs (currently Google Drive) behind a small protocol so commands can fetch consistent sessions without knowing the underlying client implementation.
- `polylogue.providers.drive.DriveProviderSession` wraps `DriveClient` and ensures retry instrumentation/telemetry stay consistent, while still presenting the familiar Drive API to pipelines and CLI helpers.

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
- Explore richer CLI ergonomics (completions, faceted status dashboards) on top of the existing registry/facade stack. Planned completion work includes:
  - bringing the dynamic engine to bash/fish so every shell benefits from live suggestions;
  - expanding suggestions beyond slugs/providers to include filtered Drive chats, recent session files, and constrained flag values discovered from argparse metadata;
  - caching completions to stay snappy even on large archives;
  - rethinking annotation formats (e.g., structured JSON) so shells can render descriptions/tooltips cleanly.

This structure makes it easier to add providers, surface richer UI, and build new tooling on top of Polylogue without duplicating plumbing code.
