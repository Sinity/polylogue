# November 2024 Refresh

- **Registrar everywhere:** Local Codex/Claude Code syncs now reuse the shared `ConversationRegistrar`, so state.json and SQLite stay in lockstep with Drive/render pipelines.
- **Pipeline telemetry:** Every stage records name, status, and duration in `PipelineContext.history`, making it easy to assert on execution paths in tests and surface actionable errors when something fails.
- **Stronger tests:** Added coverage for Drive token bootstrap, sync watcher debounce behaviour, and the registrar contract itself to keep the new architecture stable.
- **Interactive hints:** The main menu now reminds users how to navigate gum pickers—arrow keys, Enter, `q`—so the UX feels intentional from the first run.
