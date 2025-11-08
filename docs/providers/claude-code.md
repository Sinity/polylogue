# Claude Code Session Imports

Claude Code stores IDE transcripts beneath `~/.claude/projects/`. Polylogue mirrors those JSONL streams into Markdown while preserving workspace context, shell snapshots, and tool outputs.

## Project Layout

- `projects/<workspace>/*.jsonl`: primary session logs containing interleaved `summary`, `user`, `assistant`, `tool_use`, and `tool_result` entries. Workspace names mirror absolute paths with `/` replaced by `-` (e.g., `/realm/project/demo` → `-realm-project-demo`).
- Supplemental folders:
  - `commands/`: prompt macros and quick actions.
  - `extras/`: binary artefacts linked from sessions.
  - `ide/`: editor state metadata.
  - `shell-snapshots/<session>/<timestamp>.jsonl`: captured terminal transcripts.
  - `todos/`, `tools/`: aux data referenced by tool calls.
- Workspace indexes (`history.json`, `activeProject.json`, `recentCommands.json`) summarise recent sessions and UI state.

## Import Approach

- Stream the per-session JSONL file in order, rebuilding parent/child relationships via `parentUuid`, `leafUuid`, and `sessionId` when branches arise.
- Render `user` and `assistant` events as Markdown turns, keeping embedded code diffs or logs intact.
- Treat `summary` nodes as front-matter notes or callouts—the compaction checkpoints provide useful context for truncated histories.
- Pair `tool_use` entries with matching `tool_result` payloads (shared IDs) and decide whether to inline or extract long outputs using the same heuristics as the Codex importer.
- Attach shell snapshots or referenced files from `shell-snapshots/` and `extras/` into the conversation’s `attachments/` folder so nothing is lost during compaction.

### Branch Export Modes

Claude Code sessions respect the same `--branch-export full|overlay|canonical` flag as other providers. The default `full` layout mirrors the IDE’s branching history in `conversation.common.md` and `branches/<branch-id>/` directories. `overlay` trims the per-branch full copies, leaving just the overlays plus the canonical transcript, while `canonical` writes only `conversation.md` for a flatter archive. The flag applies to `polylogue import claude-code`, `polylogue sync claude-code`, and watchers because they share the registrar-backed pipeline.

## Automation

- Run `polylogue sync claude-code` for one-shot mirroring or `polylogue watch claude-code` for continuous sync. Both commands honour collapse thresholds, HTML output, pruning, and diff generation.
- The importer preserves file mtimes and reuses per-conversation slugs so reruns remain idempotent and Git-friendly.
- Use the `--force` flag when you need to overwrite locally edited transcripts; otherwise manual tweaks stay intact and the session is marked dirty.
- Inline markers such as `\[1]` are normalised to `[1]` so footnotes or numbered references read cleanly in the rendered Markdown.
- Session stats expose `totalWordsApprox`/`inputWordsApprox`, keeping word estimates next to every token count.
- The importer writes canonical and branch-aware artifacts side by side. After each run you’ll find `<slug>.md` plus `<slug>/conversation.md`, `<slug>/conversation.common.md`, and a `branches/<branch-id>/` tree containing `<branch-id>.md` and `overlay.md`. Metadata is recorded in `XDG_STATE_HOME/polylogue/polylogue.db` so downstream tools can query the branch graph.
