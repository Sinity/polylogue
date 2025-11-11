# Polylogue Message-Level Refactor & Branch Support Plan

This document captures the agreed architecture and implementation plan so development can continue even if the active context is lost.

---

## 0. Terminology
- **Conversation**: Provider-level chat object (identified by `(provider, conversation_id)`).
- **Message**: Individual turn (user/assistant/tool/etc.) within a conversation.
- **Branch**: A divergent path in the conversation message graph (e.g., multiple ChatGPT continuations).
- **Canonical branch**: The branch the provider marks as current/active.

---

## Roadmap Overview

| Phase | Focus | Key Deliverables |
| --- | --- | --- |
| **A. Persistence** | Hardening the SQLite persistence layer and migration helpers. | Section 1 (schema) and Section 7 (migration tasks). |
| **B. Message Pipeline** | Normalize provider payloads into a branch-aware graph and regenerate Markdown variants. | Sections 2–3 (ingestion, rendering, filesystem layout). |
| **C. Automation** | Extend watchers/import commands to use the new pipeline for all providers. | Section 4 (generic watcher + CLI updates). |
| **D. Insights** | Move telemetry and indexing onto SQLite; expose richer status outputs. | Sections 5–6 (observability + embeddings). |
| **E. Polish & Docs** | Final documentation, UX adjustments, and backlog cleanup. | Sections 7–8 (documentation rollout + enhancements). |

Treat Phase A as a prerequisite for every other phase. Later phases can proceed in parallel once the core schema and migration paths are stable.

---

## 1. Storage & Schema

### 1.1 SQLite database
`polylogue.db` is the single source of truth for state/runs; the schema below describes the canonical layout (legacy JSON caches are only read for migration).

- `conversations`  
  - `provider TEXT`  
  - `conversation_id TEXT`  
  - `slug TEXT`  
  - `title TEXT`  
  - `current_branch TEXT`  
  - `root_message_id TEXT`  
  - `last_updated TEXT` (provider timestamp)  
  - `content_hash TEXT` (hash of canonical branch body)  
  - `metadata_json TEXT` (collapse threshold, attachment policy, etc.)  
  - **PK** `(provider, conversation_id)`

- `branches`  
  - `provider TEXT`  
  - `conversation_id TEXT`  
  - `branch_id TEXT` (opaque id such as `root/0/1`)  
  - `parent_branch_id TEXT`  
  - `label TEXT` (provider-supplied or generated)  
  - `depth INTEGER`  
  - `is_current INTEGER` (boolean)  
  - **PK** `(provider, conversation_id, branch_id)`

- `messages`  
  - `provider TEXT`  
  - `conversation_id TEXT`  
  - `message_id TEXT` (provider ID or synthetic)  
  - `parent_id TEXT` (links to previous message)  
  - `branch_id TEXT` (matches `branches.branch_id`)  
  - `position INTEGER` (sequence order within branch)  
  - `timestamp TEXT`  
  - `role TEXT`  
  - `content_hash TEXT`  
  - `raw_json TEXT` (original provider payload for lossless reconstruction)  
  - `rendered_text TEXT` (plain text form used when rebuilding Markdown)  
  - `metadata_json TEXT` (attachments, tool info, etc.)  
  - **PK** `(provider, conversation_id, message_id)`

- `runs`  
  - `timestamp TEXT`  
  - `cmd TEXT`  
  - `count INTEGER`  
  - `attachments INTEGER`  
  - `attachment_bytes INTEGER`  
  - `tokens INTEGER`  
  - `skipped INTEGER`  
  - `pruned INTEGER`  
  - `diffs INTEGER`  
  - `duration REAL` (seconds)  
  - `out TEXT` (destination path)  
  - `provider TEXT` (optional)  
  - `branch_id TEXT` (optional)  

- `messages_fts` (SQLite FTS5 virtual table)  
  - `provider`  
  - `conversation_id`  
  - `branch_id`  
  - `message_id`  
  - `content` (tokenised text)

### 1.2 Migration
1. On first run after upgrade, read legacy `state.json` and `runs.json`.  
2. Populate SQLite tables (`conversations`, `runs`).  
3. Ingest existing Markdown to seed `messages` if necessary (optional; may re-render from provider export).  
4. Preserve JSON files for rollback; new writes go to SQLite.  
5. Run migration logic idempotently so a partially upgraded install can retry without manual cleanup.  
6. Emit a JSON/structured log summary describing how many conversations/runs migrated and any files skipped.

### 1.3 Data retention & backups
- Before migration begins, instruct the user (or automation) to snapshot the existing Markdown directories and state files.
- Keep the old `state.json` / `runs.json` alongside a migration stamp so operators can roll back if needed.
- Provide a `polylogue doctor --migrate-check` mode that confirms schema health without writing data (useful for CI and smoke tests).

---

## 2. Import / Render Workflow

### 2.1 Message ingestion
1. Parse provider payload into message objects (one per turn/tool invocation). Normalise provider-specific quirks (e.g., ChatGPT mapping graphs, Claude tool_use blocks) into a common internal schema.  
2. Derive `message_id`, `parent_id`, `branch_id`, `timestamp`, `role`, `content_hash`, and cache the raw JSON payload for lossless reconstruction.  
3. Upsert into `messages`. If `content_hash` differs for the same `message_id`, treat it as a new revision: emit a fresh `branch_id` and flag the old branch as non-current.  
4. Update `branches` table: create entries for new branch paths; mark canonical branch via provider metadata (or inferred heuristics when the provider is silent).  
5. Update `conversations` with the latest canonical branch (`current_branch`), `last_updated`, canonical content hash, and derived stats (token counts, attachment totals).  
6. Record attachment references and Drive IDs during ingestion so the renderer has enough metadata to fetch/link assets without re-parsing the source file.

### 2.2 Markdown generation
Support three modes (configurable, default = canonical branch):

1. **Canonical transcript**: Render current branch into `conversation.md` in the conversation folder.  
2. **Full branch transcripts**: Render every branch—canonical path included—into `branches/<branch-id>/<branch-id>.md`, recursing for nested branches. The canonical branch’s full transcript is identical to the upstream `conversation.md`, but is duplicated inside `branches/<canonical-id>/` so that every branch directory is self-contained.  
3. **Overlay format**:  
   - Store the shared prefix (messages up to the split point) as `conversation.common.md`.  
   - Each branch directory contains:  
     - `overlay.md` with only the divergent messages that follow the branch point (referencing the common prefix one level up).  
     - `<branch-id>.md` (the full transcript for that branch, combining the prefix plus overlay).

Recursive branching: branch directories recurse with the same structure, so sub-branches carry their own `overlay.md` and `<child-branch>.md`. Branch sets are never singular: the moment a split occurs, the canonical path is treated as one branch and at least one alternate branch is present. Deterministic naming (e.g., provider ID, synthetic slug) ensures there are no gaps even when providers omit or reshuffle their own labels.

### 2.3 Attachments & metadata
- Attachments stored per conversation as today.  
- `metadata_json` (collapse threshold, attachment policy, dirty flag) stored in SQLite.  
- Markdown front matter includes a `polylogue` block referencing branch info, state hash, and dirty flag.

---

## 3. Filesystem Layout

When the pipeline writes a conversation, it produces a self-contained directory tree under the provider root. The canonical transcript and branch variants live side by side so that consumers (humans, downstream tools) can choose either a full rendering or a delta-only overlay.

```
<markdown_root>/
  <provider>/                        # e.g., drive-sync, chatgpt, claude-code
    <conversation-slug>/             # deterministic slug for this conversation
      conversation.md                # canonical branch (latest provider branch)
      conversation.common.md         # shared prefix for all downstream branches
      attachments/                   # extracted files linked from any branch
        <attachment files...>
      branches/
        <branch-id>/                 # deterministic branch identifier (canon included)
          <branch-id>.md             # full transcript for this branch
          overlay.md                 # only the divergent messages after the split
          attachments/               # optional per-branch attachments (rare)
            <branch-specific assets>
          branches/                  # nested structure repeats for deeper forks
            <child-branch-id>/
              <child-branch-id>.md
              overlay.md
              ...
```

Key points:

- The canonical branch (provider-selected current path) is duplicated inside `branches/<canonical-id>/` so that every branch directory is self-contained. Its `<branch-id>.md` is byte-identical to `conversation.md`.
- `conversation.common.md` captures the shared sequence of messages up to the first divergence. Branch overlays reference this file so they only need to store the delta that follows.
- Attachments that apply to the entire conversation live at the top-level `attachments/`. Branch-specific assets (rare) can nest inside a branch directory’s own `attachments/` if needed.
- Branch identifiers are deterministic (provider IDs when available, otherwise synthetic slugs). The moment a split occurs there will be at least two branch directories: the canonical continuation and one alternate. Deeper splits continue this pattern recursively.
- Providers with structured source paths (e.g., Codex date folders, Claude project hierarchies) mirror that structure inside `<provider>/` before the conversation slug so existing tooling can rely on stable locations.

---

## 4. Local Sync

### 4.1 Generic watcher
Create a provider-agnostic watcher which:
- Accepts a root path and provider parser.  
- Supports recursive watching (necessary for Codex/Claude Code).  
- Detects file creation/modification (JSON/JSONL) and batches rapid bursts with debounce settings.  
- Calls provider importer → message pipeline.  
- Reuses the SQLite state to skip sessions that have not changed, mirroring the same idempotence guarantees as one-shot imports.  
- Mirrors source layout in the markdown root (see Filesystem section).

Targets:
- Codex (`~/.codex/sessions`)  
- Claude Code (`~/.claude/projects`)  
- ChatGPT local exports (zip/tar extracted directory)  
- Claude.ai exports (if provided)  
- Future providers as needed.

### 4.2 CLI integration
- Extend `polylogue watch` to accept `--provider` and optional `--base-dir`.  
- Default output structure auto-generated per provider.  
- For non-watched exports (`polylogue import chatgpt`), reuse the same message pipeline.

---

## 5. Observability

- Use the SQLite `runs` table (JSON caches are legacy-only).  
- CLI `status` reads from SQLite, aggregating per command and provider:
  - counts, attachments, tokens, diffs, duration.  
  - provider summary groups commands (`sync`, `codex-watch`, etc.).  
- `status --json` returns structured payload; `status --watch --interval N` repeats.  
- Optional `status --dump json` to export last N rows for external tooling.  
- Mirror the summary to stderr/syslog-friendly lines so cron/systemd jobs surface failures without parsing JSON.  
- Every render/sync/import/watch run emits a `polylogue_run` JSON line on stderr (set `POLYLOGUE_RUN_LOG=0` to disable) so journald/cron logs contain machine-readable telemetry, including Drive retry metadata.  
- `status --dump-only` writes the requested JSON without printing summaries, and the automation snippets expose `--status-log/--status-limit` flags so `ExecStartPost`/cron entries automatically record those dumps for dashboards.  
- No HTTP or Prometheus integration; users can run SQL queries or scripts as needed.

---

## 6. Indexing & Embeddings

- Maintain SQLite FTS for message text.  
- Optional Qdrant support remains behind `POLYLOGUE_INDEX_BACKEND=qdrant`.  
- Future: when Voyage embeddings are ready, export message text + metadata to that pipeline; ingest resulting vectors into Qdrant (or whatever backend we choose).  
- Until then, vectors are placeholders (simple numeric features) so the plumbing is ready but inert.

Environment variables:
- `POLYLOGUE_INDEX_BACKEND` = `sqlite` (default), `qdrant`, `none`.  
- `POLYLOGUE_QDRANT_URL`, `POLYLOGUE_QDRANT_API_KEY`, `POLYLOGUE_QDRANT_COLLECTION`.

---

## 7. TODO Checklist

1. **Schema Migration**
   - [x] Implement SQLite tables (conversations, branches, messages, messages_fts, runs).  
   - [x] Migration script from `state.json` / `runs.json`.  
   - [x] Tests covering forward/backward compatibility.

2. **Importer Refactor**
   - [x] Parse messages, compute hashes/IDs, branch metadata.  
   - [x] Upsert into SQLite tables.  
   - [x] Ensure old exports (older timestamps) are skipped gracefully.

3. **Markdown Export**
   - [x] Implement canonical render, overlays, and full branch copies.  
   - [x] CLI flags to choose export mode.  
   - [x] Mirror provider-specific directory structures.

4. **Local Sync Generalisation**
   - [x] Generic watcher that supports provider-specific parsers.  
   - [x] Add ChatGPT/Claude exporters to pipeline.  
   - [x] Update CLI (`watch` subcommand).

5. **Observability Update**
   - [x] Switch `status` to SQLite runs table.  
   - [x] Add `status --dump`.  
   - [x] Update tests (`tests/test_status.py`).

6. **Index Integration**
   - [x] Adjust FTS to index messages (render/sync pipelines now register chunks directly with the registrar so `messages_fts` stays authoritative).  
   - [x] Keep Qdrant adapter stub; plan for real embeddings once Voyage pipeline consumes SQLite.

7. **Documentation & Rollout**
   - [x] Update README, docs/automation.md, and provider docs with the new architecture, branch export modes, environment variables, and local sync behaviour.  
   - [x] Publish migration guidance (backup strategy, downgrade path, feature flag if applicable).  
   - [x] Announce the change to beta users and capture early feedback before the general release.

---

## 8. Additional Enhancements

- **Drive sync discovery**: Replace the current “no dots in filename” filter in `DriveClient.list_chats()` with a MIME-type allowlist or explicit provider metadata so Gemini transcripts whose titles include periods are still synced, while Drive-native docs remain excluded.
- **Gemini importer follow-ups**: After the message-level pipeline lands, migrate Gemini rendering onto it, replace filename heuristics with MIME detection, and evaluate optional embedding/vector indexing for Gemini content.
- **Importer auto-detection**: Extend `polylogue import <path>` so archives automatically route to the correct importer (zip → provider module, JSON/JSONL → inferred provider).
- **Watcher telemetry**: (Done) Watchers now reuse the same run pipeline, emitting `polylogue_run` JSON lines and optional `status --dump` snapshots for automation snippets so unattended jobs feed dashboards.
- **Recurring export guidance**: Document or script repeatable Playwright/automation flows for ChatGPT/Claude exports once we are confident in the workflow and messaging to users.
- **Richer previews**: Prototype a TUI/HTML preview that shows Markdown, HTML, and attachments side by side, powered by the new chunk-level metadata.

---

## 9. Notes & Considerations

- **Branch naming**: use provider metadata when available; else generate deterministic IDs (`branch-<numeric>`).  
- **Branch overlays**: must reference shared prefix path; ensure Markdown includes clear markers.  
- **Attachments**: remain filesystem-based; no need to duplicate in SQLite.  
- **Error handling**: message-level ingestion must tolerate missing parent IDs (log + skip) rather than abort entire import.  
- **Testing strategy**:  
  - Unit tests for branch rendering, importer idempotency, watcher change detection.  
  - Integration tests verifying SQLite schema, state migration, and CLI outputs.
- **Rolling upgrades**: keep ability to read existing Markdown; regen from provider data when needed.  
- **Documentation pointers**:  
  - Import pipeline internals live in `docs/import_pipeline.md`.  
  - CLI ergonomics (clipboard, credentials, automation flags) live in `docs/cli_tips.md`.  
  - Live capture caveats are tracked in `docs/live_capture.md`.

---

This plan supersedes previous high-level notes. All future implementation should adhere to the details above unless explicitly updated.
