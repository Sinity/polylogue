# ChatGPT Export & Import Workflow

Polylogue ingests chat.openai.com exports (consumer ChatGPT) by walking the provider’s bundle and normalising the threaded “mapping” graph into a linear markdown transcript.

## Export Bundle

- Exports arrive as a ZIP that contains `conversations.json`, `user.json`, attachments, and shared threads.
- Each conversation stores a `mapping` dictionary keyed by node IDs. Nodes expose:
  - `parent` / `children` edges that describe the conversation tree.
  - `message.content.parts` containing the rendered text, code blocks, lists, and tables.
  - `metadata` with citations, attachments, and source pointers.
- Attachments referenced through `attachment://` URIs (or attachment metadata) live in the ZIP beside the JSON payload.

## Import Strategy

- Flatten the `mapping` using the provider-supplied `current_node` as the canonical branch. A depth-first walk that follows `parent` pointers reconstructs the ordered turns.
- For each `content.part`:
  - Preserve lists and tables as Markdown structures.
  - Render code blocks with their language hints (`code` parts expose `mime_type` or `language`).
  - Convert citation markers (private-use characters such as `\uE200`) into Markdown footnotes using the accompanying entry in `message.metadata.citations`.
- Tool/function messages show up as `recipient == "assistant"` payloads or `content_type == "tool_results"`. Pair the call/result when the export links them so the Markdown contains a single tool block.

### Branch Export Modes

`polylogue import chatgpt` (and downstream sync/watch flows) accept `--branch-export full|overlay|canonical` so you can control how the on-disk tree looks:

- `full` (default) writes `conversation.md`, `conversation.common.md`, and every branch’s `branches/<branch-id>/{<branch-id>.md, overlay.md}` directory.
- `overlay` keeps the canonical transcript plus per-branch `overlay.md` files, omitting the full per-branch copies when all you need are the deltas.
- `canonical` limits the output to `conversation.md` under each slug, skipping the branch tree entirely when you want a flat archive.

Because chat imports now travel through the same registrar/pipeline stack as sync/watch, the `--branch-export` flag behaves consistently everywhere (render, sync drive/codex/claude-code, and watchers).

## Attachments

- When attachments exist inside the ZIP, copy them into the conversation’s `attachments/` directory and link from Markdown.
- If the export references a remote Google Drive asset without including the file, fall back to an external link so the reader can still reach the asset.

## Operational Notes

- There is no public API for historical exports. Users must trigger the ZIP export manually (Settings → Data Controls → Export) or orchestrate a headless browser flow.
- Polylogue runs are idempotent: once the ZIP is refreshed, `polylogue import chatgpt` reuses stored slugs and skips conversations whose hashes match previous runs.
- Use `--force` when you need to overwrite locally-edited Markdown; otherwise the importer preserves manual tweaks and marks the conversation as dirty.
- Inline escape sequences such as `\[1]` in the export are normalised to `[1]` so numeric references render cleanly.
- Per-turn headers show UTC timestamps with a relative offset from the start of the conversation (for example, `2024-04-30T19:00:46Z (+22s)`), keeping both absolute and contextual timing at a glance.
- Front matter stats now add `totalWordsApprox`/`inputWordsApprox` alongside the token counts so every “tokens” figure has a matching word estimate.
- Each import populates `XDG_STATE_HOME/polylogue/polylogue.db` with conversation, branch, and message metadata. The canonical transcript still lives at `<slug>.md`, while a branch-aware directory tree is written alongside it: `<slug>/conversation.md`, `<slug>/conversation.common.md`, and `branches/<branch-id>/{<branch-id>.md, overlay.md}`.

## Local Sync Workflow

For recurring exports, you no longer need to call `polylogue import chatgpt …` manually each time. Treat the bundles like a local provider instead:

1. Save each ZIP (or its extracted folder with `conversations.json`) under `$XDG_DATA_HOME/polylogue/exports/chatgpt` — by default `~/.local/share/polylogue/exports/chatgpt`. The directory is created automatically the first time you run the sync.
2. Run `polylogue sync chatgpt` to ingest every bundle in that directory, or pick specific bundles interactively when not using `--all`.
3. Pass `--base-dir /path/to/exports` if you keep the bundles somewhere else, and reuse the familiar flags (`--branch-export`, `--html`, etc.) to keep outputs consistent with the other providers.

Watch mode stays limited to Codex/Claude Code—ChatGPT exports arrive as one-off bundles, so there’s no live directory to monitor. Kick off `polylogue sync chatgpt` whenever the export email arrives and the registrar will keep slugs, stats, and branch metadata perfectly in sync.
