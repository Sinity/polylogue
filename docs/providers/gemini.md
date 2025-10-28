# Gemini (Google AI Studio) Imports

Polylogue renders Gemini conversation exports—whether pulled from Google AI Studio directly or synced via Drive—into consistent Markdown with attachment metadata and optional HTML previews.

## Source Formats

- **AI Studio JSON downloads**: Individual transcripts exported from the UI contain a `chunkedPrompt.chunks` array with interleaved user/model turns. Attachment references appear as `driveDocument.id` entries.
- **Drive folders**: The AI Studio “chat exports” folder exposes the same JSON payloads plus attachments that can be fetched through the Drive API.

## Import Strategy

- Parse `chunkedPrompt.chunks` in order, validating each chunk with Polylogue’s shared schema to preserve role, text, and tool metadata.
- Collect `driveDocument.id` references while rendering. When Drive access is configured, Polylogue resolves each ID, downloads the file, and writes it to the conversation’s `attachments/` directory; otherwise, it falls back to a remote Drive link in the Markdown.
- Populate YAML front matter with transcript metadata (title, provider, timestamps) so reruns can reuse slugs, hashes, and attachment decisions.
- Optional HTML previews respect the configured theme (set via the interactive settings menu or config defaults) and include quick links to attachments and Drive assets.

## Sync Notes

- `polylogue render` targets local JSON exports; `polylogue sync drive` connects to the designated Drive folder (`AI Studio` by default) to pull new conversations, honouring filters like `--since`, `--until`, and `--name-filter`.
- Collapse thresholds (`--collapse-threshold`) control how aggressively large Gemini responses are folded in the Markdown viewer.
- Attachments downloaded from Drive inherit the remote modification timestamps; Polylogue applies the same mtime to the rendered Markdown/HTML so Git diffs reflect real content changes.
