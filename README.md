# Gemini Markdown (gmd)

Opinionated tools to render Gemini chat JSON to Markdown and to sync a Google Drive folder ("AI Studio") to an annotated Markdown mirror with linked attachments.

## Install & Auth
- Requires Python 3.9+ and Google API libs (see `flake.nix` or install via pip).
- First Drive use opens a console auth; token saved as `token.json`.

## Quick Start
- Render local JSONs (links only):
  - `python3 gmd.py render sinex --out-dir sinex_md`
- Sync Drive folder by name (no ID needed):
  - `python3 gmd.py sync --folder-name "AI Studio" --out-dir gemini_synced --credentials credentials.json`
  - Skip downloads: add `--remote-links`

## Config (~/.gmdrc)
JSON file with sane defaults; environment variables override.
```
{
  "folder_name": "AI Studio",
  "credentials": "/abs/path/credentials.json",
  "collapse_threshold": 10,
  "out_dir_render": "/abs/path/gmd_out",
  "out_dir_sync": "/abs/path/gemini_synced",
  "remote_links": false
}
```
Env overrides: `GMD_FOLDER_NAME`, `GMD_CREDENTIALS`, `GMD_COLLAPSE_THRESHOLD`, `GMD_OUT_DIR_RENDER`, `GMD_OUT_DIR_SYNC`, `GMD_REMOTE_LINKS`.

## Commands
- `gmd render <file|dir>`: Render local JSON(s) to Markdown.
  - Flags: `--out-dir`, `--credentials`, `--remote-links`, `--download-dir`, `--force`, `--collapse-threshold`, `--interactive`, `-v`.
- `gmd sync`: One-way Drive→local sync to Markdown.
  - Flags: `--folder-name`, `--out-dir`, `--credentials`, `--remote-links`, `--since`, `--until`, `--name-filter`, `--collapse-threshold`, `--interactive`, `--force`, `--prune`, `--dry-run`, `-v`.
- `gmd status`: Show current config, cached folder IDs, recent runs.

## Formatting
- Callouts with grouped model thought+response; long responses folded (configurable).
- YAML includes model params, token/turn stats, attachment counts, context flags, and Drive metadata in sync mode.
- Attachments are always linked (never embedded). Per‑chat `_attachments` folders on sync.

## Interactive (TUI-ish)
- Add `--interactive` to select local files (render) or remote chats (sync) via `fzf` if installed. Previews show basic metadata.

