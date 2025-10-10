# CLI Tips

Polylogue’s CLI bundles a handful of ergonomic helpers that smooth out repetitive workflows. These features are optional and fail gracefully when the environment lacks the required dependencies.

## Clipboard & Credential Workflows

- **Drive onboarding**: During the first Drive sync, Polylogue checks the system clipboard for an OAuth client JSON. If found (and confirmed), it saves the payload to `$XDG_CONFIG_HOME/polylogue/credentials.json`, avoiding manual copy/paste steps.
- **Manual credential import**: When no clipboard payload exists, the CLI guides you through selecting a local file or opening Google’s setup guide. Credentials and tokens always land under `$XDG_CONFIG_HOME/polylogue/`.
- **Copy rendered Markdown**: Pass `--to-clipboard` to render/import commands. When exactly one Markdown file is produced, Polylogue copies it via `pyperclip` and reports success or a warning if clipboard support is unavailable.
- **Graceful fallbacks**: Missing `pyperclip` or clipboard backends never block imports—they only suppress the convenience prompts.
