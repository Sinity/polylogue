# Automation Recipes

Polylogue ships with interactive workflows by default, but you can keep local archives up to date without manual effort. This page collects the recommended approaches.

## Real-Time Watchers

The CLI includes long-running watchers that mirror local Codex and Claude Code session stores as soon as files change:

```bash
# Watch ~/.codex/sessions/ and sync into the standard archive
python3 polylogue.py watch codex --out /realm/data/chatlog/markdown/codex

# Watch ~/.claude/projects/ for Claude Code sessions
python3 polylogue.py watch claude-code --out /realm/data/chatlog/markdown/claude-code
```

Watchers reuse your configured collapse thresholds and HTML defaults. Pass `--collapse-threshold`, `--html`, or `--html-theme` to override. The `--debounce` flag (default: 2 seconds) throttles repeated syncs when many files change at once.

## Systemd Timer Template

For environments that prefer scheduled jobs, create a systemd service/timer pair. The example below runs the Codex sync every 10 minutes in a user session:

`~/.config/systemd/user/polylogue-sync-codex.service`
```ini
[Unit]
Description=Polylogue Codex sync

[Service]
Type=oneshot
Environment=XDG_STATE_HOME=%h/.local/state
WorkingDirectory=/realm/project/aichat-to-md
ExecStart=%h/.nix-profile/bin/python3 polylogue.py sync-codex --plain --json
```

`~/.config/systemd/user/polylogue-sync-codex.timer`
```ini
[Unit]
Description=Polylogue Codex sync timer

[Timer]
OnBootSec=2m
OnUnitActiveSec=10m
Persistent=true

[Install]
WantedBy=default.target
```

Enable the timer with `systemctl --user enable --now polylogue-sync-codex.timer`. Duplicate the pair with different names for Claude Code or Drive syncs, adjusting arguments as needed.

## Cron Alternative

When systemd is unavailable, a simple cron entry can invoke the same non-interactive commands:

```
*/30 * * * * XDG_STATE_HOME="$HOME/.local/state" cd /realm/project/aichat-to-md && python3 polylogue.py sync-codex --plain --json >> "$HOME/.cache/polylogue-sync.log" 2>&1
```

Combine cron with `polylogue status --json` to monitor recent runs or parse the generated log file.

## Tips

- Always use `--plain` for unattended runs so Polylogue skips interactive gum/skim prompts.
- Combine `--json` with your favourite log processor to detect failures.
- Watchers and scheduled jobs reuse the same slug/state tracking introduced in this release, so repeat exports and recurring syncs overwrite the correct Markdown without leaving duplicates.
