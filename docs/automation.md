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

Watchers run an initial full sync, then rerun whenever `.jsonl` session files change inside the source tree. They reuse your configured collapse thresholds and HTML defaults; pass `--collapse-threshold`, `--html`, or `--html-theme` to override. The `--debounce` flag (default: 2 seconds) throttles rapid bursts of file events so a single editing session doesn’t thrash the pipeline.

If the optional `watchfiles` dependency is missing, the CLI prints guidance instead of failing—meaning you can still fall back to scheduled syncs. Every watcher pass flows through `_log_local_sync`, so the resulting telemetry (attachment counts, diffs, skipped sessions) appears in `polylogue status --json` alongside one-shot runs.

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

## Automation CLI

Polylogue now bundles helpers for both ad-hoc snippets and declarative configs:

```bash
# systemd service + timer (defaults to repo root + sys.executable)
python3 polylogue.py automation systemd --target codex --interval 15m --out /realm/data/chatlog/markdown/codex

# matching cron entry
python3 polylogue.py automation cron --target codex --schedule "*/20 * * * *" --out /realm/data/chatlog/markdown/codex

# inspect the underlying metadata (used by the NixOS module)
python3 polylogue.py automation describe --target codex
```

Available targets today: `codex`, `claude-code`, `drive-sync`, `gemini-render`, `chatgpt-import` (each target carries sensible defaults for collapse thresholds, HTML flags, and folder names that you can override with CLI/Nix options).

Use `--working-dir`, `--extra-arg`, `--collapse-threshold`, or `--html` to customise the generated command. Pipe the output to `tee` or redirect into your service/cron files as desired.

### NixOS module

When using Polylogue as a flake input, import `inputs.polylogue.nixosModules.polylogue` and describe timers declaratively. Each known target (`codex`, `claude-code`) exposes the same knobs as the CLI:

```nix
{
  services.polylogue = {
    enable = true;
    user = "polylogue";
    workingDir = "/var/lib/polylogue";
    stateDir = "/var/lib/polylogue/state";
    targets.codex = {
      enable = true;
      outputDir = "/realm/data/chatlog/markdown/codex";
      collapseThreshold = 18;
      html = true;
      timer.interval = "15m";
    };
  };
}
```

The module consumes the same `automation_targets.json` metadata used by the CLI, so upgrades automatically follow new target definitions.

## Tips

- Always use `--plain` for unattended runs so Polylogue skips interactive gum/skim prompts.
- Combine `--json` with your favourite log processor to detect failures.
- Watchers and scheduled jobs reuse the same slug/state tracking introduced in this release, so repeat exports and recurring syncs overwrite the correct Markdown without leaving duplicates.
