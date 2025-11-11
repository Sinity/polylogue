# Automation Recipes

Polylogue ships with interactive workflows by default, but you can keep local archives up to date without manual effort. This page collects the recommended approaches.

## Real-Time Watchers

The CLI includes long-running watchers that mirror local Codex and Claude Code session stores as soon as files change:

```bash
# Watch ~/.codex/sessions/ and sync into the standard archive
python3 polylogue.py watch codex --out ~/polylogue-data/codex

# Watch ~/.claude/projects/ for Claude Code sessions
python3 polylogue.py watch claude-code --out ~/polylogue-data/claude-code
```

Watchers run an initial full sync, then rerun whenever `.jsonl` session files change inside the source tree. They reuse your configured collapse thresholds and HTML defaults; pass `--collapse-threshold` or `--html` to override. The `--debounce` flag (default: 2 seconds) throttles rapid bursts of file events so a single editing session doesn’t thrash the pipeline. Every run flows through the same registrar/pipeline stack as one-shot commands, so attachment counts, Drive retries, and diffs show up both in `polylogue status --json` and in the structured `polylogue_run` JSON lines that Polylogue now emits on stderr (set `POLYLOGUE_RUN_LOG=0` to silence those logs when needed).

## Systemd Timer Template

For environments that prefer scheduled jobs, create a systemd service/timer pair. The example below runs the Codex sync every 10 minutes in a user session:

`~/.config/systemd/user/polylogue-sync-codex.service`
```ini
[Unit]
Description=Polylogue Codex sync

[Service]
Type=oneshot
Environment=XDG_STATE_HOME=%h/.local/state
WorkingDirectory=/srv/polylogue
ExecStart=%h/.nix-profile/bin/python3 polylogue.py sync codex --plain --json
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

Enable the timer with `systemctl --user enable --now polylogue-sync-codex.timer`. Duplicate the pair with different names for Claude Code or Drive syncs (for Drive use `sync drive` and add the folder flags).

## Cron Alternative

When systemd is unavailable, a simple cron entry can invoke the same non-interactive commands:

```
*/30 * * * * XDG_STATE_HOME="$HOME/.local/state" cd /srv/polylogue && python3 polylogue.py sync codex --plain --json >> "$HOME/.cache/polylogue-sync.log" 2>&1
```

Combine cron with `polylogue status --json` to monitor recent runs or parse the generated log file. When you need a rolling JSON artifact for other tooling, append `&& python3 polylogue.py status --plain --dump "$HOME/.cache/polylogue-status.json" --dump-limit 50 --dump-only` after the sync command (the automation CLI can now do this for you automatically via `--status-log`).

## Automation CLI

Polylogue now bundles helpers for both ad-hoc snippets and declarative configs:

```bash
# systemd service + timer (defaults to repo root + sys.executable)
python3 polylogue.py automation systemd --target codex --interval 15m --out ~/polylogue-data/codex

# matching cron entry
python3 polylogue.py automation cron --target codex --schedule "*/20 * * * *" --out ~/polylogue-data/codex

# inspect the underlying metadata (used by the NixOS module)
python3 polylogue.py automation describe --target codex
```

Available targets today: `codex`, `claude-code`, `drive-sync`, `gemini-render`, `chatgpt-import` (each target carries sensible defaults for collapse thresholds, HTML flags, and folder names that you can override with CLI/Nix options). ChatGPT/Claude bundles can also piggyback on the generic sync automation pattern—point your job at `polylogue sync chatgpt|claude --plain --json` (or `polylogue watch chatgpt|claude` for continuous monitoring) and keep exporting ZIPs into `$XDG_DATA_HOME/polylogue/exports/{chatgpt,claude}`.

Use `--working-dir`, `--extra-arg`, `--collapse-threshold`, or `--html on|off|auto` to customise the generated command. Because the automation CLI reuses the same flag builders as the main commands, the snippets exactly mirror how `--html` behaves elsewhere (`--html` alone implies `on`). Pipe the output to `tee` or redirect into your service/cron files as desired. New `--status-log /path/to/status.json` and `--status-limit N` options tell the generated service/cron entries to run `polylogue status --dump-only ...` immediately after each sync so you always have a fresh JSON view of recent runs on disk without extra scripting, and the matching `--status-summary /path/to/metrics.json` (plus optional `--status-summary-providers drive,codex`) flag writes aggregated summaries using `polylogue status --summary … --summary-only` so dashboards never have to parse tables.

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
      outputDir = "~/polylogue-data/codex";
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
