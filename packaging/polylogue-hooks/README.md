# polylogue-hooks

Hook adapter for [Polylogue](https://github.com/sinity/polylogue) — captures
Claude Code and Codex session lifecycle events into the Polylogue daemon's
sidecar directory.

This distribution is a pure-stdlib Python implementation of the
`polylogue-hook` console script. It has **no runtime dependencies** and is safe
to install in minimal environments (e.g. inside the AI coding agent's own
runtime) without pulling the full `polylogue` archive substrate.

```bash
pip install polylogue-hooks

# Wire into Claude Code settings.json:
# "hooks": {
#   "SessionStart": [{"hooks": [{"type": "command", "command":
#     "polylogue-hook SessionStart --provider claude-code"}]}]
# }
```

Prefer `polylogue hooks install --harness claude-code|codex` when the main
Polylogue CLI is available; it performs the idempotent structured settings
merge and supplies the explicit provider flag needed by both harnesses.

The bash equivalent under `contrib/polylogue-hook` in the main repository
remains supported for operators who prefer a shell-only install path.

See [docs/hooks.md](https://github.com/sinity/polylogue/blob/master/docs/hooks.md)
for the full event catalog, sidecar layout, and configuration.
