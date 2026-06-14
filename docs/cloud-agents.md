# Cloud-Agent Setup (Claude Code Web / Codex Cloud)

Polylogue is one of the easier projects to run inside a managed cloud-agent
sandbox: it is pure Python, depends only on pre-built wheels, has no native
secrets requirement, and all on-disk paths are overridable via
`POLYLOGUE_ARCHIVE_ROOT`.

This page is the operator checklist. The repo-side wiring lives in:

- [`.claude/settings.json`](../.claude/settings.json) — env + Bash allowlist
- [`.claude/setup.sh`](../.claude/setup.sh) — sandbox bootstrap
- [`CLAUDE.md`](../CLAUDE.md) — "Cloud lane" section (agent-facing rules,
  rendered into `AGENTS.md` by `devtools render-agents`)

See also: `2026-05-27_Build_Plan.md` §D (Cloud agent enablement).

## Command surface

| Command                                       | In cloud? | Notes                                                       |
| --------------------------------------------- | --------- | ----------------------------------------------------------- |
| `uv run pytest tests/unit -q`                 | yes       | Fast smoke; no `/realm` access needed.                      |
| `uv run pytest tests/property -q`             | yes       | `HYPOTHESIS_PROFILE=ci` keeps budgets bounded.              |
| `uv run ruff check polylogue tests`           | yes       | Linter.                                                     |
| `uv run ruff format --check polylogue tests`  | yes       | Format gate (matches CI `lint` job).                        |
| `uv run mypy polylogue`                       | yes       | Type check.                                                 |
| `uv run devtools verify`                      | yes       | Slow. Prefer scoped invocations during iteration.           |
| `uv run devtools render-all --check`          | yes       | Generated-file drift check (also runs in CI).               |
| `polylogued run --no-api --no-watch --no-browser-capture` | yes\*     | \*Only against synthetic fixtures in `/tmp/polylogue-archive`. `--no-api` alone is insufficient; you must also disable live watch and browser capture for a truly inert sandbox. |
| Real archive imports (`~/.claude/projects/…`) | NO        | Never upload personal corpus into a managed sandbox.        |
| Browser-capture flows                         | NO        | Needs interactive cookies; relocated to ethereal host.      |
| Any `/realm/data/...` path                    | NO        | Not mounted in cloud sandboxes; would resolve to nothing.   |

## Privacy and plan tier

Cloud-agent sandbox content inherits the data-handling tier of the account
under which the agent runs. The repo cannot enforce this — it is an operator
responsibility. Confirm tier **before** enabling cloud-agent lanes on
sensitive repos.

**Claude Code Web (Anthropic):**

1. Open <https://console.anthropic.com/settings/privacy>.
2. Confirm "Help improve Claude" is **off** for the account used for Claude
   Code Web. Pro/Max consumer tier defaults to opt-IN — flip explicitly.
3. ZDR (Zero Data Retention) disables Claude Code Web entirely. Do not enable
   on a ZDR-only account.

**Codex Cloud (OpenAI):**

1. ChatGPT → Settings → Data Controls → "Improve the model for everyone" → off.
2. On a Team/Enterprise workspace, Codex Cloud inherits the workspace data
   controls — verify the workspace setting too.
3. Business/Enterprise plans: data is excluded from training by default.

Document which account/tier is in use somewhere durable (issue, ops notebook)
so future setup work does not accidentally regress.

## Codex Cloud environment hints

Codex Cloud reads `AGENTS.md` for repo-scoped guidance, but its sandbox env
vars are configured outside the repo, in the Codex Cloud environment-settings
panel. Mirror the values from `.claude/settings.json` there:

| Variable                    | Value                     | Why                                               |
| --------------------------- | ------------------------- | ------------------------------------------------- |
| `POLYLOGUE_ARCHIVE_ROOT`    | `/tmp/polylogue-archive`  | Keeps writes inside the sandbox tmpfs.            |
| `POLYLOGUE_FORCE_PLAIN`     | `1`                       | Disables Rich pretty-printers (matches CI).       |
| `HYPOTHESIS_PROFILE`        | `ci`                      | Bounded property-test budget.                     |

If Codex Cloud supports a per-environment bootstrap script, point it at
`.claude/setup.sh` (or paste its contents) so `uv` is installed and dev deps
are synced before the first command runs.

## What to do if the sandbox is broken

1. Re-run `bash .claude/setup.sh` (idempotent).
2. Check `uv --version` is on `PATH` — sandbox restarts sometimes drop
   `~/.local/bin` from `PATH`; the setup script re-prepends it.
3. If `uv sync --frozen` fails, the lockfile is out of sync with
   `pyproject.toml` — fix locally and push, not from the sandbox.
4. If anything tries to read `/realm/data/...`, that is a bug — surface it; the
   cloud lane must not depend on the data lake.
