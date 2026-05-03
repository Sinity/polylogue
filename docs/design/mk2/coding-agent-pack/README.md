# Polylogue MK2 implementation pack

This pack turns the MK2 design artifact into instructions a coding agent can execute without regressing Polylogue's current CLI or creating UI sprawl.

Use order:

1. Paste `01-github-issue-feat-surfaces.md` into a new GitHub issue.
2. Give `02-coding-agent-prompt.md` to the coding agent for the first implementation planning pass.
3. Use `03-web-ui-functional-aesthetic-spec.md` and `06-design-tokens.json` as the web UI target.
4. Use `04-cli-enhancement-spec.md` as the CLI completion/table/fuzzy selector target.
5. Use `05-daemon-api-contract.json` as the local API/status contract.
6. Use `07-verification-and-visual-evidence.md` to prevent the UI from becoming unverifiable decoration.
7. Use `08-mk2-corrections.md` to keep the agent from copying MK2's wrong details literally.

Non-negotiable summary:

- Preserve the query-first CLI: `polylogue [query] [filters] [verb]`.
- Do not add `polylogue ui`.
- Do not keep compatibility aliases or deprecated names.
- Do not use `--json`; canonical machine output is `--format json`.
- Split long-running service/protocol surfaces away from the interactive client:
  - `polylogue` = query/read/archive CLI.
  - `polylogued` = live ingestion + browser receiver + local API + web reader.
  - `polylogue-mcp` = MCP stdio bridge.
  - `devtools` = source-checkout repo control plane.
- Keep `messages` and `raw` as real read surfaces. They are not accidental sprawl.
- Do not turn `insights` into a prettier `products` island. Derived data belongs inline with conversations/sessions, plus concrete aggregate nouns if useful.
- Browser capture is automatic by default with prominent visible state and disable controls, unless a later explicit decision reverses this.
- Web UI, TUI, CLI, MCP, and static/export surfaces must consume shared query/read/status/derived contracts.
