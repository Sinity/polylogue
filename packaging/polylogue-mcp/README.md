# polylogue-mcp

MCP server entrypoint for the [Polylogue](https://github.com/sinity/polylogue)
AI chat archive.

This distribution is a thin wrapper that pins the matching `polylogue` release
and re-exports the `polylogue-mcp` console script. Install it when you want the
MCP stdio bridge installed under a stable, narrowly-scoped name without typing
out the full `polylogue` package name.

```bash
pip install polylogue-mcp
polylogue-mcp --help
```

Note: the runtime dependency closure is identical to `pip install polylogue`
today, because the MCP server reuses the archive substrate (storage, search,
repository). Future releases may split runtime dependencies further; the
PyPI name `polylogue-mcp` is reserved for that direction.

See the [main repository](https://github.com/sinity/polylogue) and the
[architecture overview](https://github.com/sinity/polylogue/blob/master/docs/architecture.md#3-surfaces)
for the MCP server's contract surface and tool inventory.
