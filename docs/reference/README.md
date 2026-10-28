# Generated reference

The designed `cargo xtask docs` generator will write the pages below from the
declarations in the code (the command table, the settings catalog, the origin,
query-unit, derivation, assertion-kind and Sinex-kind declarations), through
the same code that serves them at run time. They will be committed, and a
golden test will compare them with the rendered declarations. Change the
declaration and regenerate these pages instead of editing them by hand.

The generator is not implemented yet. Use `polylogue --help`,
`polylogue explain` and MCP tool schemas for the current commands and fields.
The current derived-data reference is
[`../derived.md`](../derived.md), rendered from the declarations and checked
by a golden test. The embedded [`../agent-manual.md`](../agent-manual.md)
receives the current generated command reference at run time.

## Planned pages

These filenames are planned destinations, not existing pages.

| Page | Generated from |
|---|---|
| `commands.md` | the command table: CLI form, MCP tool and schema, HTTP route, latency class, failure kinds |
| `settings.md` | the typed configuration, the changeable settings catalog, environment variables, Home Manager and NixOS module options |
| `origins.md` | origin declarations: locations, arrival, recognition, reading version |
| `query-units.md` | query units and fields, with each field's meaning and unknown rule |
| `derived.md` | derivation declarations: construct, unit, denominator, validity, evidence |
| `assertion-kinds.md` | assertion kinds: key, value shape, link roles |
| `sinex-kinds.md` | the `polylogue.*` kinds Polylogue registers with Sinex |
