# Final CLI command floor

## Purpose

Make the public CLI small and coherent. The floor is `find QUERY then ACTION`, plus `import`, `config`, and `ops`.

## Public roots

- `polylogue`
- `polylogue find QUERY`
- `polylogue find QUERY then read|mark|analyze|remove|continue`
- `polylogue import PATH...`
- `polylogue config ...`
- `polylogue ops ...`

A quoted query may omit `find` only when the parser can treat it unambiguously as a query. Unquoted multi-token root magic should not exist.

## Action ownership

`read` owns terminal/browser/file/raw/message/export use cases.

`mark` owns tags, notes, pin/star/archive-style state, corrections, and feedback-like user overlays.

`analyze` owns stats, facets, cost, neighbors, diagnostics, insights, similarity, and correlation-style views.

`remove` owns destructive archive mutation and must use shared selection/cardinality guards.

`continue` survives only if it builds a real successor-session or work-packet bootstrap. If it only prints text, it is a read view.

## First slices

1. Inventory current root commands and map each to read, mark, analyze, remove, continue, import, config, ops, or devtools.
2. Add shared selection and cardinality path for actions.
3. Move old command behavior under verbs or devtools.
4. Remove public registrations and docs for old roots in the same PR that adds replacements.
5. Add command-tree tests proving stale roots are gone.
6. Add examples using demo archives.

## Acceptance

- Public help shows only the final roots and grammar.
- Old roots are absent from command registration, docs, completion, and tests.
- Read/mark/analyze/remove share one selection path.
- Ambiguity errors are typed and machine-readable.
- #2006 full query DSL ceiling is not required for this command-floor issue.
