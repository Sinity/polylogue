Title: "[testdiet 13] Output and schema injection safety"

Job ID: `testdiet-13`
Result ZIP: `testdiet-13-output-injection-safety-r01.zip`

## Mission

Implement production-route safety survivors for attacker-controlled provider,
schema, annotation, saved-query, and generated-output tokens crossing
Polylogue's CLI, JSON, HTTP, MCP, rendered docs/schema, and archive paths.
Prove that values remain data rather than becoming SQL, shell, path, markup,
terminal-control, or schema/code authority; outputs remain parseable; and
invalid identifiers fail through the intended typed boundary.

Start from actual ingress and output call sites, not a synthetic bad-string
catalog. Build a compact malicious-token corpus covering quotes, separators,
Unicode confusables/control characters, traversal, terminal escapes, markup,
SQL-shaped strings, and oversized values where boundedness is contractual.
Exercise real serializers/renderers/storage parameterization and independent
parse/readback or side-effect oracles.

Name a parameterization, escaping, path-normalization, schema-enum, or output
framing mutation. Retain narrow security diagnostics even when other examples
are dominated. Do not build a second sanitizer registry or claim general
security from string-absence assertions.
