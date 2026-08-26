# Marker sigil collision scan

The marker sigil is `::` for line markers (`::kind(args): body`) and `[[kind:
body]]` for inline markers. The line form is escaped as `\::`.

On 2026-08-26 the read-only live-corpus scan was run against the local
Polylogue archive roots available on this host:

```sh
rg -n --hidden --glob '!*.db-shm' --glob '!*.db-wal' \
  '(^|[^\\]):{2}[a-z][a-z0-9_-]*:' /realm/state /realm/data/captures
```

Result: one textual candidate was found in the scanned hook records (exit
status 0, one matching record). It is prose embedded in captured tool data,
not a standalone authored marker; this is the recorded collision that rules
out treating an unanchored `::kind:` prefix as structure. The repository
itself also contains fixture and documentation examples by design.

The parser only recognizes a line-anchored marker, ignores fenced Markdown,
and requires the explicit backslash escape for prose beginning with `::`.
