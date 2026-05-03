# Polylogue CLI enhancement spec

## Principle

The CLI is already a powerful query-first shell surface. Do not regress it into a generic command tree.

Primary grammar:

```text
polylogue [query terms] [root filters] [verb]
```

Verbs over selected conversations:

```text
list
count
stats
show
open
bulk-export
delete
messages
raw
select
```

## Dynamic completion

Completion is mandatory, not polish. Without it, the CLI's object space is too large.

### Descriptor fields

Extend the query/command descriptor layer with fields like:

```python
@dataclass(frozen=True)
class CompletionSpec:
    kind: Literal["static", "archive", "daemon", "computed"]
    source: str                 # provider, repo, cwd_prefix, tag, tool, conversation_id, etc.
    label_template: str         # visible value
    description_template: str   # right-side completion help
    preview_fields: tuple[str, ...]
    cost_class: Literal["cheap", "bounded", "expensive"]
    max_results: int = 50
    cache_ttl_s: float | None = 5.0
```

Query descriptors should expose:

- CLI flags;
- MCP/API field names;
- strict validator;
- SQL/runtime mapping;
- doc text;
- completion source;
- facet eligibility;
- display formatting.

### Completion targets

Implement or verify completion for:

- providers;
- repos;
- cwd prefixes;
- tags;
- tools/actions;
- message types;
- has flags;
- conversation IDs;
- saved views;
- doctor scopes;
- run stages/sources if still public;
- derived view names if exposed;
- output formats (`json`, maybe `text`/`table` if supported).

Completions must be fast and bounded. Do not run expensive FTS readiness or exact counts on every shell completion.

## Fuzzy selector

`polylogue select` should be the CLI bridge between shell and web reader.

### Conversation selection

```text
polylogue [query] [filters] select
polylogue select conversation --recent
polylogue select conversation --query "fts trigger" --repo polylogue
```

Default action in TTY: interactive fuzzy picker.

Useful flags:

```text
--print id|url|json|command
--open web|terminal|raw
--multi
--limit N
--preview/--no-preview
```

Examples:

```bash
polylogue messages "$(polylogue "fts trigger" select --print id)" --limit 50
polylogue "daemon" --repo polylogue select --open web
polylogue select repo --print value
```

### Entity selection

```text
polylogue select repo
polylogue select cwd
polylogue select provider
polylogue select tag
polylogue select tool
polylogue select source
polylogue select saved-view
polylogue select doctor-target
```

Use fzf if available/configured, fallback to pure Python. Preview content must reuse existing renderers.

## Rich rendering

Use semantic, restrained colors. Respect plain mode/non-TTY.

### Root stats

`polylogue` with no args should show compact archive summary:

```text
polylogue · 412 conversations · 4 providers · 38.2k messages · $74.10 estimated
claude-code  ████████████░░░░░░░  248  live ok
chatgpt      ████░░░░░░░░░░░░░░░   72  captured
claude-ai    ███░░░░░░░░░░░░░░░░   58
codex        ██░░░░░░░░░░░░░░░░░   34

next: polylogue --latest show · polylogue --provider claude-code list
```

### `list` table

Columns:

```text
id  provider  repo  cwd  age  msgs  tok  cost  flags  title
```

Rules:

- show short IDs but copy/open full IDs in JSON;
- provider badges use stable provider colors;
- repo/cwd truncation preserves distinguishing suffix;
- flags are semantic chips: tool, thinking, paste, raw, live, captured, starred, pinned;
- `--format json` emits a stable typed envelope or existing canonical payload, not table-shaped strings.

### `show`

`show` should be clear about whether it is header/overview or full render. Recommended:

- header/overview by default;
- show next commands: `messages`, `raw`, `open`;
- include derived/session facts inline;
- do not accidentally hydrate giant transcript unless user requests it.

### `messages`

Human format:

```text
#01 user · 14:02:11 · 84 tok · paste
  ...

#02 assistant · 14:02:18 · 431 tok · thinking, tool
  ...

next: polylogue messages <id> --limit 50 --offset 50
```

### `raw`

Help text must say raw archive artifact payloads. Do not claim true provider-event paging unless implemented.

## Colors

Provider colors are stable. Status colors are semantic. No rainbow.

- healthy/live: green/teal;
- active/current: cyan/blue;
- warning/stale: amber;
- error/blocking: red;
- quiet/disabled: gray;
- role colors subtle, not neon.

## Tests

- help snapshot for root query-first grammar;
- no `--json` in changed help surfaces;
- completion source tests with synthetic archive;
- fuzzy selector noninteractive `--print id` tests;
- TTY rendering smoke with plain-mode fallback;
- list JSON parity with API endpoint;
- `messages` pagination parity;
- `raw` output contract.
