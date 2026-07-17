---
created: 2026-07-16
purpose: Top-down audit packet for provider detection, parsing, and normalization tests
status: survey-ready
project: polylogue
---

# Source normalization

## Scope and scale

`tests/unit/sources` contains 70 Python files, about 27.7k nonblank lines, and
roughly 1,162 test/class declarations. Organize it by public normalization law,
not parser class:

1. provider detection tightness and ambiguous-shape precedence;
2. wire artifact → normalized session/message/block facts;
3. streaming versus ordinary parsing equivalence;
4. bundle/split/nesting equivalence;
5. incremental live-file append and cursor/revision semantics;
6. malformed/null/encoding/path hostility and bounded failure;
7. provider-specific semantics that genuinely cannot be shared.

## Generator portfolio

The schema generator should emit wire artifacts plus planted normalized facts.
Each provider needs:

- a semantic minimum carrying every supported relation;
- schema-derived arbitrary variants for crashlessness and null boundaries;
- ambiguity witnesses whose shape could be claimed by two detectors;
- bundle/split and ordering transformations;
- streaming/tail cases, including partial final records and later append;
- minimized historical provider oddities.

Distribution profiles may be regenerated from privacy-safe archive statistics,
but generator profiles must record schema/distribution provenance. Fixed seeds
make PR artifacts reproducible; scheduled campaigns should vary seeds and save
any failing seed plus minimized witness so the fixed portfolio does not become
the only explored path.

## Stronger laws

- parsing the same logical material as one bundle, split files, or reordered
  independent records yields identical normalized facts;
- streaming and non-streaming paths yield the same content hashes/facts;
- the tightest valid detector wins for every generated ambiguous shape;
- arbitrary schema-valid data never crashes, while invalid data fails with the
  correct bounded diagnostic and preserves already committed material;
- appending a complete record advances exactly once; an incomplete tail is
  neither lost nor duplicated after restart;
- normalized facts survive acquire → parse → store → read, rather than being
  asserted only against a parser return object.

The independent oracle is generated alongside the wire artifact from a small
provider-neutral fact blueprint. It must not call provider parser helpers to
derive expected results.

## Subtraction strategy

Retain compact provider-specific examples when they document a unique wire
quirk. Replace repeated field-by-field parser examples and private-helper/mock
assertions with generated blueprint laws where the oracle and diagnostics stay
clear. Preserve the strong existing arbitrary-input/null/security tests.

Do not rewrite all provider catalogs at once. Convert one provider family plus
the cross-provider detector ambiguity slice, measure gross/add/net LOC, then
decide whether the pattern actually reduces complexity.
