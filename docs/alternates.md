# Alternates

Alternates are written with `note --as` ([annotations.md](annotations.md#note)) and follow the revision and withdrawal rules of [annotations.md](annotations.md#current-revisions), with `base` in place of `revises`. Reading them is [reading.md](reading.md#alternates).

An alternate is another rendering of content: a translation, transliteration, simplification or summary. It is its own record, not an annotation, and annotations may be about it. Its purpose is that agents can render anything in the archive for the reader, such as translating a session the operator wants to read in Polish. The unit is `alternates`, the reference `alternate:<root>[@<rev>][#<n>]` ([references.md](references.md)).

```sql
CREATE TABLE alternate (
  id          TEXT PRIMARY KEY,
  root        TEXT NOT NULL,       -- first revision of this alternate
  base        TEXT,                -- the revision this one revises (same root); NULL for a first revision
  source      TEXT NOT NULL,       -- exact reference: session, message, range, block, document, span, annotation@rev, alternate@rev
  source_hash BLOB NOT NULL,       -- BLAKE3 of the source text at that revision
  kind        TEXT NOT NULL,       -- translation | transliteration | simplification | summary
  from_lang   TEXT, to_lang TEXT,  -- BCP 47
  author      TEXT NOT NULL, via TEXT NOT NULL,
  method      TEXT,                -- free text or a reference: model, prompt, rubric used
  machine     TEXT NOT NULL, created_at TEXT NOT NULL, commit_seq INTEGER NOT NULL  -- created_at is RFC 3339
) STRICT;

CREATE TABLE alternate_part (
  alternate     TEXT NOT NULL REFERENCES alternate(id),
  position    INTEGER NOT NULL,    -- order in the alternate
  source      TEXT NOT NULL,       -- the source child it renders: a message, block or span inside the alternate's `source`
  source_hash BLOB NOT NULL,
  relation    TEXT NOT NULL,       -- translates | transliterates | simplifies | summarizes | omits | expands | reorders
  body        BLOB NOT NULL,       -- zstd text
  PRIMARY KEY (alternate, position)
) STRICT, WITHOUT ROWID;
```

- **Writing.** Agents write alternates with `note TEXT --as translation:LANG|transliteration:LANG|simplification|summary`: on a message the text is the alternate; on a session or a page `--parts FILE|-` gives one part per line as JSON (`{of, relation, text}`), with `''` as the text. `edit` on an alternate writes a revision (`base`); `withdraw` withdraws it. Polylogue never calls a model to produce or judge content: alternates, labels and summaries are written by agents in their own harnesses.
- **Hierarchical.** An alternate whose source is a session consists of parts, one per rendered block, each naming its original block in `of`; its messages are the groups of parts by their original message. A message alternate is the same at message grain. A summary is one or more parts whose `of` is what the producing agent chose (the session, a range of messages, a block), with relation `summarizes`. Reading an alternate renders the original's structure with parts swapped in; unmapped children show "untranslated" and `omits` parts show "omitted".
- **Several parts for one child.** Several parts may name the same `of`, for example an expansion. They render in ascending `n`, joined by two line feeds. A group renders as omitted only when every part is `omits`; a mixed group keeps its text and is marked partly omitted. Two current alternates of one source are an ambiguity, never concatenated.
- **Bound to its source revision.** Correcting an alternate writes a revision with the same source. When the source changes (`source_hash` or an `of_hash` differs), the alternate shows "alternate of an earlier text"; translating again is a new alternate. Finer alignment inside one part is expressed by span references in annotations about the part.
- **Annotations get alternates too.** The source may be `annotation:X@R`. An agent translating an annotated session writes one session alternate plus one alternate per annotation it renders. On an alternate's reading, annotations about the alternate itself are shown; for an annotation about an original child, its current alternate in the reading's language is shown when one exists, otherwise a count with a link ("2 notes in the original"). The original's text is never shown as if it were about the alternate.
- **Alternates never count** as support, votes or judgment.
- **Enabled alternates are a setting** that every reading follows: `set reading.alternates translation:pl` holds an ordered list of keys. Every reader (CLI `read`, the web reader, `read --to DIR`) renders each part swapped for the first enabled, current alternate whose source has not changed, visibly marked, with the original one action away; `read --alternate KEY|none` overrides it for one reading and `--beside` shows the alternate and its original aligned ([reading.md](reading.md#alternates)).
- **Pages.** A page read follows the enabled alternates for every text it shows and every text it includes, marked where they replace the original. Translating a page is one request on the page selection: `polylogue page:flaky-ci note '' --as translation:pl --parts parts.jsonl`, each part's `of` naming an annotation the page shows. A page in two languages is one page with alternates, never two pages ([pages.md](pages.md#reading)).
- **Search.** Alternate text is indexed separately and matches only with `alternate:true` or `lang:<code>`. A message's detected language is a derived field (`message.lang`), not an alternate.
