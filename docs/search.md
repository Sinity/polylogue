# Search

Search finds sessions, messages and assertions by their words, by fragments of text, and by meaning. It is part of the query language, not a separate tool: a text term is one more condition, `similar:` is a ranking, and recall is an ordinary query the daemon builds. This page owns full-text and substring search, hits and ranking, the search plans, counts, index maintenance, semantic ranking, fusion, recall and fuzzy recall.

The language, units, defaults, paging and answers are in [query.md](query.md); `related:` rankings derived from sessions are in [measures.md](measures.md); the web search view is in [web.md](web.md).

The search index has no legitimately incomplete state: it converges with the archive, and lag shows as `up_to_date: false` in the answer, never as a different answer.

## Terms

- A bare word or `"phrase"` in a query matches the prose of messages (text blocks written by a person or a model). Several words must all occur in the same message; `word*` is a prefix; there is no stemming.
- `thinking:"…"` searches thinking blocks; `output:"…"` searches tool output (tool results and the bodies of Write, Edit and `apply_patch`). Words inside `assertions where …` search assertion values; in `documents`, document text.
- Compaction summaries are messages ([sessions.md](sessions.md#compaction-boundaries)) and are searched like prose.
- `command ~ "…"` and `path ~ "…"` are substring matches through a trigram index on tool calls (three characters or more; shorter values scan within the other conditions); `input ~ "…"` narrows calls already selected.
- `text ~ "…"` is a substring match on prose, through a trigram index over prose texts, for fragments a word tokenizer destroys (`7/10`, `v2.3.1`, `C++`, `--no-verify`). It filters; it does not rank.
- **Tokenization** folds case and diacritics, and folds letters with a stroke through one shared function (`ł→l`, `ø→o`, `đ→d`, `ħ→h`, `ı→i`, `ß→ss`). Anything that is not a letter or digit splits words.

## Hits and ranking

- Text is stored once per distinct content. A hit expands to every block using that text; then the defaults and the rest of the query apply. A subagent's hit makes its logical session match and names the subagent (`via: subagent`).
- Session rows of a text query carry `hits: [{message, session, via, written_by, at, snippet, marks}]`: the best `hits_per_row` (default 3) plus `more_hits`. `snippet` is about 12 words around the first match, shown as stored; `marks` are character offsets.
- When a term matches fewer than three prose messages, `notes` give counts and runnable queries for the same words in commands, tool output and thinking (`also in 14 tool commands: tool_calls where command ~ "nextest --partition"`).
- Every other condition restricts first; ranking orders only the survivors. Relevance is bm25 with no column weights; a session ranks by its best message; ties go to the more recent message, then identity. Relevance is the default order when a text term is present; `sort by` replaces it.

## The search leg

The compiler emits the non-text conditions as a filter plan plus a **search leg** (index, match expression, whether rank and order are needed), and the store executes it with one of three plans:

- **scope drives** when the non-text scope has at most 256 text blocks, or 64 times its block count is below the word's document frequency;
- **the index drives** otherwise: rank inside the index, filter by scope in rank order, and fetch rank batches of doubling size until the page is full, reading ties through;
- **recency drives** for a recency order with a frequent word: walk the newest sessions' blocks newest first, checking each text.

All three return the same rows in the same order. Rows are joined to blocks, messages and sessions only for the page and what fills it.

## Counts

`count` is the exact number of matching rows, computed by a separate count plan in the same read transaction and cached by question hash and read position. When counting would pass the deadline, `count` is `null` and `notes` give the query that counts with the deadline lifted; the rest of the answer is unaffected. A count is never estimated.

Counting ordinary rows applies the query's filters and stage selections without evaluating presentation fields or the final order. Fusion counts the union of selected keys; a list that only changes order adds no members. Ranking still runs when a selection such as `limit` depends on it, and missing semantic similarities are counted over the same selected rows.

## Index maintenance

The full-text indexes are contentless, keyed by text ID, with contentless delete. A text enters an index when its first block of that kind is written and leaves with its last. Ranking after re-parse retractions equals a fresh build's: whenever retracted rows exceed 10% of an index's rows, the writer rebuilds that index's statistics in the same pass. This is part of the ordinary write path, not a command.

## Semantic ranking

`similar:` orders a query's rows by meaning, using embeddings. Direction, 2026-01-24: "Not as a filter, but as a scoring criteria, to rank order items" (cc:1d6c0b33-94c7-4a2a-9957-347e5383f5c5:0bdc45a3-7aca-41a6-9b70-6f6fb086bbbe).

### Forms and meaning

- `similar:"text"` ranks by similarity to a text and needs a vector for that text. `similar:REF` (a session, message or assertion reference) ranks by a stored item's vector and needs no provider call; the seed itself is excluded.
- `similar:` ranks the whole query and appears once, among the top-level AND terms; anywhere else it is `invalid_query`.
- **A ranking, never a filter.** Conditions select; `similar:` orders the selection. Vectors are scanned exactly over the selected rows, with no approximate index and no global top-K cut-off.
- **Grain:** a message ranks by its vector, a block by its message's, a session by its best message (subagents included), an assertion by its own.
- Rows without a vector are never dropped: they follow the ranked rows, unranked, counted in `unknown.similar`; `count` stays exact.
- An answer that needs only membership (`| agg count`, `| group by …`) computes no similarity and makes no call; after a ranked selection (`| limit`), the ranking is computed.
- `related:REF` is a different ranking, derived from session structure ([measures.md](measures.md)); `fuse` combines the two.

### Model calls

- Model calls are off for now (default; open operator question); turning them on is a setting.
- **With calls off,** `similar:REF` works on every stored vector, and `similar:"text"` works when that text's vector is already stored. Otherwise the answer is `incomplete`: rows in the default order, marked unranked, with the note "semantic search for new text is off (model calls are off); `similar:` on a stored session or message still works". In a hybrid query the text side ranks alone and the note names the missing side. `embed status` and `embed preview` work; building is refused with the same reason.
- **With calls on,** a query text is embedded once and its vector stored, keyed by model and normalized text hash, so it is never paid for twice. New messages and assertions are embedded in the background at idle priority within a monthly budget, default up to $5 a month, with a preview and confirmation for paid providers (default; open operator question).
- **Providers:** a local OpenAI-compatible endpoint (default model `qwen3-embedding:0.6b`) and Voyage. Vectors from different models are compared only where the provider documents a shared space. Ruling, 2026-10-11: "These paid embeddings turn out to be worth few dollars IIRC, and even if it was more, it's not worth basing design of the projects on such considerations" (cc:e3e37f29-2490-47ae-89ef-fd40b6242e9a:07bbe1df-c4cd-4c59-b07e-540182d93fdd).
- **Provider failures:** a transient failure leaves the text without a vector until the next round; a permanent rejection is recorded once and listed in `status`; a hybrid query with a failed side runs the other side and is `incomplete`.

### What is embedded

- One vector input per message: the message's person and model prose, its text blocks in order joined by a blank line, NFC-normalized. An input past the model's limit is chunked at that limit and its best chunk ranks. Assertion values are embedded too.
- A vector's address is a function of the request only (domain, model and text, length-prefixed), so a vector once bought stays addressable whatever the index does. Vectors from earlier stores are imported by address, or by verified message identity where the address is missing; a vector that cannot be matched stays in its source file and is never dropped.
- `embeddings.db` is written only by the daemon and backed up with `archive.db`: `embedded(id, address, key, chunk, dims, vector, embedded_at)`, `query_vector(key, text_hash, vector, embedded_at)`, `embed_failure(address, key, error, failed_at)`. `index.db` holds `message_embed_input(message, address)`, written with the message.
- **Scan:** the daemon holds int8 copies of the configured model's vectors, scores the selected rows exactly in int8, then re-scores the top 200 in f32 from the stored rows.

## Fusion

A query with text terms and `similar:` fuses its rankings by reciprocal rank: each row scores Σ 1/(60 + rank) over the lists that rank it; ties go to recency, then identity. Fusion never changes membership or `count`. Each row carries `found_in`; `--why` adds each list's rank and contribution. A list that could not run is omitted and named, and the answer is `incomplete`. Combining queries with `| union` and `| fuse` uses the same fusion ([query.md](query.md)).

## Recall

`recall TEXT [--cwd DIR] [--keep]` (MCP `recall`) answers "what do I already know that bears on this", across every repository unless the query scopes it. It is one ordinary query the daemon builds and shows in the answer, paged by its cursor. It fuses five lists:

1. prose messages containing any of the hint's words, by bm25 over an OR, so more words rank higher;
2. tool output, the same way;
3. assertions (claims, decisions, findings, lessons) whose value has any of the words;
4. `similar:` to the whole hint, when a vector for it exists;
5. sessions in the caller's directory, as a `fuse` list.

Words found in more than 20% of texts are dropped from the OR lists when the hint has other words, and `notes` name them. Hits are grouped by session in fused order, with snippet, `found_in`, and the attached assertions and judgments; text not written by the operator is labelled with its author. A budget cuts what is shown, never what is ranked. When exact search finds nothing and the semantic list did not run, the answer says so: "semantic search is off, so related work under other wording may exist". Recall for agents (in briefs and packs) is in [agents.md](agents.md#packs).

### Fuzzy recall

Fuzzy recall finds a half-remembered conversation by explicit, explained tolerances:

- `word~` (or `--fuzzy` on recall) also matches index words within edit distance 1 (4–6 letters) or 2 (7 or more), and words sharing its first four letters, from the daemon's vocabulary. Exact matches outrank approximate ones; `notes` list each substitution (`backpu → backup (312 texts)`).
- `around:<date|month|"spring 2026">` adds closeness to a time window as one more fusion list; `since:` and `until:` stay filters.
- `similar:` on the hint, when vectors exist.

Field names and values never match approximately; a mistyped field is an error with a suggestion.

## Laws

- **search-plans-agree**: the three search plans return identical rows in identical order for the same query on a generated archive.
- **folding-and-fragments**: `lodz` finds `Łódź`; `text ~ "7/10"` finds the literal; the Morrow `chrono` search returns its 3 sessions and 8 hits, with subagent hits under their logical session.
- **retraction-ranking**: ranking after retracting half the texts equals a fresh build's.
- **count-exact-or-null**: `count` is exact, or `null` with a counting query in `notes`; never an estimate.
- **similar-off-no-call**: with model calls off, `similar:"x"` is `incomplete` with the note and makes no network request (a provider that fails on any call proves it); `similar:"x" | agg count` makes no call.
- **similar-ranks-never-filters**: `similar:session:<id> repo:Y` ranks only repository Y's sessions and returns every one of them, unranked ones after ranked.
- **fusion-membership**: a three-list fusion matches an independent evaluator; fusion never changes membership or `count`; chained unions fuse once.
- **recall-ranks-partial-matches**: on Morrow, `weekday arithmetic` ranks subagent #3's answer first, then #5 and #6; a five-word hint where no message has all five still returns ranked hits; a 100-hit recall pages through every hit once.
- **fuzzy-recall**: `"backpu~ tools" around:2026-04` finds the backup-tools fixture, lists its substitutions, and ranks exact matches above approximate ones.
