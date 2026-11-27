# Search

Search matches sessions, messages and annotations by their words, by fragments of text, and by meaning. It is part of the query language, not a separate tool: a single plain word on the command line is a search, a text term is one more condition, `similar:` is a ranking, and the brief of a search is a plain query the daemon builds. This page owns the plain word, full-text and substring search, hits and ranking, the search plans, counts, index maintenance, semantic ranking and `embed`, fusion, and the lists the brief of a search fuses.

The language, units, defaults, paging and replies are in [query.md](query.md); `related:` rankings derived from sessions are in [measures.md](measures.md); the web search view is in [web-views.md](web-views.md#search-searchq).

The search index has no legitimately incomplete state: it converges with the archive, and lag appears in the reply's `freshness` report with a degraded outcome, never as a different reply.

## The plain word

A single plain word as the first argument of a line is a search for that word ([interfaces.md](interfaces.md#the-first-word), rule 9): `polylogue chrono` searches prose, tool input and tool output for "chrono", each as [Terms](#terms) describes, and ranks the sessions by relevance fused with recency ([Fusion](#fusion)). The footer says `searched the word "chrono"` and names the ranking; `--explain` and the reply's `read_as` show the query it ran. A plain word is a selection like any other, so verbs follow it: `polylogue chrono read` reads the first match starting at its first hit, and `polylogue chrono brief` is the brief of that search ([Recall](#recall)). When pages match the word, one line above the hits names them with the line that lists them: `2 pages mention "chrono": polylogue 'pages where "chrono"'` ([pages.md](pages.md#finding-pages)).

Command, verb and unit names and saved query names are not searched as plain words. Searching for one takes inner quotes, which make it a phrase term in a query: `polylogue '"status"'` searches prose for the word "status".

## Terms

- A bare word or `"phrase"` in a query matches prose: text blocks written by a person or a model, text whose author is unknown, and compaction summaries. Other harness prompts are excluded. Several words must all occur in the same message; `word*` is a prefix; there is no stemming.
- `thinking:"…"` searches thinking blocks; `output:"…"` searches tool output (tool results and the bodies of Write, Edit and `apply_patch`). Words inside `annotations where …` search annotation values; in `documents`, document text.
- Compaction summaries are messages ([sessions.md](sessions.md#compaction-boundaries)) and are searched like prose.
- `command ~ "…"` and `path ~ "…"` are substring matches through a trigram index on tool calls (three characters or more; shorter values scan within the other conditions); `input ~ "…"` narrows calls already selected.
- `text ~ "…"` is a substring match on prose, through a trigram index over prose texts, for fragments a word tokenizer destroys (`7/10`, `v2.3.1`, `C++`, `--no-verify`). It filters; it does not rank.
- **Tokenization** folds case and diacritics, and folds letters with a stroke through one shared function (`ł→l`, `ø→o`, `đ→d`, `ħ→h`, `ı→i`, `ß→ss`). Anything that is not a letter, a digit or a private-use character splits words.

## Hits and ranking

- Text is stored once per distinct content. A hit expands to every block using that text; then the defaults and the rest of the query apply. A subagent's hit makes its logical session match and names the subagent (`via: subagent`).
- Session rows of a text query carry `hits: [{message, session, via, written_by, at, snippet, marks}]`: the best `hits_per_row` (default 3) plus `more_hits`. `snippet` is about 12 words around the first match, shown as stored and cut to at most 400 characters (a word longer than that is cut itself); `marks` are character offsets.
- When a term matches fewer than three prose messages, `notes` give counts and runnable queries for the same words in commands, tool output and thinking (`also in 14 tool commands: tool_calls where command ~ "nextest --partition"`).
- Every other condition restricts first; ranking orders only the survivors. The words that rank are the top-level prose words when the query has any, else its top-level `output:` terms; words under OR or NOT, `thinking:` and substring terms filter without ranking. Relevance is bm25 with no column weights; a session ranks by its best message; ties go to the more recent message, then identity. Relevance is the default order when a term that ranks is present (a top-level prose word, else a top-level `output:` term); a query whose words are all under OR or NOT is ordered by recency; `sort by` replaces it. A plain word is the exception: it fuses this relevance order with recency ([Fusion](#fusion)). The chosen source also supplies the snippet when it searches the same terms; a different snippet term set selects its own best source.

## The search leg

The compiler emits the non-text conditions as a filter plan plus a **search leg** (index, match expression, whether rank and order are needed). A word matching at most 256 eligible block occurrences uses the index directly. For larger index legs, the store chooses among three plans:

- **scope drives** when the non-text scope has at most 256 text blocks, or 64 times its block count is below the word's document frequency;
- **the index drives** otherwise: rank inside the index, filter by scope in rank order, and fetch rank batches of doubling size until the page is full, reading ties through;
- **recency drives** for a recency order with a frequent word: walk the newest sessions' blocks newest first, checking each text.

All three return the same rows in the same order. Rows are joined to blocks, messages and sessions only for the page and what fills it. The index prefix is combined into each parent row's best score before the cursor applies, so weaker later hits cannot reintroduce an earlier row. Prefix aggregation uses bounded temporary storage; cursors carry only order values.

For this cost decision, the word's frequency counts eligible block occurrences before the non-text scope, including every block that shares an indexed text. The full-text vocabulary counts distinct stored texts; it can make a body repeated across thousands of messages look rare and underestimate expansion work. The store first checks for a rare index leg, so deciding a cheap search does not require walking a large scope with few eligible blocks. Otherwise it probes scope blocks and matching block occurrences together, stopping as soon as the 256-block or 64-to-one comparison is decided. The probe stores no prefix and changes only the physical plan.

When scope drives a batch of snippets, eligible block identities are computed and stored once for the text probes and source selection. The planning probe streams those identities so deciding the plan does not first read the whole scope.

For a frequent word in a direct newest-first session or message sort, candidate parent rows stream in the requested time order. Bounded current batches use the same scoped best-source selection. Reading continues until the page fills or the candidates end; a batch size never limits the reply. Shared descendants cannot return a parent row outside the current batch or repeat one from an earlier batch. Only the returned order values enter the continuation.

## Counts

`count` is the exact number of matching rows, computed after the page, snippets, attachments and requested explanations by a separate count plan in the same read transaction, and cached by the bound question and read position. Continuations reuse a count from that position. A cache miss or a changed read position runs the exact count again. A request with a deadline leaves time to send the prepared page. If the count is still pending, the page returns with `count: null` and a runnable query in `notes` that lifts the deadline. The worker checks cancellation when its current database read returns. A count is never estimated.

Counting rows applies the query's filters and transform selections without evaluating presentation fields or the final order. Fusion counts the union of selected keys; a list that only changes order adds no members. Ranking still runs when a selection such as `limit` depends on it, and missing semantic similarities are counted over the same selected rows. The reply reports missing similarities when its own rows have semantic ranking; a `similar:` used only in a union or fuse operand does not enable that diagnostic. Overlapping operands count each selected identity once, and a later operand cannot fill the first supplying operand’s unknown similarity.

## Index maintenance

- **One entry for each distinct searchable text**, keyed by the text's key in this index ([archive.md](archive.md#text)). A text enters a full-text index when the first block of that kind that holds it is written and leaves with the last, as part of the write path, not a command: the index's own schema keeps membership and the trigram index on tool calls with the shown rows, so composition writes the shown rows and nothing else. A build composes before that upkeep exists in its file, then sets every text's membership from the blocks in one statement and rebuilds the trigram index; its entries are written while it composes (below).
- **Full text lives in files beside the index, and how many is not observable.** The entries are held in several files, split by text key, so a build fills them at the same time on separate cores while it composes: each text a composed block holds is read, folded and written to its file by that file's own thread as soon as the block is written, and when composition ends the entries are brought to the membership the blocks give, adding what a block written before the build began holds and removing what a block that left again held. A search runs over all of them and ranks with the statistics of the whole corpus (rows, tokens, and the rows holding each phrase, summed over the files), so its results and their order are those of a single index over the same texts, however many files there are and however the texts are spread among them. Ties in rank are broken by natural keys.
- **Entries follow the index through one queue.** The transaction that adds or removes a text queues the change to its entries; the search thread reads the queue and its texts, and the writer applies the entries to the files. A hit is joined to the text's row and its blocks, so an entry that is ahead of or behind the index never shows a text that is missing or removed; a flush applies every waiting change before it returns, and a reply says when search text is pending ([reading.md](reading.md#paging-and-budgets)).
- **Ranking after upkeep equals a build's.** Statistics count the texts present and nothing else: a removed text's entry leaves with it, so the same record gives the same ranking whether its index was kept current or built from nothing.

## Semantic ranking

`similar:` orders a query's rows by meaning, using embeddings. It ranks rows rather than filtering them.

### Forms and meaning

- `similar:"text"` ranks by similarity to a text and needs a vector for that text. The vector of each quoted text is asked for before the request takes a reader, so waiting for the provider holds no reader slot; concurrent reads of one text share one provider call, and the returned vector is stored, so later reads, after a restart too, use the stored vector. `similar:REF` (a session, message or annotation reference) ranks by a stored item's vector and needs no provider call; the seed itself is excluded.
- `similar:` ranks the whole query and appears once, among the top-level AND terms; anywhere else it is `invalid_query`.
- **A ranking, never a filter.** Conditions select; `similar:` orders the selection. Vectors are scanned exactly over the selected rows, with no approximate index and no global top-K cut-off.
- **Grain:** a message ranks by its vector, a block by its message's, a session by its best message (subagents included), an annotation by its own.
- Rows without a vector are never dropped: they follow the ranked rows, unranked, counted in `unknown.similar`; `count` stays exact.
- A reply that needs only membership (`| agg count`, `| group by …`) computes no similarity and makes no call; after a ranked selection (`| limit`), the ranking is computed.
- `related:REF` is a different ranking, derived from session structure ([measures.md](measures.md)); `fuse` combines the two.

### Unavailable semantic ranking

With `embed.provider off`, a query ranked only by `similar:` is refused with `embeddings_off`: "semantic search is off; enable it with `polylogue set embed.provider local`". Returning every session would give the query text an apparent match it never established. When no row has a semantic score, a hybrid keeps its text relevance scores and their tie order. With embeddings off it says "rows are ranked by text relevance alone" after the reason.

An unreadable embeddings store, invalid embedding configuration, a model with no stored vectors, model calls off and a failed provider each have their own typed reason and sentence. Listings and readings use the same reason. When embeddings are configured but ranking cannot run, the reply is degraded: text relevance ranks a hybrid query alone; otherwise rows are ordered by time and marked unranked. A model with no stored vectors names `polylogue embed`; an unreadable store includes its error. A deadline or cancellation remains the read's timeout or cancellation, including the stage `vector wait` when waiting for another request's vectors.

### Model calls

- Model calls are off for now (default; open operator question); `polylogue set embed.calls on` turns them on. The provider and model (`embed.provider`, `embed.model`) name the vectors that rank either way.
- **With calls off,** `similar:REF` works on every stored vector, and `similar:"text"` works when that text's vector is already stored. Otherwise the reply is `incomplete`: rows in the default order, marked unranked, with the reason that model calls are off and `polylogue set embed.calls on` enables them. In a hybrid query the text side ranks alone and the note names the missing side. The vectors' state in `status` and the preview of `polylogue embed` work; `polylogue embed --yes` is refused with the same reason.
- **With calls on,** a query text is embedded once and its vector stored, keyed by model and normalized text hash, so it is never paid for twice. New messages and annotations are embedded in the background at idle priority within a monthly budget, default up to $5 a month (default; open operator question); spending beyond what the budget covers goes through [`embed`](#building-vectors).
- **Providers:** a local OpenAI-compatible endpoint (default model `qwen3-embedding:0.6b`) and Voyage. The Voyage key is only ever sent to Voyage's origin (`https://api.voyageai.com`): a Voyage provider configured with an endpoint elsewhere is refused at startup, and a request to another origin never carries the key. The local provider only sends text to a loopback address (`127.0.0.0/8`, `::1`, an IPv4-mapped loopback address or `localhost`) and never through a proxy, even when one is set in the environment, so message text never leaves the machine through it. Vectors from different models are compared only where the provider documents a shared space.
- **Provider failures:** a transient failure leaves the text without a vector until the next round; a permanent rejection is recorded once and listed in `status`; a hybrid query with a failed side runs the other side and is `incomplete`.
- Provider responses are validated as whole indexed batches before storage: exactly one nonempty, finite, nonzero vector of the same dimension per requested input, with unique indexes covering that request. Malformed responses are transient failures, never partial purchases or permanent input rejections. Normalization preserves finite tiny and large f32 components.
- A stored chunk does not make its whole text complete. The preview and the build use the missing chunk identities for the current embedding key; the existing key selection holds dimension recipes apart. A retry requests only missing chunks, preserves purchased ones, and refuses returned dimensions that differ from the selected key. Permanent provider rejections remain recorded and skipped.

### Building vectors

Vectors for text the daemon has not embedded are bought with `polylogue embed [--query 'QUERY'] [--yes]`: without `--yes` it reports what would be embedded and what it would cost, and with `--yes` it spends and builds ([installing.md](installing.md#embedding)). Spending stays an explicit act, so the preview is the default. Once model calls are on, the daemon embeds new text in the background within the budget above; `status` shows the vectors stored, missing and failed. Over MCP, `embed` is an operation of `polylogue_act`.

A logical session row selects prose in its root and known structural members,
including continuations. The query's limit counts returned rows, and
`sessions where physical:true` selects only the returned physical parts.
When combined sources name the same session, its first supplying operand
decides its logical or physical meaning, just as it supplies the row's values.
A kept result selects its captured physical identities, without adding current
descendants. Displayed aliases never replace those native identities. Preview
and purchase use the same selected prose.

### What is embedded

- One vector input per message: the message's person and model prose, its text blocks in order joined by a blank line, NFC-normalized. An input past the model's limit is chunked at that limit and its best chunk ranks. Annotation values are embedded too.
- A vector's address is a function of the request only (domain, model and text, length-prefixed), so a vector once bought stays addressable whatever the index does. Python Polylogue's vectors are imported by the address of the request that bought them (`polylogue import PATH --origin python-vectors`, [capture.md](capture.md#import)): each is stored for every text here whose request, the model and the NFC text, has that address, keeping its values and purchase time. A vector Python Polylogue stored only under its message identity cannot show which text it embedded, since that identity's hash covers position and parser output; it stays in its source file and is reported, as is a vector whose text is not here or is embedded in parts here. Nothing is bought again and no vector is dropped.
- Prepared Python Polylogue sources carry an NFC request preimage for every address. Import validates the preimage against the historical request address and model, then uses the index's normalized text identity to select all current raw spellings before decoding their bodies. Raw text hashes remain the stored vector keys. Original stores without request preimages use the existing full current-prose matching pass; a missing or invalid preimage in a marked prepared source is an error, not a sign that its text is absent.
- At the same text hash, full embedding key and chunk address, restore retains the vector with the newest known purchase time, independent of import order. A newer purchase updates the current values and time without changing the row ID, including when only the time changed. An older import reports that the newer address is present; an identical equal-time repeat changes nothing. Conflicting dimensions are refused; conflicting bytes at equal purchase times are also refused. Sources without a known purchase time remain explicitly unmatched. Earlier paid bytes stay in the stored original source stores.
- `embeddings.db` is written only by the daemon and backed up with `archive.db`: `embedded(id, address, key, chunk, dims, vector, embedded_at)`, `query_vector(key, text_hash, vector, embedded_at)`, `embed_failure(address, key, error, failed_at)`. A build that adds a table adds it to an existing `embeddings.db` at open and never rewrites the stored vectors ([archive.md#evolution](archive.md#evolution)). `index.db` holds `message_embed_input(message, address)`, written with the message.
- **Scan:** the daemon holds int8 copies of the configured model's vectors and scores the selected rows' texts exactly in int8, each when the query first asks for it, so the work follows the selection, not the loaded vectors; it then re-scores the top 200 of those texts in f32 from the stored rows, which fixes the order where quantization ties or misorders.
- **Warm vectors:** [Warm state](latency.md#warm-state) owns how requests share vector loads, extend snapshot revisions and bound the cache.

## Fusion

A plain word fuses its relevance order with a recency order the same way. A query with text terms and `similar:` fuses its rankings by reciprocal rank: each row scores Σ 1/(60 + rank) over the lists that rank it; ties go to recency, then identity. Fusion never changes membership or `count`. Each row carries `found_in`; `--why` adds each list's rank and contribution. A list that could not run is omitted and named, and the reply is `incomplete`. Combining queries with `| union` and `| fuse` uses the same fusion ([transforms.md](transforms.md)).

## Recall

Recall is the brief of a search: `polylogue 'WORDS' brief`, where the query holds only search words or quoted phrases (no field, unit or transform), a single plain word included. It replies to "what do I already know that bears on this", across every repository, ranked toward `--cwd DIR` (default the current directory). Over MCP it is the `brief` step with that selection: `polylogue_read {"arguments": {"selection": "retry after rate limit", "steps": [{"verb": "brief"}]}}`. What the brief contains, its budget and `--for` are [agent-context.md](agent-context.md#brief); this section owns the search under it. It is one plain query the daemon builds and shows in the reply, paged by its cursor, fusing five lists:

1. prose messages containing any of the words, by bm25 over an OR, so more words rank higher;
2. tool output, the same way;
3. annotations (claims, decisions, findings, lessons) whose value has any of the words;
4. `similar:` to the whole text, when a vector for it exists;
5. sessions in the `--cwd` directory, as a `fuse` list.

Words found in more than 20% of texts are dropped from the OR lists when the text has other words, and `notes` name them. Hits are grouped by session in fused order, with snippet, `found_in`, and the attached annotations and judgments, the operator's decisions first; text not written by the operator is labelled with its author. A budget cuts what is shown, never what is ranked. When exact search matches nothing and the semantic list did not run, the reply says so: "semantic search is off, so related work under other wording may exist". The query the brief shows can itself be run or saved like any query.

**Keeping, narrowing and the reader.** `polylogue 'retry after rate limit' brief save` keeps the search's rows as a kept result in the same line, since `save` acts on the selection the brief read ([saved-queries.md](saved-queries.md#save)). A condition in the selection narrows: `polylogue 'retry after rate limit repo:sinex since:30d' brief` is the brief of the rows that query selects, ranked by the same words over those rows only. `word~` makes a word approximate ([Fuzzy recall](#fuzzy-recall)). `--for operator` writes the brief for the operator to read ([agent-context.md](agent-context.md#briefs-for-the-operator)); `--for SESSION` leaves out what that session already has.

### Fuzzy recall

Fuzzy recall locates a half-remembered conversation by explicit, explained tolerances:

- `word~` also matches index words within edit distance 1 (4–6 letters) or 2 (7 or more), and words sharing its first four letters, from the daemon's vocabulary. Exact matches outrank approximate ones; `notes` list each substitution (`backpu → backup (312 texts)`). A `word~` term is a search word, so `polylogue 'backpu~ restic' brief` is still the brief of a search.
- `around:<date|month|"spring 2026">` adds closeness to a time window as one more fusion list; `since:` and `until:` stay filters. `around:` is a field condition, so `polylogue 'backpu~ around:"spring 2026"'` lists ranked sessions, and its brief is the brief of those rows.
- `similar:` on the text, when vectors exist.

Field names and values never match approximately; a mistyped field is an error with a suggestion.
