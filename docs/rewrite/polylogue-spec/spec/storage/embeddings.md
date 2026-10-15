# embeddings.db

Purchased embedding vectors, keyed by text hash. Only the daemon writes it.
Deleting it means buying the vectors again, so it is backed up with
`archive.db` (`README.md`).

```sql
-- One vector per text, embedding key and chunk.
CREATE TABLE embedded (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- explicit, so VACUUM never renumbers it; never reused,
                                     --   so the highest id is the embeddings part of a read's view
  text_hash   BLOB NOT NULL,         -- the text that was embedded
  model       TEXT NOT NULL,         -- the full embedding key: provider/model/dimensions/input type/normalization,
                                     --   e.g. 'voyage/voyage-4/1024/document/unit'
  chunk       INTEGER NOT NULL DEFAULT 0,  -- texts longer than the model's input are embedded in chunks
  dims        INTEGER NOT NULL,
  vector      BLOB NOT NULL,         -- dims little-endian f32 values
  embedded_at INTEGER NOT NULL,
  UNIQUE (text_hash, model, chunk)
) STRICT;
```

```sql
-- A text the provider permanently rejected (HTTP 400 and similar): recorded once
-- with the provider's error, listed in `status`, retried only when the
-- embedding key or the text changes.
CREATE TABLE embed_failure (
  text_hash   BLOB NOT NULL,
  model       TEXT NOT NULL,         -- the full embedding key, as in `embedded`
  error       TEXT NOT NULL,         -- the provider's error, as given
  failed_at   INTEGER NOT NULL,
  PRIMARY KEY (text_hash, model)
) STRICT, WITHOUT ROWID;
```

- A transient failure (timeout, rate limit, 5xx) is not recorded; the text
  stays without a vector and the next round tries again.
- **No vector for excised text.** A vector is inserted only in a transaction
  that first checks that its text hash is still wanted (a `text` row used by
  an embedded block, or the value of a current note) and is not named by a
  `text:` excision; a provider response for a text that fails the check is
  discarded, so a request in flight during an excision cannot write the
  vector back.
- **Search.** `near:` scans int8 copies of the configured model's vectors
  held in memory and re-scores its top 200 from the f32 rows here
  (`behavior.md`, warm state); no SQLite extension and no approximate index.
  Measured on the operator's 167,943 purchased voyage-4 vectors (1,024
  dimensions, unit length): per-vector int8 copies with re-scoring found
  the exact top 10 for all 200 sample queries even when re-scoring only the
  top 50 (recall@10 1.0), and the int8 scan reads 172 MB per query against
  688 MB for f32; an exact f32 scan took 12.7 ms per query in NumPy
  (`evidence/vector-search-probe.md`). An approximate index (HNSW, IVF) is
  not needed until the vector count is far larger; if it ever is, its recall
  is measured against this exact scan first.
- **Changing model.** A model with a different dimension adds rows for that
  model; the previous model's vectors stay and are used again if it is
  configured again. One model's vectors serve another only when the provider
  documents a shared embedding space (the Voyage 4 family does); otherwise a
  model change embeds again. The configured model is used everywhere a vector
  is computed or compared.
- **Keyed by text hash**, so identical text is embedded once, a rebuild of
  `index.db` never invalidates a vector, and two machines' embeddings combine
  as a union. Vectors from the Python archive are imported once by matching
  text hashes (`plan/migration.md`).

Evidence: `polylogue-n846`, `egm8` (permanent failures), `polylogue-k617z`
(shared embedding space), `polylogue-mn0si` (one configured model);
`polylogue-kwsb.3` and `evidence/conceptual-lifecycle.md` #16 (a vector
written back after excision).
