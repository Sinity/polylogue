# Per-query prose search tables

Each prose query builds its FTS tables in a private in-memory connection. After
the schema is created, that connection's inserts run in one transaction;
searches on the connection see earlier inserts, including inserts made after an
earlier search. Dropping the query discards the transaction and the index. A
failed or cancelled build is discarded, never reused as a complete corpus.

The transaction covers only this scratch connection. Catalog, visibility,
generation and excision checks run on their own connections, unchanged.
