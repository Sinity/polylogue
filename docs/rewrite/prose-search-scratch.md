# Query-private prose scratch writes

Each prose query builds its FTS tables in a private in-memory connection. After schema creation, one transaction batches that connection’s inserts. Searches on that connection see preceding additions, including additions after an earlier search. Dropping the query discards both the transaction and its scratch index. A failed or cancelled builder is discarded rather than reused as a complete corpus.

This transaction does not cover retained catalog, live visibility, generation or excision connections. Their current checks and query population remain unchanged. Per-Message, corpus, document, cancellation and deadline admission remain unchanged. Component measurements motivate batching; they do not establish that a prior whole-query timeout is fixed.
