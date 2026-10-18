# Provider-state installation

Parsing a window produces a delta of encoded state cells. The owner writes it
in several transactions: stage a prefix of cells per transaction, seal the
complete delta, apply prefixes, then publish. Each transaction takes at most 64
cells and 2 MiB of encoded staging or application work, counted with the store
codec's per-member framing, key and value cost. A cell that does not fit waits
for the next transaction; if the first cell does not fit, the turn fails before
writing anything.

Every transaction checks the parser-state and work tokens and rolls back on
failure. Only publication advances the visible source position. On shutdown,
unsealed work is discarded and the input is parsed again; sealed work is
finished without parsing again. The actor checks cancellation and serves its
mailbox between transactions.
