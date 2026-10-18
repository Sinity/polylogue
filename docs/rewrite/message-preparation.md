# Message preparation batching

Claude preparation skips absent selected-fact slots in a response cell it has
already read, examining at most eight indices per scan and checking
cancellation. Supplied nulls, empty strings and empty objects count as
supplied. Codex consumes its `AdvanceBlock` step (no IO) together with the next
block step, which still does at most one provider lookup. Older stored cursor
shapes are still read.

Consecutive small fields and source ranges that the worker has already checked
are sent as typed runs. Each entry keeps its own cursor and sequence check, and
the writer runs the same per-item checks as for a single item (current source,
visibility, phase, active producer, codec, field and range). A run commits in
one transaction; a failing member rolls back the whole run. Large values,
in-flight producers, structural operations and publication are not batched.

The worker also merges consecutive cursor-only `Advance` results into the next
checked field or citation. Before a streaming producer or structural operation
it stores one cursor. Active value and origin producers and their `Verified`
checkpoints still go one action at a time. Cancellation is checked between
advances; running out of allowance ends a seek with its progress stored.

A run starts with the first ordinary action under that adapter's normal limit.
If the run as a whole does not fit, that action is returned alone. Otherwise
the lookahead shares what is left of the original allowance: parser cells,
encoded namespace, key and value bytes, selector visits, requested record
bytes, value and descriptor hash work, entries, slot keys, cursors and guard
metadata. Repeated selector visits are charged each time. Backing-object
verification, frame decompression, allocations, RSS and elapsed time are not
limited.

On the synthetic eight-extra case this takes 18 worker actions instead of 65
and 29 committed cursor steps instead of 61, with all eleven fields and eleven
citations kept.

Directory construction inserts each final node and its edges inside the
caller's transaction. A node is visible only after every edge, child lease and
root transfer has succeeded and the transaction commits; a failure rolls the
whole construction back. The three construction statements are compiled once
per connection and bound with current values on each execution.
