Title: "[beads 13] Order-independent lineage writes"

Job ID: `beads-13`
Result ZIP: `beads-13-lineage-order-independence-r01.zip`

Implement `polylogue-866e`: equivalent parent/child/replacement histories must
converge to the same physical tails, links, and composed transcript regardless
of ingestion order. Current stateful property failures cover stale sibling
replacement and a missing parent branch point that can crash read composition.

Start from the saved property failures and reduce them to named deterministic
transition fixtures with physical-row/link-row/composed-read oracles. Fix
production write/composition behavior rather than weakening lineage semantics,
discarding valid child content, or adding a test-only repair path. Missing
branch points must remain readable through a typed relation/completeness state.
