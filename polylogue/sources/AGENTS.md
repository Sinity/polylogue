# Sources

## Code Review Rules

- Parser field choice: flag a parser that takes a truncated or display field
  when a complete one exists, reads only one of the top-level and nested
  placements the provider emits, or emits the same content from two fields. A
  dropped field needs a declared reason in the hash partition or origin spec.
- Flag a new detector in `dispatch.py` placed looser than its shape; an
  earlier parser then claims its records.
- Flag any reverse lookup from Origin to Provider that picks one of several
  matches. Safe path: refuse when more than one provider matches.
- Pre-acquisition exclusion has one owner, `classify_pre_acquisition` in
  `live/batch_support.py`. Flag a second predicate that decides exclusion for
  intake or the cold-build baseline.
