# Sources

## Code Review Rules

- Parser field choice: flag a parser that takes a truncated or display field
  when a complete one exists, reads only one of the top-level and nested
  placements the provider emits, or emits the same content from two fields. A
  dropped field needs a declared reason in the hash partition or origin spec.
- Flag a detector declaration whose `detector_tightness` or `mode_rank` in
  `origin_specs.py` is looser than its shape, wherever its predicate lives;
  an earlier parser then claims its records.
- Flag a reverse lookup from Origin to Provider that picks one of several
  matches without independent evidence. Safe path: refuse, or use a declared
  hint such as the `family_hint` of `provider_from_origin` in
  `core/sources.py`.
- Pre-acquisition exclusion has one owner, `classify_pre_acquisition` in
  `live/batch_support.py`. Flag a second predicate that decides exclusion for
  intake or the cold-build baseline.
