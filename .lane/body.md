Summary

Add a small in-memory archive oracle for cross-surface differential tests.
It consumes the production query parser's typed AST, evaluates common session
predicates, tracks lineage ancestry, and emits session-grain counts, facets,
and canonical cost facts.

Problem

Cross-surface tests had semantic fact normalizers but no executable reference
model. A second grammar or surface-specific oracle would allow alternate
engines to drift without a semantic baseline.

Solution

Add `tests.infra.reference_model.ReferenceArchive` and focused tests covering
AST-backed Boolean selection, session-grain aggregation, text selection, and
lineage tracking. The model delegates cost semantics to the canonical pricing
fold and does not write archive state.

Verification

- `nix develop --command ruff check tests/infra/reference_model.py tests/unit/core/test_reference_model.py` — All checks passed.
- `nix develop --command ruff format --check tests/infra/reference_model.py tests/unit/core/test_reference_model.py` — files formatted.
- `nix develop --command devtools test tests/unit/core/test_reference_model.py` — 2 passed.

Residual risk

This first slice covers session predicates and common structural fields. Vector
predicates, full terminal-unit pipelines, and production CorpusProgram-to-all-
surface differential wiring remain follow-up work.
