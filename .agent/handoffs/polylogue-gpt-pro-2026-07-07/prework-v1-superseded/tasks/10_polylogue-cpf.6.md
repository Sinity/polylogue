# 10. polylogue-cpf.6 — Inject clock seam for relative-date parsing and audit sort_key_ms epoch pins

Priority: **P1**  
Lane: **temporal-honesty**  
Readiness: **ready-now / code-local plus audit artifact**

Depends on packet(s): polylogue-cpf.5

## Why this is urgent / critical-path

Relative date filters and synthetic timestamp fallbacks affect which sessions appear in queries. They must be deterministic in tests and explicit when timeless rows are included.

## Static diagnosis / likely mechanism

Root causes:
- `parse_date` hardcodes `datetime.now(tz=timezone.utc)` as dateparser `RELATIVE_BASE` (`polylogue/core/dates.py:10-44`), so `since:yesterday` is not injectable.
- Query lowering calls `parse_date(value)` directly (`polylogue/storage/sqlite/archive_tiers/archive.py:6966`).
- Many read paths use `COALESCE(..., s.sort_key_ms, 0)` for ordering/window rows (`archive.py:4914+`, `5163+`, `5247+`, etc.), which can pin unknown-time rows to epoch instead of labeling them as unknown/synthetic.

## Implementation plan

Implementation shape:
1. Add `polylogue/core/clock.py` with a `Clock` protocol, `SystemClock`, and `FixedClock`.
2. Change `parse_date(date_str, *, now=None, clock=None)` and use the provided base for relative parsing.
3. Thread clock through query-spec construction and storage lowerers; CLI/daemon pass default system clock, tests pass fixed clock.
4. Add a small audit artifact listing each `sort_key_ms` fallback to `0`, classified as fixed, safe intentional, or needs follow-up.
5. Replace unsafe order/window fallbacks with explicit NULL handling (`IS NULL`, `NULLS LAST` emulation) or surface `time_confidence=synthetic` where timeless rows must remain visible.
6. For `since/until` filters, exclude truly timeless rows by default unless an explicit include-timeless flag/profile says otherwise.

## Test plan

Tests:
- `parse_date("7 days ago", now=fixed)` returns deterministic UTC.
- CLI/daemon structured query using relative date returns stable results under a fixed clock.
- timeless rows do not match `since` solely due to `0` fallback.
- ordering places timeless rows deterministically without pretending they occurred in 1970.
- grep/lint test prevents direct `datetime.now()` in query date parsing.

## Verification command / proof

`devtools test tests/unit/core/test_dates.py tests/unit/archive/test_query_dates.py -k 'relative or clock or sort_key or timeless'` plus audit artifact review.

## Pitfalls

Do not make tests monkeypatch global datetime. The point is a clean seam. Do not delete timeless rows from reads; classify/exclude/include deliberately.

## Files/functions to inspect or touch

- `polylogue/core/dates.py:10-44`
- `polylogue/storage/sqlite/archive_tiers/archive.py:6966`
- `polylogue/storage/sqlite/archive_tiers/archive.py:4914+`
- `polylogue/storage/sqlite/archive_tiers/archive.py:5163+`
- `polylogue/archive/query/*`
- `polylogue/cli/archive_query.py`
- `polylogue/daemon/http.py`
