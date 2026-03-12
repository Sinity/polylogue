# Benchmark Campaign: search-filters

- Description: FTS and ConversationFilter benchmark domain
- Commit: `d833643dccdc045456d767168b07913f50b8b488`
- Worktree dirty: no
- Created: `2026-03-12T04:57:11.646793+00:00`
- Runtime: `28.88s`
- Command: `/realm/project/polylogue/.venv/bin/python -m pytest -q --override-ini=addopts=-ra -n 0 -p no:randomly --benchmark-enable --benchmark-json=/tmp/nix-shell.QMNzZH/benchmark-search-filters-p4qqyrev/pytest-benchmark.json tests/benchmarks/test_search_filters.py`
- Tests: `tests/benchmarks/test_search_filters.py`
- Benchmarks: `11`
- Warn threshold: `10.0%`
- Fail threshold: `20.0%`

## Slowest Benchmarks

| Benchmark | Mean (s) | Median (s) | Ops/s | Rounds |
| --- | ---: | ---: | ---: | ---: |
| `tests/benchmarks/test_search_filters.py::test_bench_fts_search_common_term` | 0.016776 | 0.016395 | 59.61 | 18 |
| `tests/benchmarks/test_search_filters.py::test_bench_filter_has_tool_use` | 0.012292 | 0.010172 | 81.36 | 70 |
| `tests/benchmarks/test_search_filters.py::test_bench_filter_combined` | 0.008280 | 0.007525 | 120.78 | 124 |
| `tests/benchmarks/test_search_filters.py::test_bench_filter_semantic_file_ops` | 0.007417 | 0.006668 | 134.83 | 116 |
| `tests/benchmarks/test_search_filters.py::test_bench_fts_search_multi_word` | 0.005649 | 0.004340 | 177.01 | 219 |
| `tests/benchmarks/test_search_filters.py::test_bench_filter_limit_scaling[200]` | 0.005060 | 0.004915 | 197.62 | 145 |
| `tests/benchmarks/test_search_filters.py::test_bench_filter_provider` | 0.004146 | 0.003246 | 241.17 | 242 |
| `tests/benchmarks/test_search_filters.py::test_bench_filter_limit_scaling[50]` | 0.003178 | 0.003168 | 314.64 | 281 |
| `tests/benchmarks/test_search_filters.py::test_bench_filter_limit_scaling[10]` | 0.002366 | 0.002498 | 422.59 | 317 |
| `tests/benchmarks/test_search_filters.py::test_bench_fts_search_rare_term` | 0.001255 | 0.001166 | 797.09 | 397 |

## Notes

- Canonical search/filter latency domain.
- Keep on session-seeded DB fixtures for comparability.
