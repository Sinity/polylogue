# Mutmut Campaign: `site-builder`

- Recorded on `2026-03-11T09:58:59.595335+00:00`
- Commit: `2bdb267e93b79f1f0dc863f86b5ed859e4e0dbdd`
- Worktree dirty: `no`
- Description: Static-site builder and CLI archive contracts
- Workspace: `/tmp/nix-shell.rzHew1/mutmut-site-builder-tq8ao4hc/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/site/builder.py`, `polylogue/cli/commands/site.py`
- Selected tests: `tests/integration/test_site.py`, `tests/integration/test_site_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 245 |
| Survived | 228 |
| Timeout | 1 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `62.41s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_generate_pagefind_config` | 46 |
| `_generate_conversation_page` | 34 |
| `_generate_provider_indexes` | 23 |
| `__init__` | 21 |
| `_build_async` | 16 |
| `_scan_archive` | 14 |
| `_generate_root_index` | 12 |
| `_generate_dashboard` | 12 |
| `_iter_conversation_indexes` | 9 |
| `_search_document` | 8 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `_search_markup` | 1 |

## Dominant Not-Checked Clusters

| Function | Count |
| --- | ---: |
| _none_ | 0 |

## Survivor Keys

- `polylogue.site.builder.x__format_summary_date__mutmut_3`
- `polylogue.site.builder.x__format_summary_date__mutmut_4`
- `polylogue.site.builder.x__format_summary_date__mutmut_5`
- `polylogue.site.builder.x__format_summary_date__mutmut_6`
- `polylogue.site.builder.x__format_summary_date__mutmut_7`
- `polylogue.site.builder.x__format_summary_date__mutmut_8`
- `polylogue.site.builder.x__format_summary_date__mutmut_9`
- `polylogue.site.builder.x__format_summary_date__mutmut_10`
- `polylogue.site.builder.x__format_summary_date__mutmut_11`
- `polylogue.site.builder.x__format_summary_date__mutmut_12`
- `polylogue.site.builder.x__format_summary_date__mutmut_13`
- `polylogue.site.builder.x__format_summary_date__mutmut_14`
- `polylogue.site.builder.x__format_summary_date__mutmut_15`
- `polylogue.site.builder.x__format_summary_date__mutmut_16`
- `polylogue.site.builder.x__format_summary_date__mutmut_17`
- `polylogue.site.builder.x__format_summary_date__mutmut_18`
- `polylogue.site.builder.xǁSiteBuilderǁ__init____mutmut_6`
- `polylogue.site.builder.xǁSiteBuilderǁ__init____mutmut_8`
- `polylogue.site.builder.xǁSiteBuilderǁ__init____mutmut_11`
- `polylogue.site.builder.xǁSiteBuilderǁ__init____mutmut_12`
- `polylogue.site.builder.xǁSiteBuilderǁ__init____mutmut_13`
- `polylogue.site.builder.xǁSiteBuilderǁ__init____mutmut_14`
- `polylogue.site.builder.xǁSiteBuilderǁ__init____mutmut_17`
- `polylogue.site.builder.xǁSiteBuilderǁ__init____mutmut_24`
- `polylogue.site.builder.xǁSiteBuilderǁ__init____mutmut_25`
- ... 203 more

## Timeout Keys

- `polylogue.site.builder.xǁSiteBuilderǁ_search_markup__mutmut_1`
