# Mutmut Campaign: `site-builder`

- Recorded on `2026-03-12T04:14:01.278470+00:00`
- Commit: `58264c2c47beaaa5522139c54700c20910833267`
- Worktree dirty: `no`
- Description: Static-site builder and CLI archive contracts
- Workspace: `/tmp/nix-shell.MEqXuM/mutmut-site-builder-ot5wtw_d/repo`
- Command: `mutmut run`

## Scope

- Mutated paths: `polylogue/site/builder.py`, `polylogue/cli/commands/site.py`
- Selected tests: `tests/integration/test_site.py`, `tests/integration/test_site_laws.py`

## Counts

| Status | Count |
| --- | ---: |
| Killed | 240 |
| Survived | 224 |
| Timeout | 10 |
| Not checked | 0 |
| Suspicious | 0 |
| Skipped | 0 |

- Runtime: `108.46s`
- Exit code: `0`

## Dominant Survivors

| Function | Count |
| --- | ---: |
| `_generate_pagefind_config` | 46 |
| `_generate_conversation_page` | 34 |
| `__init__` | 21 |
| `_generate_provider_indexes` | 21 |
| `_build_async` | 16 |
| `_scan_archive` | 14 |
| `_generate_dashboard` | 12 |
| `_generate_root_index` | 10 |
| `_iter_conversation_indexes` | 9 |
| `_search_document` | 8 |

## Dominant Timeouts

| Function | Count |
| --- | ---: |
| `_generate_root_index` | 4 |
| `_scan_archive` | 3 |
| `_generate_provider_indexes` | 2 |
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
- ... 199 more

## Timeout Keys

- `polylogue.site.builder.xǁSiteBuilderǁ_scan_archive__mutmut_12`
- `polylogue.site.builder.xǁSiteBuilderǁ_scan_archive__mutmut_14`
- `polylogue.site.builder.xǁSiteBuilderǁ_scan_archive__mutmut_15`
- `polylogue.site.builder.xǁSiteBuilderǁ_generate_root_index__mutmut_9`
- `polylogue.site.builder.xǁSiteBuilderǁ_generate_root_index__mutmut_20`
- `polylogue.site.builder.xǁSiteBuilderǁ_generate_root_index__mutmut_24`
- `polylogue.site.builder.xǁSiteBuilderǁ_generate_root_index__mutmut_25`
- `polylogue.site.builder.xǁSiteBuilderǁ_generate_provider_indexes__mutmut_28`
- `polylogue.site.builder.xǁSiteBuilderǁ_generate_provider_indexes__mutmut_29`
- `polylogue.site.builder.xǁSiteBuilderǁ_search_markup__mutmut_1`
