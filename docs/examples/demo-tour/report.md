# Polylogue Demo Tour Report

Status: **passed**

This report was produced by `polylogue demo tour` against the deterministic
private-data-free demo archive.

## Timings

- First query result: 7.956s (budget 30s)
- Full tour: 13.442s (budget 420s)

## Archive

- Archive root: `archive`
- Sessions: 11
- Messages: 42
- User overlays: present

## Steps

| Step | Exit | Duration | Bytes | Output |
| --- | ---: | ---: | ---: | --- |
| archive facets | 0 | 4.155s | 1426 | `01-archive-facets.txt` |
| pytest evidence drilldown | 0 | 1.999s | 199 | `02-pytest-evidence-drilldown.txt` |
| session evidence by id | 0 | 1.883s | 360 | `03-session-evidence-by-id.txt` |
| query facets | 0 | 1.603s | 1357 | `04-query-facets.txt` |

## Problems

- none

## Artifacts

- Transcript: `transcript.txt`
- JSON report: `report.json`
- Recording tape: `recording.tape`
