# Handoff-Pack Uplift Pilot

This n=1 pilot compared two agents reconstructing the current Polylogue devloop continuation state.

- Raw-ref arm: given only session `019f12b5-fc19-7110-b069-4f49a78da82d` plus normal archive/repo tools.
- Pack arm: allowed the bounded handoff packet generated from that session.

## Result

The raw-ref arm scored 8/10. The handoff-pack arm scored 5/10.

The pack arm found useful prior-slice context, but it misidentified the current slice as the earlier bounded CLI search/select latency work. The raw-ref arm reconstructed that the devloop had moved to `polylogue-jxe.2`, the two-arm handoff-pack uplift protocol.

## Interpretation

This is a negative diagnostic pilot, not a general result. The bounded handoff packet was honest but stale: it summarized the subject session before the follow-on `jxe.2` protocol started. The raw-ref arm used broader archive and Beads evidence, which was better for current-state reconstruction but more expensive and hit a product hang in `polylogue continue --format json`.

## Product Feedback

- Handoff packets need freshness metadata and successor/context links when used for continuation.
- The experiment should compare frozen subject continuation tasks, not the live current devloop after the packet is already stale, unless freshness is the target construct.
- `polylogue-qt3` remains important: declarative read-package regeneration should be single-process and progress-visible.
- The raw-ref hang in `polylogue continue --format json` is actionable performance evidence.

## Files

- `protocol.json`: fixed task, arms, allowed/forbidden context, and construct-validity limits.
- `metrics/ground-truth.json`: scoring truth captured from current conductor/Beads state.
- `metrics/rubric.json`: 10-point scoring rubric.
- `metrics/score.json`: scored arm outputs.
- `arms/raw-ref-output.md`: raw-reference arm response.
- `arms/handoff-pack-output.md`: handoff-pack arm response.
