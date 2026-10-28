# Derived data

Generated from the product declarations. Session products read parsed rows; read-time products may read settings, notes and the query clock.

## session counts (K23)

Counts and prose words over a session's shown active-path rows

Kind: Exact. Unit: messages; prompts; words; tool calls; compactions. Denominator: one physical session.

Unknown: Behind count fields compute current rows and aggregates report pending; a recorded empty session has zero counts.

Reads: message.written_by, message.prompt_kind, message.words, message.on_active_path, message.duplicate_of, block.kind, tool_call.action, tool_call.outcome, tool_call.run_by_script, event.kind.

Tests: stored_counts_match_the_shown_path_and_action_population, stored_counts_and_time_match_morrow_after_the_real_loop.

## session_span (K15)

Recorded session clocks, clustered intervals, tool intervals and the last recorded item

Kind: Parameterized. Unit: milliseconds; prompts; tool calls. Denominator: one physical session, shown active-path messages.

Unknown: Absent clocks and durations are null; missing, equal and reversed tool times are counted separately.

Reads: message.at, message.written_by, message.prompt_kind, message.stop_reason, message.on_active_path, message.duplicate_of, block.kind, tool_call.at, tool_call.result_block_pk, tool_call.outcome, tool_call.command_norm, tool_call.action, tool_call.tool, event.at.

Parameter: cluster_gap = 300000 milliseconds.

Tests: span_and_lanes_match_a_hand_computed_session, tool_intervals_extend_clusters_and_share_overlap, clustering_sensitivity_matches_the_morrow_shaped_sample.

Evidence: Morrow #2, 18 own messages: clustered 62 minutes at a 5-minute gap, 75 minutes at a 10-minute gap. Time lanes at both gaps: tools 1704000 ms, model 2348000 ms, human 448000 ms (2026-10-28).

## session_time (K16)

Wall span divided between tool commands, waiting for the model, waiting for the human and unattributed time

Kind: Parameterized. Unit: milliseconds. Denominator: session_span.wall_ms.

Unknown: Missing or reversed transitions are unattributed; a session without a clock has no time rows.

Reads: message.at, message.written_by, message.prompt_kind, message.stop_reason, message.on_active_path, message.duplicate_of, block.kind, tool_call.at, tool_call.result_block_pk, tool_call.outcome, tool_call.command_norm, tool_call.action, tool_call.tool, event.at.

Parameter: cluster_gap = 300000 milliseconds.

Tests: span_and_lanes_match_a_hand_computed_session, tool_intervals_extend_clusters_and_share_overlap, time_lanes_sum_to_wall_for_overlapping_intervals.

Evidence: Morrow #2, 18 own messages: clustered 62 minutes at a 5-minute gap, 75 minutes at a 10-minute gap. Time lanes at both gaps: tools 1704000 ms, model 2348000 ms, human 448000 ms (2026-10-28).

## billing-cycle outlook (K12)

Projected cycle usage under dated, explicitly configured plan terms

Kind: Parameterized. Unit: nano-USD; tokens. Denominator: one UTC billing cycle, split at each plan change; responses and recorded reports apart.

Unknown: Each unit counts known and unknown items; absent usage stays null; untimed items stay outside the cycle.

Reads: setting.set_at, setting.value, message.at, message.cost_nusd, message.provider_cost_nusd, message token lanes, usage_report.

Parameter: trailing_observed_days = 7 observed UTC days.

Tests: projections_preserve_history_zero_elapsed_and_recent_spike, grouped_billing_keeps_nano_usd_unknowns_and_future_rows_apart, a_plan_change_splits_the_cycle_and_prices_only_its_own_span.

Evidence: Ten synthetic observed days, nine at 0 then 700: projected 2700 with a 7-observed-day trailing mean, 2100 with a 10-observed-day trailing mean, for a 30-day cycle at day 10 (2026-10-28).

## facet counts (K29)

Selected and default-population session counts by value, with IDF and lift

Kind: Exact. Unit: physical sessions. Denominator: selected sessions and the same unit's defaults.

Unknown: Null is its own bucket; absent weights have a null denominator; pending session endings are counted.

Reads: session recorded columns, message.model, message.written_by, block.kind, tool_call.action, note tags, session_span.ends_with.

Tests: facets_group_distinct_sessions_and_compute_idf_and_lift_at_read, facets_use_the_selection_units_default_population, time_comparison_and_facets_page_with_the_query_cursor.

## session compare (K30)

Recorded fields side by side, with varying fields and unknown counts over the whole selection

Kind: Exact. Unit: field values per physical session. Denominator: the full selected population, before paging.

Unknown: Missing values remain null and are counted by field; pending stored products are excluded from population summaries.

Reads: session count columns, session_span, session_time, message.cost_nusd.

Parameter: cluster_gap = 300000 milliseconds.

Tests: comparisons_use_recorded_fields_and_keep_missing_values, comparisons_keep_whole_selection_differences_on_every_page, cost_comparison_keeps_integer_nano_usd_and_unknown_totals.
