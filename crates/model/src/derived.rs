//! Definitions and evidence for computed values, shared by help and readers.
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Validity {
    Exact,
    Parameterized,
    Extracted,
    Judged,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Parameter {
    pub name: &'static str,
    pub value: i64,
    pub unit: &'static str,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Construct {
    pub name: &'static str,
    pub catalog: &'static str,
    pub represents: &'static str,
    pub reads: &'static [&'static str],
    pub unit: &'static str,
    pub denominator: Option<&'static str>,
    pub unknown: &'static str,
    pub parameters: &'static [Parameter],
    pub validity: Validity,
    pub tests: &'static [&'static str],
    pub figures: &'static [&'static str],
}

pub const CLUSTER_GAP_MS: i64 = 300_000;
pub const TIME_PARAMETERS: &[Parameter] = &[Parameter {
    name: "cluster_gap",
    value: CLUSTER_GAP_MS,
    unit: "milliseconds",
}];

pub const SESSION_SPAN: Construct = Construct {
    name: "session_span",
    catalog: "K15",
    represents: "Recorded session clocks, clustered intervals, tool intervals and the last recorded item",
    reads: &["message.at", "message.written_by", "message.prompt_kind", "message.stop_reason", "message.on_active_path", "message.duplicate_of", "block.kind", "tool_call.at", "tool_call.result_block_pk", "tool_call.outcome", "tool_call.command_norm", "tool_call.action", "tool_call.tool", "event.at"],
    unit: "milliseconds; prompts; tool calls",
    denominator: Some("one physical session, shown active-path messages"),
    unknown: "Absent clocks and durations are null; missing, equal and reversed tool times are counted separately",
    parameters: TIME_PARAMETERS,
    validity: Validity::Parameterized,
    tests: &["span_and_lanes_match_a_hand_computed_session", "tool_intervals_extend_clusters_and_share_overlap", "clustering_sensitivity_matches_the_morrow_shaped_sample"],
    figures: &["Morrow #2, 18 own messages: clustered 62 minutes at a 5-minute gap, 75 minutes at a 10-minute gap. Time lanes at both gaps: tools 1704000 ms, model 2348000 ms, human 448000 ms (2026-10-28)"],
};

pub const SESSION_TIME: Construct = Construct {
    name: "session_time",
    catalog: "K16",
    represents: "Wall span divided between tool commands, waiting for the model, waiting for the human and unattributed time",
    reads: SESSION_SPAN.reads,
    unit: "milliseconds",
    denominator: Some("session_span.wall_ms"),
    unknown: "Missing or reversed transitions are unattributed; a session without a clock has no time rows",
    parameters: TIME_PARAMETERS,
    validity: Validity::Parameterized,
    tests: &["span_and_lanes_match_a_hand_computed_session", "tool_intervals_extend_clusters_and_share_overlap", "time_lanes_sum_to_wall_for_overlapping_intervals"],
    figures: SESSION_SPAN.figures,
};

pub const SESSION_COUNTS: Construct = Construct {
    name: "session counts",
    catalog: "K23",
    represents: "Counts and prose words over a session's shown active-path rows",
    reads: &["message.written_by", "message.prompt_kind", "message.words", "message.on_active_path", "message.duplicate_of", "block.kind", "tool_call.action", "tool_call.outcome", "tool_call.run_by_script", "event.kind"],
    unit: "messages; prompts; words; tool calls; compactions",
    denominator: Some("one physical session"),
    unknown: "Pending derivations have null fields; a recorded empty session has zero counts",
    parameters: &[],
    validity: Validity::Exact,
    tests: &["stored_counts_match_the_shown_path_and_action_population", "stored_counts_and_time_match_morrow_after_the_real_loop"],
    figures: &[],
};

pub const OUTLOOK:Construct=Construct {
    name:"billing-cycle outlook",catalog:"K12",
    represents:"Projected cycle usage under dated, explicitly configured plan terms",
    reads:&["setting.set_at","setting.value","message.at","message.cost_nusd","message.provider_cost_nusd","message token lanes","usage_report"],
    unit:"nano-USD; tokens",denominator:Some("one UTC billing cycle, split at each plan change; responses and recorded reports apart"),
    unknown:"Each unit counts known and unknown items; absent usage stays null; untimed items stay outside the cycle",
    parameters:&[Parameter{name:"trailing_observed_days",value:7,unit:"observed UTC days"}],
    validity:Validity::Parameterized,
    tests:&["projections_preserve_history_zero_elapsed_and_recent_spike","grouped_billing_keeps_nano_usd_unknowns_and_future_rows_apart","a_plan_change_splits_the_cycle_and_prices_only_its_own_span"],
    figures:&["Ten synthetic observed days, nine at 0 then 700: projected 2700 with a 7-observed-day trailing mean, 2100 with a 10-observed-day trailing mean, for a 30-day cycle at day 10 (2026-10-28)"],
};

pub const FACETS:Construct=Construct {
    name:"facet counts",catalog:"K29",
    represents:"Selected and default-population session counts by value, with IDF and lift",
    reads:&["session recorded columns","message.model","message.written_by","block.kind","tool_call.action","note tags","session_span.ends_with"],
    unit:"physical sessions",denominator:Some("selected sessions and the same unit's defaults"),
    unknown:"Null is its own bucket; absent weights have a null denominator; pending session endings are counted",
    parameters:&[],validity:Validity::Exact,
    tests:&["facets_group_distinct_sessions_and_compute_idf_and_lift_at_read","facets_use_the_selection_units_default_population","time_comparison_and_facets_page_with_the_query_cursor"],figures:&[],
};

pub const COMPARE:Construct=Construct {
    name:"session compare",catalog:"K30",
    represents:"Recorded fields side by side, with varying fields and unknown counts over the whole selection",
    reads:&["session count columns","session_span","session_time","message.cost_nusd"],
    unit:"field values per physical session",denominator:Some("the full selected population, before paging"),
    unknown:"Missing values remain null and are counted by field; pending stored products are excluded from population summaries",
    parameters:TIME_PARAMETERS,validity:Validity::Exact,
    tests:&["comparisons_use_recorded_fields_and_keep_missing_values","comparisons_keep_whole_selection_differences_on_every_page","cost_comparison_keeps_integer_nano_usd_and_unknown_totals"],figures:&[],
};

/// The checked-in reference uses these exact declarations.
pub fn markdown(products:&[Construct])->String {
    let mut out=String::from("# Derived data\n\nGenerated from the product declarations. Session products read parsed rows; read-time products may read settings, notes and the query clock.\n");
    for p in products {
        out.push_str(&format!("\n## {} ({})\n\n{}\n\nKind: {:?}. Unit: {}. Denominator: {}.\n\nUnknown: {}.\n\nReads: {}.\n",p.name,p.catalog,p.represents,p.validity,p.unit,p.denominator.unwrap_or("none"),p.unknown,p.reads.join(", ")));
        for param in p.parameters {out.push_str(&format!("\nParameter: {} = {} {}.\n",param.name,param.value,param.unit));}
        out.push_str(&format!("\nTests: {}.\n",p.tests.join(", ")));
        for figure in p.figures {out.push_str(&format!("\nEvidence: {figure}.\n"));}
    }
    out
}

pub fn all()->Vec<Construct>{vec![SESSION_COUNTS,SESSION_SPAN,SESSION_TIME,OUTLOOK,FACETS,COMPARE]}
