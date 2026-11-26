//! `| measure facets` (docs/measures.md, facet counts with IDF and lift)
//! over Morrow, read the way the daemon reads it: the facet rows of the
//! measured query, then the weights from the selected and global sessions.

use crate::{program, session_list};
use polylogue_store::measure::{Paging, Selection, facets};
use polylogue_store::query::execute::{QueryPopulation, prepare_measure};
use polylogue_testkit::{Morrow, Set};
use serde_json::{Value, json};

/// The selection `origin:claude-code` holds the logical sessions #1, #2
/// and #4 (#3, a subagent, folds into #2). The unit's defaults alone hold
/// the six logical sessions #1, #2, #4, #5, #7 and #8 (#6 folds into #5).
///
/// IDF = ln(global sessions / global sessions with the value);
/// lift = (selected with the value / selected) / (global with it / global):
/// - origin claude-code: 3 of 3 selected, 3 of 6 global: IDF ln(6/3) =
///   ln 2 = 0.6931471805599453, lift (3/3)/(3/6) = 2.
/// - origin codex, chatgpt, claude-ai: 0 selected, 1 of 6 global: IDF
///   ln 6 = 1.791759469228055, lift (0/3)/(1/6) = 0.
/// - model claude-opus-5-5 (only #3's, counted for #2): 1 selected, 1
///   global: IDF ln 6, lift (1/3)/(1/6) = 2.
/// - model claude-fable-5-1 (#1, #2, #4): as origin claude-code.
/// - gpt-6.1-sol (#5 and #6), gpt-6 (#7) and #8's unrecorded model: 0
///   selected, 1 global each.
#[test]
fn facets_count_selected_and_global_sessions_and_weigh_them() {
    let morrow = Morrow::build(Set::Main);
    let pool = morrow.reads();
    let reader = pool.reader().unwrap();
    assert_eq!(
        session_list(&reader, "sessions where origin:claude-code"),
        ["#4", "#2", "#1"]
    );

    let mut measured = program(
        &reader,
        "sessions where origin:claude-code | measure facets",
        &[],
    )
    .unwrap();
    prepare_measure(&mut measured, Some(Paging::Facets), "", None).unwrap();
    let mut values = reader.query_page(&measured, None).unwrap().rows;
    let scope = Selection {
        sessions: Some(
            reader
                .query_keys(&measured, QueryPopulation::MeasureSelection)
                .unwrap(),
        ),
        population: Some(
            reader
                .query_keys(&measured, QueryPopulation::MeasurePopulation)
                .unwrap(),
        ),
        ..Default::default()
    };
    let summary = facets::summary(&reader, &scope).unwrap();
    assert_eq!(
        serde_json::to_value(&summary).unwrap(),
        json!({"outcome": "ok", "selected_sessions": 3, "global_sessions": 6, "pending": 0})
    );
    facets::weights(&mut values, &summary);

    const LN_2: f64 = 0.6931471805599453;
    const LN_6: f64 = 1.791759469228055;
    let row = |family: &str, value: Value, selected: u64, global: u64, idf: f64, lift: f64| {
        json!({
            "family": family, "value": value,
            "selected_count": selected, "global_count": global,
            "idf": idf, "lift": lift,
        })
    };
    let of = |family: &str| -> Vec<Value> {
        let mut rows: Vec<Value> = values
            .iter()
            .filter(|row| row["family"] == family)
            .cloned()
            .collect();
        rows.sort_by_key(|row| row["value"].to_string());
        rows
    };
    assert_eq!(
        of("origins"),
        [
            row("origins", json!("chatgpt"), 0, 1, LN_6, 0.0),
            row("origins", json!("claude-ai"), 0, 1, LN_6, 0.0),
            row("origins", json!("claude-code"), 3, 3, LN_2, 2.0),
            row("origins", json!("codex"), 0, 1, LN_6, 0.0),
        ]
    );
    assert_eq!(
        of("models"),
        [
            row("models", json!("claude-fable-5-1"), 3, 3, LN_2, 2.0),
            row("models", json!("claude-opus-5-5"), 1, 1, LN_6, 2.0),
            row("models", json!("gpt-6"), 0, 1, LN_6, 0.0),
            row("models", json!("gpt-6.1-sol"), 0, 1, LN_6, 0.0),
            row("models", Value::Null, 0, 1, LN_6, 0.0),
        ]
    );
}
