use super::*;
use crate::{Bindings, plan::prepare};

fn selection(text: &str) -> Selection {
    selection_with(text, &[])
}

fn bindings() -> Bindings {
    Bindings {
        origins: vec!["claude-code".into(), "codex".into()],
        ..Bindings::default()
    }
}

fn selection_with(text: &str, fields: &[String]) -> Selection {
    prepare(text, &bindings(), |_| Ok(()))
        .unwrap()
        .selection(fields)
        .unwrap()
        .unwrap()
}

fn error(text: &str) -> QueryError {
    prepare(text, &bindings(), |_| Ok(()))
        .unwrap()
        .selection(&[])
        .unwrap_err()
}

/// The filter's top-level conditions, written back as text and sorted, so
/// differently written queries compare by what they select.
fn conditions(selection: &Selection) -> Vec<String> {
    let catalog = Bindings::default().catalog();
    let mut conditions: Vec<String> = selection
        .top_level()
        .into_iter()
        .map(|condition| {
            let query = Query {
                source: Source::Unit {
                    unit: selection.unit.name().into(),
                    implicit: false,
                    condition: Some(condition),
                },
                transforms: vec![],
                attach: vec![],
            };
            query_language::format::format_query(&query, &catalog).unwrap()
        })
        .collect();
    conditions.sort();
    conditions
}

/// `A B`, `A | where B` and `(A) AND (B)` select the same rows: one filter,
/// the same defaults, no stage left.
#[test]
fn where_stages_after_the_source_join_its_filter() {
    let joined = selection("sessions where repo:morrow origin:codex");
    // Every transform before `stages_from` joined the filter.
    for (text, stages_from) in [
        ("sessions where repo:morrow | where origin:codex", 1),
        ("sessions where (repo:morrow) AND (origin:codex)", 0),
    ] {
        let other = selection(text);
        assert_eq!(conditions(&other), conditions(&joined), "{text}");
        assert_eq!(other.defaults, joined.defaults, "{text}");
        assert_eq!(other.stages_from, stages_from, "{text}");
    }
    assert_eq!(
        selection("sessions where repo:morrow | where origin:codex | sort by started").stages_from,
        1
    );
}

/// A where stage after `last`, after an order condition, or before
/// `| matches` stays a stage, because the conjunction means something else.
#[test]
fn where_stages_that_change_meaning_stay_stages() {
    for text in [
        "sessions where last | where origin:codex",
        "sessions where repo:morrow | where last",
        "sessions where exists tool_calls(action:shell) | where origin:codex | matches",
    ] {
        assert_eq!(selection(text).stages_from, 0, "{text}");
    }
}

/// The source's own top-level conditions bind its defaults; a later where
/// stage only narrows.
#[test]
fn defaults_come_from_the_source_alone() {
    let all = defaults::declared(Unit::Sessions).to_vec();
    assert_eq!(selection("sessions").defaults, all);
    assert!(
        !selection("sessions where hidden:true")
            .defaults
            .contains(&DefaultId::SessionsVisible)
    );
    assert_eq!(
        selection("sessions where repo:morrow | where hidden:true").defaults,
        all
    );
}

/// `similar:` ranks rather than filters: it leaves the filter, and a row
/// unit without its own similarity ranks by its session's.
#[test]
fn similar_is_the_semantic_leg() {
    let ranked = selection("messages where similar:\"retry budget\" role:user");
    assert_eq!(
        ranked.semantic.as_ref().map(|s| (s.text.as_str(), s.unit)),
        Some(("retry budget", Unit::Messages))
    );
    assert_eq!(conditions(&ranked), ["messages where role:\"user\""]);
    let calls = selection("tool_calls where similar:deploy");
    assert_eq!(calls.semantic.map(|s| s.unit), Some(Unit::Sessions));
    assert!(calls.filter.is_empty());
    // A second similar: stays in the filter for lowering to refuse.
    let twice = selection("sessions where similar:deploy similar:retry");
    assert_eq!(twice.semantic.map(|s| s.text), Some("deploy".to_owned()));
    assert_eq!(twice.filter.len(), 1);
}

#[test]
fn similar_by_a_reference_is_not_built() {
    assert_eq!(
        error("sessions where similar:claude-code:abc").kind(),
        "not_built"
    );
}

/// The columns shown are a projection beside the selection.
#[test]
fn fields_are_a_projection() {
    let fields = vec!["id".to_owned(), "repo".to_owned()];
    let shown = selection_with("sessions where repo:morrow", &fields);
    assert_eq!(shown.projection, Projection::Fields(fields));
    let plain = selection("sessions where repo:morrow");
    assert_eq!(plain.projection, Projection::Default);
    assert_eq!(conditions(&shown), conditions(&plain));
    assert_eq!(shown.defaults, plain.defaults);
}
