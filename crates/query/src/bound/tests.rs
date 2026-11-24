use super::*;
use crate::{
    Bindings, Options,
    defaults::declared,
    filter::{FieldTest, Test},
    plan::prepare,
};

fn bound(text: &str) -> Bound {
    let origins = vec!["claude-code".to_owned(), "codex".to_owned()];
    let bindings = Bindings {
        origins: origins.clone(),
        ..Bindings::default()
    };
    let clock = crate::time::Clock::new(
        jiff::Timestamp::from_millisecond(1_790_000_000_000).unwrap(),
        crate::time::DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::constant(4, 0, 0, 0),
        )
        .unwrap(),
    );
    prepare(text, &bindings, |_| Ok(()))
        .unwrap()
        .bound(&Options::fixture(clock, origins))
        .unwrap()
}

fn filter(text: &str) -> Filter {
    bound(text).selection.unwrap().filter.unwrap()
}

fn field_name(filter: &Filter) -> &'static str {
    match filter {
        Filter::Field(FieldTest { field, .. }) => field.declaration().name,
        other => panic!("not a field condition: {other:?}"),
    }
}

/// `exists` holds the related rows bound to their own unit, with the
/// defaults the condition's own top-level conditions leave in place.
#[test]
fn rows_a_condition_reads_are_bound_to_their_unit() {
    let Filter::Exists { rows, .. } = filter("sessions where exists messages(role:user)") else {
        panic!("not an exists condition");
    };
    assert_eq!(rows.unit, Unit::Messages);
    assert!(matches!(
        rows.filter.as_deref(),
        Some(Filter::Field(FieldTest { test: Test::OneOf { values, .. }, .. }))
            if values == &["user".to_owned()]
    ));
    assert_eq!(rows.defaults.plain, declared(Unit::Messages));
    let Filter::Count { rows, count, .. } =
        filter("sessions where count(messages where duplicate:true) > 2")
    else {
        panic!("not a count condition");
    };
    assert_eq!(count, 2);
    // A top-level condition on a default's field lifts that default.
    assert!(
        !rows
            .defaults
            .plain
            .contains(&crate::defaults::DefaultId::NotDuplicate)
    );
}

/// A query inside a condition is bound whole: its selection and stages.
#[test]
fn a_query_in_a_condition_is_bound_whole() {
    let Filter::In { field, query, .. } =
        filter("sessions where repo in (sessions where origin:codex | group by repo)")
    else {
        panic!("not a membership condition");
    };
    assert_eq!(field, "repo");
    let selection = query.selection.as_ref().unwrap();
    assert_eq!(selection.unit, Unit::Sessions);
    assert_eq!(field_name(selection.filter.as_ref().unwrap()), "origin");
}

/// A `| where` over a derived relation binds its condition, each leaf also
/// bound to the relation's unit; `| matches` over a relation selects the
/// patterns the source carries.
#[test]
fn relation_stages_are_bound() {
    let query = bound("sessions | sort by cost | where kind:main OR cost > 1");
    let StageBinding::Where(condition) = &query.stages[1] else {
        panic!("not a where binding: {:?}", query.stages[1]);
    };
    let RelationTest::Any(parts) = &condition.kind else {
        panic!("not a disjunction");
    };
    assert!(parts.iter().any(|part| matches!(
        &part.kind,
        RelationTest::Leaf {
            leaf: RelationLeaf::Compare { op: Op::Gt, .. },
            ..
        }
    )));
    let Filter::Any(unit) = condition.unit_filter().unwrap() else {
        panic!("not a disjunction over the unit");
    };
    assert!(
        unit.iter()
            .any(|part| matches!(part, Filter::Field(_)) && field_name(part) == "kind")
    );

    let query = bound("sessions where exists tool_calls(action:shell) | matches");
    let StageBinding::UnitRows(Ok(selection)) = &query.stages[0] else {
        panic!("not unit rows: {:?}", query.stages[0]);
    };
    assert!(matches!(
        selection.conjuncts().as_slice(),
        [Filter::Exists { rows, .. }] if rows.unit == Unit::ToolCalls
    ));
}

/// An order condition makes the source a relation boundary: it is read
/// over the relation, exposing its matching row, while the conditions
/// beside it still count alone.
#[test]
fn an_order_condition_is_read_over_the_relation() {
    let query =
        bound("messages where role:assistant AND after(messages where role:user) within 1m");
    let selection = query.selection.as_ref().unwrap();
    assert_eq!(selection.filter, None);
    assert_eq!(selection.clauses.len(), 1);
    assert_eq!(selection.clauses[0].text, "role:\"assistant\"");
    let RelationTest::All(parts) = &query.source_condition.unwrap().kind else {
        panic!("not a conjunction");
    };
    assert!(parts.iter().any(|part| matches!(
        &part.kind,
        RelationTest::Order {
            exposed: true,
            query: Ok(_),
            ..
        }
    )));
}

/// Each nested query binds once: a binding repeated at every level of
/// `id in (…)` would take time exponential in the depth.
#[test]
fn deeply_nested_queries_bind_once_each() {
    let depth = 64;
    let mut text = "sessions | group by id".to_owned();
    for _ in 0..depth {
        text = format!("sessions where id in ({text}) | group by id");
    }
    let mut query = bound(&text);
    let mut levels = 0;
    while let Some(Filter::In { query: inner, .. }) = query
        .selection
        .as_ref()
        .and_then(|selection| selection.filter.clone())
    {
        query = std::sync::Arc::unwrap_or_clone(inner);
        levels += 1;
    }
    assert_eq!(levels, depth);
}
