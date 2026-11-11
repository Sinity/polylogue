use super::*;
use crate::{Bindings, plan::prepare};

fn plan(text: &str) -> crate::plan::Plan {
    let bindings = Bindings {
        origins: vec!["claude-code".into(), "codex".into()],
        ..Bindings::default()
    };
    prepare(text, &bindings, |_| Ok(())).unwrap()
}

/// Each stage of `text` after its selection: its transform indexes and kind.
fn shapes(text: &str) -> Vec<(Range<usize>, StageKind)> {
    let plan = plan(text);
    stages(&plan.expanded_query().query().transforms, 0)
        .unwrap()
        .into_iter()
        .map(|stage| (stage.transforms, stage.kind))
        .collect()
}

#[test]
fn a_grouping_and_the_row_selection_or_aggregate_after_it_are_one_stage() {
    use StageKind::*;
    for (text, expected) in [
        (
            "sessions | group by origin | first 1",
            vec![(0..2, PerGroup)],
        ),
        ("sessions | group by origin | count", vec![(0..2, PerGroup)]),
        (
            "sessions | group by origin | agg count() | limit 1",
            vec![(0..2, PerGroup), (2..3, Rows)],
        ),
        // A grouping read by anything else is a stage of its own, and so
        // is a row selection after it.
        (
            "sessions | group by origin | sort by count | first 1",
            vec![(0..1, Rows), (1..2, Rows), (2..3, Rows)],
        ),
        (
            "sessions | group by origin | group by count | first 1",
            vec![(0..1, Rows), (1..3, PerGroup)],
        ),
    ] {
        assert_eq!(shapes(text), expected, "{text}");
    }
}

#[test]
fn consecutive_unions_and_fuses_are_one_fusion() {
    use StageKind::*;
    for (text, expected) in [
        (
            "sessions where origin:codex | union (sessions where origin:claude-code) | fuse (sessions where \"tests\") | limit 2",
            vec![(0..2, Fusion), (2..3, Rows)],
        ),
        (
            "sessions | union (sessions where origin:codex) | limit 2 | union (sessions where origin:claude-code)",
            vec![(0..1, Fusion), (1..2, Rows), (2..3, Fusion)],
        ),
        // Intersect and except restrict membership one operand at a time.
        (
            "sessions | intersect (sessions where origin:codex) | except (sessions where origin:claude-code)",
            vec![(0..1, Rows), (1..2, Rows)],
        ),
    ] {
        assert_eq!(shapes(text), expected, "{text}");
    }
}

#[test]
fn the_selection_holds_the_stages_after_its_folded_filters() {
    let plan = plan("sessions where repo:morrow | where origin:codex | group by origin | first 1");
    let clock = crate::time::Clock::new(
        jiff::Timestamp::from_millisecond(1_790_000_000_000).unwrap(),
        crate::time::DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::constant(4, 0, 0, 0),
        )
        .unwrap(),
    );
    let selection = plan.selection(&[], &clock).unwrap().unwrap();
    assert_eq!(
        selection.stages,
        vec![Stage {
            transforms: 1..3,
            kind: StageKind::PerGroup
        }]
    );
}

#[test]
fn a_measure_ends_the_query() {
    let plan = plan("sessions | measure cost | limit 1");
    let transforms = &plan.expanded_query().query().transforms;
    let error = stages(transforms, 0).unwrap_err();
    let QueryError::Invalid { message, position } = error else {
        panic!("{error:?}");
    };
    assert_eq!(message, "`| measure WHAT` is the last stage");
    // The refusal points at the stage after the measure.
    assert_eq!(position, transforms[1].span.start);
    assert_eq!(shapes("sessions | limit 3 | measure cost").len(), 2);
}
