//! Measures read from per-session facts have no message times, so a scope
//! bounded in time is refused with its reason, never answered over the
//! whole sessions (docs/measures.md).

use crate::program;
use polylogue_model::Reason;
use polylogue_store::StoreError;
use polylogue_store::measure::{Selection, survival, time};
use polylogue_store::query::execute::QueryPopulation;
use polylogue_testkit::{Morrow, Set};

/// 2026-10-02 00:00 UTC, inside the week Morrow covers.
const OCTOBER_2: i64 = 1_790_899_200_000;

#[test]
fn per_session_measures_refuse_since_until_and_undated_scopes() {
    let morrow = Morrow::build(Set::Main);
    let pool = morrow.reads();
    let reader = pool.reader().unwrap();
    let every = program(&reader, "kind:*", &[]).unwrap();
    let sessions = reader.query_keys(&every, QueryPopulation::Answer).unwrap();
    assert_eq!(sessions.len(), 8);
    let scope = Selection {
        sessions: Some(sessions),
        ..Default::default()
    };

    // The control: the same sessions unbounded. Two of the eight fail a
    // tool call (#2's cargo test, #4's cargo clippy; FACTS.md), the other
    // six are censored at their last activity.
    let curve = survival::first_failure(&reader, &scope, false).unwrap();
    assert_eq!(
        (
            curve.sessions,
            curve.events,
            curve.censored,
            curve.unmeasured,
            curve.pending
        ),
        (8, 2, 6, 0, 0)
    );
    assert_eq!(time::times(&reader, &scope).unwrap().len(), 8);

    let bounded = [
        Selection {
            since: Some(OCTOBER_2),
            ..scope.clone()
        },
        Selection {
            until: Some(OCTOBER_2),
            ..scope.clone()
        },
        Selection {
            undated: true,
            ..scope.clone()
        },
    ];
    for selection in &bounded {
        for (measure, refusal) in [
            (
                "survival",
                survival::first_failure(&reader, selection, false).map(drop),
            ),
            ("time", time::times(&reader, selection).map(drop)),
        ] {
            match refusal {
                Err(StoreError::Refused(reason)) => assert_eq!(
                    reason,
                    Reason::MeasureTakesNoTimeBounds {},
                    "{measure} over {selection:?}"
                ),
                other => panic!("{measure} over {selection:?} answered {other:?}"),
            }
        }
    }
}
