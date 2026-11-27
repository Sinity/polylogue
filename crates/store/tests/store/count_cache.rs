//! A query's exact count is reused at the same read position and
//! computed again otherwise. Which counts were computed is what a
//! request's read limit records as its `count` stage, the timings the
//! daemon reports for a request.

use crate::program;
use polylogue_store::Lane;
use polylogue_store::cancellation::Cancellation;
use polylogue_store::read::limit::ReadLimit;
use polylogue_testkit::{Morrow, Set};

/// Whether asking for `text`'s count, as a request of its own, computed it.
fn computes(pool: &polylogue_store::ReadPool, text: &str) -> bool {
    let limit = ReadLimit::without_deadline(Cancellation::default());
    let reader = pool.checkout(Lane::Interactive, &limit).unwrap();
    let program = program(&reader, text, &[]).unwrap();
    assert!(reader.query_count(&program).unwrap().is_some(), "{text}");
    limit
        .stage_timings()
        .iter()
        .any(|(stage, _)| *stage == "count")
}

/// The reader pool keeps the 64 most recent distinct counts: the 65th
/// evicts the oldest, and the newest stays.
#[test]
fn the_sixty_fifth_distinct_count_evicts_the_oldest() {
    let morrow = Morrow::build(Set::Main);
    let pool = morrow.reads();
    let question = |n: u32| format!("sessions where messages >= {n}");
    assert!(computes(&pool, &question(0)));
    assert!(!computes(&pool, &question(0)), "asked again, it is reused");
    for n in 1..=64 {
        assert!(computes(&pool, &question(n)), "{}", question(n));
    }
    assert!(!computes(&pool, &question(64)), "the newest is kept");
    assert!(!computes(&pool, &question(1)), "the second oldest is kept");
    assert!(computes(&pool, &question(0)), "the oldest was evicted");
}
