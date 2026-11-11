//! A default first page reads its own rows, not the archive: SQLite's work
//! for it stays the same as sessions and messages are added.

use super::*;
use crate::fixture::ManySessions;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// SQLite virtual machine steps for the first page of `text` over an
/// archive of `sessions` sessions of three messages each.
fn first_page_steps(text: &str, sessions: usize) -> u64 {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    f.many_sessions(ManySessions {
        prefix: "/home/op/work",
        sessions,
        messages: 3,
        answered: 0,
        last: 1_800_000_000_000,
    })
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut opts = options();
    opts.page_size = None;
    let program = crate::query::compile(text, &opts).unwrap();
    let steps = Arc::new(AtomicU64::new(0));
    let counted = Arc::clone(&steps);
    let conn = reader.conn().unwrap();
    conn.progress_handler(
        1,
        Some(move || {
            counted.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    let mut rows = 0;
    reader
        .visit_page(
            &program,
            None,
            false,
            i64::try_from(program.page_size + 1).unwrap(),
            |_| {
                rows += 1;
                Ok::<_, SelectionError>(ControlFlow::Continue(()))
            },
        )
        .unwrap();
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    assert_eq!(rows, program.page_size + 1, "{text}");
    steps.load(Ordering::Relaxed)
}

#[test]
fn page_steps_flat_as_observations_grow() {
    for text in ["sessions", "messages"] {
        let small = first_page_steps(text, 100);
        let large = first_page_steps(text, 2_000);
        // Twenty times the rows may deepen an index by a level, never
        // add work per row.
        assert!(
            large <= small + small / 10,
            "{text}: {small} steps over 100 sessions, {large} over 2000"
        );
    }
}
