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
    let (rows, steps, page_size) = page_steps(text, sessions, false);
    assert_eq!(rows, page_size + 1, "{text}");
    steps
}

/// The rows and SQLite virtual machine steps of the first page of `text`
/// over `sessions` sessions of three messages each, every session's answer
/// with a text when `embeddings` (so `similar:` has texts to score), and
/// the page size.
fn page_steps(text: &str, sessions: usize, embeddings: bool) -> (u64, u64, u64) {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    f.many_sessions(ManySessions {
        prefix: "/home/op/work",
        sessions,
        messages: 3,
        answered: if embeddings { sessions } else { 0 },
        last: 1_800_000_000_000,
    })
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut opts = options();
    opts.page_size = None;
    opts.embeddings = embeddings;
    let program = crate::query::compile(text, &opts).unwrap();
    assert_eq!(program.near.is_some(), embeddings, "{text}");
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
    (rows, steps.load(Ordering::Relaxed), program.page_size)
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

#[test]
fn similar_steps_scale_with_selected_rows() {
    // Session 3 alone works in d3/w3 below 1261 sessions.
    for (text, selected) in [
        (
            "sessions where cwd:/home/op/work/d3/w3 similar:\"finished work\"",
            1,
        ),
        (
            "messages where session.cwd:/home/op/work/d3/w3 similar:\"finished work\"",
            3,
        ),
    ] {
        let (rows, small, _) = page_steps(text, 100, true);
        assert_eq!(rows, selected, "{text}");
        let (rows, large, _) = page_steps(text, 1_200, true);
        assert_eq!(rows, selected, "{text}");
        // Twelve times the texts may deepen an index by a level; the
        // similarities scored are the selected rows' alone.
        assert!(
            large <= small + small / 5,
            "{text}: {small} steps over 100 sessions, {large} over 1200"
        );
    }
}

/// Queries paged key first (native rows, ranked or not) and whole (a
/// relation after a sort stage).
const PAGED: &[&str] = &[
    "sessions",
    "sessions where cwd:/home/op/work",
    "messages where role:user",
    "messages where synthetic",
    "sessions | sort by started",
    "messages where role:user | sort by role",
];

fn archive() -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    f.many_sessions(ManySessions {
        prefix: "/home/op/work",
        sessions: 12,
        messages: 3,
        answered: 5,
        last: 1_800_000_000_000,
    })
    .unwrap();
    (dir, f)
}

fn program(text: &str, page_size: u64) -> Program {
    let mut opts = options();
    opts.page_size = Some(page_size);
    crate::query::compile(text, &opts).unwrap()
}

fn keys(rows: &[Value]) -> Vec<Value> {
    rows.iter().map(|row| row["_pk"].clone()).collect()
}

/// Every page of `program`, each continuing after the last row's order
/// values; `between` runs after the first page.
fn pages_after(
    f: &mut Fixture,
    program: &Program,
    mut between: impl FnMut(&mut Fixture),
) -> Vec<Value> {
    let mut after: Option<Vec<Value>> = None;
    let mut rows = vec![];
    loop {
        let page = {
            let pool = f.reads().unwrap();
            let reader = pool.reader().unwrap();
            reader.query_page(program, after.as_deref()).unwrap()
        };
        assert!(page.rows.len() as u64 <= program.page_size);
        let last = page.rows.last().cloned();
        rows.extend(page.rows);
        let Some(last) = last.filter(|_| page.more) else {
            return rows;
        };
        if after.is_none() {
            between(f);
        }
        after = Some(
            program
                .order
                .iter()
                .map(|key| last[&key.column].clone())
                .collect(),
        );
    }
}

/// Concatenated pages are the unpaged answer, in its order and each row
/// once, for every page size, whether the page reads keys first or the
/// whole relation.
#[test]
fn pages_concatenate_to_the_unpaged_answer() {
    let (_dir, mut f) = archive();
    for text in PAGED {
        let whole = {
            let pool = f.reads().unwrap();
            let reader = pool.reader().unwrap();
            keys(&reader.query_rows(&program(text, 1)).unwrap())
        };
        assert!(whole.len() > 3, "{text}: {whole:?}");
        for size in 1..=whole.len() as u64 + 1 {
            let paged = keys(&pages_after(&mut f, &program(text, size), |_| {}));
            assert_eq!(paged, whole, "{text}, page size {size}");
        }
    }
}

/// A row the query does not select, written between pages, moves no page:
/// no selected row is skipped or repeated.
#[test]
fn a_row_written_between_pages_skips_and_repeats_nothing() {
    for text in PAGED {
        let (_dir, mut f) = archive();
        let whole = {
            let pool = f.reads().unwrap();
            let reader = pool.reader().unwrap();
            keys(&reader.query_rows(&program(text, 1)).unwrap())
        };
        let paged = keys(&pages_after(&mut f, &program(text, 2), |f| {
            // A session and a message the defaults leave out, at times
            // after the first page's anchor in both orders.
            let session = f
                .session(SessionFacts {
                    id: "test:spawned",
                    kind: "subagent",
                    last: Some(1_800_000_000_000 - 90_000),
                    ..Default::default()
                })
                .unwrap();
            f.message(
                session,
                MessageFacts {
                    id: "test:spawned:m",
                    role: "user",
                    written_by: "human",
                    at: Some(1_800_000_000_000 - 300_000),
                    on_active_path: Some(false),
                    ..Default::default()
                },
            )
            .unwrap();
        }));
        assert_eq!(paged, whole, "{text}");
    }
}
