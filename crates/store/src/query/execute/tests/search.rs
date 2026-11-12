use super::*;
use crate::fixture::ManySessions;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

// The original downward-subtree query is an independent membership/rank oracle.
const DOWNWARD: &str = "WITH hits AS MATERIALIZED (
    SELECT rowid AS text_id,bm25(text_fts) AS rank FROM text_fts WHERE text_fts MATCH ?1
) SELECT id,rank FROM (
    SELECT s.id,s.last_activity_at,(SELECT min(h.rank) FROM hits h
      JOIN block b ON b.text_id=h.text_id JOIN message m ON m.pk=b.message_pk
      WHERE b.kind='text' AND (m.written_by IN ('human','model','unknown') OR m.prompt_kind='compaction_summary')
        AND m.duplicate_of IS NULL AND coalesce(m.on_active_path,1)=1
        AND m.session_pk IN (WITH RECURSIVE tree(pk) AS (
          SELECT s.pk UNION SELECT l.child_pk FROM link l
          JOIN session parent ON parent.id=l.parent_id JOIN tree ON tree.pk=parent.pk
          WHERE l.kind='spawn' AND l.child_pk IS NOT NULL
        ) SELECT pk FROM tree)) AS rank
    FROM session s WHERE s.origin='test' AND s.kind='main'
) WHERE rank IS NOT NULL ORDER BY rank,last_activity_at DESC,id";

fn archive(unrelated: usize, cycle: bool) -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let mut sessions = std::collections::BTreeMap::new();
    for (id, kind) in [
        ("root-a", "main"),
        ("root-b", "main"),
        ("leaf", "subagent"),
        ("deep", "subagent"),
        ("inactive-only", "main"),
        ("duplicate-only", "main"),
        ("harness-only", "main"),
    ] {
        let pk = fixture
            .session(SessionFacts {
                id: &format!("test:{id}"),
                kind,
                last: Some(1000),
                ..Default::default()
            })
            .unwrap();
        sessions.insert(id, pk);
    }
    fixture
        .message(
            sessions["deep"],
            MessageFacts {
                id: "deep-hit",
                role: "user",
                written_by: "human",
                text: Some("needle needle"),
                ..Default::default()
            },
        )
        .unwrap();
    fixture
        .message(
            sessions["root-b"],
            MessageFacts {
                id: "own-hit",
                role: "assistant",
                written_by: "model",
                text: Some("a longer ordinary report with one needle"),
                ..Default::default()
            },
        )
        .unwrap();
    for (id, active, written_by) in [
        ("inactive-only", Some(false), "human"),
        ("duplicate-only", Some(true), "human"),
        ("harness-only", Some(true), "harness"),
    ] {
        fixture
            .message(
                sessions[id],
                MessageFacts {
                    id,
                    role: "user",
                    written_by,
                    prompt_kind: (written_by == "harness").then_some("harness"),
                    text: Some("needle"),
                    on_active_path: active,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    fixture
        .link("test:root-a", "test:leaf", "spawn", None)
        .unwrap();
    fixture
        .link("test:leaf", "test:deep", "spawn", None)
        .unwrap();
    fixture
        .many_sessions(ManySessions {
            prefix: "/home/op/unrelated",
            sessions: unrelated,
            messages: 1,
            answered: 0,
            last: 1000,
        })
        .unwrap();
    let conn = crate::archive::open_write(&fixture.archive.index_db(), 1024).unwrap();
    conn.execute(
        "UPDATE message SET duplicate_of='deep-hit' WHERE id='duplicate-only'",
        [],
    )
    .unwrap();
    if cycle {
        // Search reachability is bounded even when retained source links cycle.
        // This test reads query rows, independently of transcript layout indexes.
        conn.execute("INSERT INTO link(parent_id,child_id,child_pk,kind,how,evidence,input_pk)
          VALUES('test:deep','test:leaf',?1,'spawn',NULL,'fixture',(SELECT pk FROM input_ref LIMIT 1))", [sessions["leaf"]]).unwrap();
    }
    drop(conn);
    (dir, fixture)
}

fn options_for_search() -> crate::query::Options {
    let mut options = options();
    options.fields = vec!["id".into()];
    options.page_size = None;
    options
}

fn result(rows: &[Value]) -> Vec<(String, f64)> {
    rows.iter()
        .map(|row| {
            (
                row["id"].as_str().unwrap().to_owned(),
                row["rank"].as_f64().unwrap(),
            )
        })
        .collect()
}

fn counted(reader: &Reader<'_>, run: impl FnOnce() -> Vec<Value>) -> (Vec<Value>, u64) {
    let steps = Arc::new(AtomicU64::new(0));
    let counter = Arc::clone(&steps);
    let conn = reader.conn().unwrap();
    conn.progress_handler(
        1,
        Some(move || {
            counter.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    let rows = run();
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    (rows, steps.load(Ordering::Relaxed))
}

fn read(fixture: &Fixture, analyze: bool) -> (u64, u64) {
    if analyze {
        crate::archive::open_write(&fixture.archive.index_db(), 1024)
            .unwrap()
            .execute_batch("ANALYZE")
            .unwrap();
    }
    let pool = crate::ReadPool::open(&fixture.archive).unwrap();
    let reader = pool.reader().unwrap();
    let program =
        crate::query::compile("sessions where kind:main needle", &options_for_search()).unwrap();
    let (actual, actual_steps) = counted(&reader, || {
        let rows = reader.query_rows(&program).unwrap();
        assert_eq!(reader.query_count(&program).unwrap(), Some((2, 0)));
        rows
    });
    let (expected, original_steps) = counted(&reader, || {
        reader
            .query(DOWNWARD, &[crate::SqlValue::Text("\"needle\"".into())])
            .unwrap()
    });
    assert_eq!(result(&actual), result(&expected));
    let mut ids = actual
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    ids.sort();
    assert_eq!(ids, ["test:root-a", "test:root-b"]);
    (actual_steps, original_steps)
}

#[test]
fn session_hits_and_ranks_follow_subagent_ancestors_including_cycles() {
    for cycle in [false, true] {
        let (_dir, fixture) = archive(20, cycle);
        read(&fixture, false);
        read(&fixture, true);
    }
}

#[test]
fn rare_word_steps_stay_flat_as_unrelated_history_grows_with_or_without_statistics() {
    let (_small_dir, small) = archive(100, false);
    let (_large_dir, large) = archive(2000, false);
    let runs = [false, true].map(|analyze| (read(&small, analyze), read(&large, analyze)));
    for (analyze, ((small_steps, old_small), (large_steps, old_large))) in
        [false, true].into_iter().zip(runs)
    {
        assert!(
            large_steps <= small_steps + small_steps / 10,
            "statistics={analyze}: {small_steps} steps over100 unrelated sessions, {large_steps} over2000"
        );
        // The oracle retains the original per-row traversal, so it also proves
        // that this cost control discriminates against the replaced query.
        assert!(
            old_large > old_small * 8,
            "statistics={analyze}: original {old_small} -> {old_large}"
        );
    }
    let ((plain_small, _), (plain_large, _)) = runs[0];
    let ((analyzed_small, _), (analyzed_large, _)) = runs[1];
    assert!(
        analyzed_small <= plain_small + plain_small / 10,
        "100 unrelated: {plain_small} -> {analyzed_small} after ANALYZE"
    );
    assert!(
        analyzed_large <= plain_large + plain_large / 10,
        "2000 unrelated: {plain_large} -> {analyzed_large} after ANALYZE"
    );
}

#[test]
fn session_text_search_remains_readable_with_first_build_indexes_deferred() {
    let (_dir, fixture) = archive(20, true);
    let write = crate::archive::open_write(&fixture.archive.index_db(), 1024).unwrap();
    for index in crate::writer::deferred_index_names() {
        write
            .execute_batch(&format!("DROP INDEX IF EXISTS {index}"))
            .unwrap();
    }
    drop(write);
    // The same independent membership/rank oracle remains exact while a
    // first build has deferred its natural-key and secondary read indexes.
    read(&fixture, false);
}

fn many_hits(blocks: usize, output: bool) -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let mut owners = Vec::new();
    for (id, kind, last) in [
        ("old", "main", 1000),
        ("new", "main", 2000),
        ("leaf", "subagent", 1000),
    ] {
        owners.push(
            fixture
                .session(SessionFacts {
                    id: &format!("test:{id}"),
                    kind,
                    last: Some(last),
                    ..Default::default()
                })
                .unwrap(),
        );
    }
    fixture
        .link("test:old", "test:leaf", "spawn", None)
        .unwrap();
    // A retained cycle must not change the owner best-score fold.
    fixture
        .link("test:leaf", "test:old", "spawn", None)
        .unwrap();
    for n in 0..=blocks {
        let owner = if n == blocks {
            owners[1]
        } else if n % 2 == 0 {
            owners[0]
        } else {
            owners[2]
        };
        let text = format!("needle token{n:06}");
        let pk = fixture
            .message(
                owner,
                MessageFacts {
                    id: &format!("test:hit-{n:06}"),
                    role: "assistant",
                    written_by: "model",
                    text: (!output).then_some(text.as_str()),
                    ..Default::default()
                },
            )
            .unwrap();
        if output {
            fixture
                .tool_call(
                    owner,
                    pk,
                    crate::fixture::CallFacts {
                        tool: "Bash",
                        action: "shell",
                        outcome: "ok",
                        output: Some(&text),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }
    (dir, fixture)
}

fn pages(reader: &Reader<'_>, program: &Program) -> Vec<Value> {
    let mut after = None;
    let mut rows = Vec::new();
    loop {
        let page = reader.query_page(program, after.as_deref()).unwrap();
        if page.rows.is_empty() {
            break;
        }
        after = page.rows.last().map(|row| {
            program
                .order
                .iter()
                .map(|key| row[&key.column].clone())
                .collect::<Vec<_>>()
        });
        rows.extend(page.rows);
        if !page.more {
            // The final cursor also has no weaker duplicate owner left behind it.
            assert!(
                reader
                    .query_page(program, after.as_deref())
                    .unwrap()
                    .rows
                    .is_empty()
            );
            break;
        }
    }
    rows
}

#[test]
fn ranked_pages_read_score_ties_and_do_not_repeat_many_hit_owners() {
    for output in [false, true] {
        let (_dir, fixture) = many_hits(384, output);
        for statistics in [false, true] {
            if statistics {
                crate::archive::open_write(&fixture.archive.index_db(), 1024)
                    .unwrap()
                    .execute_batch("ANALYZE")
                    .unwrap();
            }
            let pool = fixture.reads().unwrap();
            let reader = pool.reader().unwrap();
            let term = if output { "output:needle" } else { "needle" };
            let program = crate::query::compile(
                &format!("sessions where kind:main {term}"),
                &options_for_search(),
            )
            .unwrap();
            let mut program = program;
            program.page_size = 1;
            let expected = reader.query_rows(&program).unwrap();
            let actual = pages(&reader, &program);
            assert_eq!(result(&actual), result(&expected));
            assert_eq!(
                actual
                    .iter()
                    .map(|row| row["id"].as_str().unwrap())
                    .collect::<Vec<_>>(),
                ["test:new", "test:old"]
            );
            assert_eq!(reader.query_count(&program).unwrap(), Some((2, 0)));
            let scoped =
                crate::query::compile(&format!("id:test:new {term}"), &options_for_search())
                    .unwrap();
            assert_eq!(pages(&reader, &scoped).len(), 1);
        }
    }
}

#[test]
fn many_hit_page_and_continuation_work_grows_with_input_not_its_square() {
    let mut work = Vec::new();
    for blocks in [128, 512] {
        let (_dir, fixture) = many_hits(blocks, false);
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        let mut program =
            crate::query::compile("sessions where kind:main needle", &options_for_search())
                .unwrap();
        program.page_size = 1;
        let (rows, steps) = counted(&reader, || pages(&reader, &program));
        assert_eq!(rows.len(), 2);
        work.push(steps);
    }
    // A fourfold input grows the prefix fold roughly by N log N. The old
    // per-candidate owner reconstruction exceeded this bound already at256.
    assert!(work[1] < work[0] * 7, "128→512 blocks: {work:?}");
}

#[test]
fn first_ranked_page_does_not_expand_worse_shared_body_owners() {
    let mut work = Vec::new();
    for owners in [64, 512] {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::new(dir.path()).unwrap();
        let sessions = fixture
            .many_sessions(ManySessions {
                prefix: "/home/op/shared-body",
                sessions: owners,
                messages: 1,
                answered: 0,
                last: 1000,
            })
            .unwrap();
        let body = format!("needle {}", "ordinary ".repeat(80));
        for (n, session) in sessions.into_iter().enumerate() {
            fixture
                .message(
                    session,
                    MessageFacts {
                        id: &format!("test:weak-{n}"),
                        role: "assistant",
                        written_by: "model",
                        text: Some(&body),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        for (id, text) in [("strong", "needle"), ("runner-up", "needle second")] {
            let session = fixture
                .session(SessionFacts {
                    id: &format!("test:{id}"),
                    kind: "main",
                    last: Some(2000),
                    ..Default::default()
                })
                .unwrap();
            fixture
                .message(
                    session,
                    MessageFacts {
                        id: &format!("test:{id}-message"),
                        role: "user",
                        written_by: "human",
                        text: Some(text),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        let mut program = crate::query::compile("needle", &options_for_search()).unwrap();
        program.page_size = 1;
        // The same logical selection without the prefix driver is the old
        // full-index page, not an unpaged answer with extra hydration work.
        let mut unlimited = program.clone();
        match &mut unlimited.statements {
            crate::query::compile::Statements::Keyed(rows) => rows.search = None,
            _ => panic!("native search has keyed rows"),
        }
        for statistics in [false, true] {
            if statistics {
                crate::archive::open_write(&fixture.archive.index_db(), 1024)
                    .unwrap()
                    .execute_batch("ANALYZE")
                    .unwrap();
            }
            let pool = fixture.reads().unwrap();
            let reader = pool.reader().unwrap();
            let (actual, steps) =
                counted(&reader, || reader.query_page(&program, None).unwrap().rows);
            let (old, old_steps) = counted(&reader, || {
                reader.query_page(&unlimited, None).unwrap().rows
            });
            assert_eq!(result(&actual), result(&old));
            assert_eq!(actual[0]["id"], "test:strong");
            work.push((statistics, steps, old_steps));
        }
    }
    for statistics in [false, true] {
        let small = work.iter().find(|entry| entry.0 == statistics).unwrap();
        let large = work
            .iter()
            .rev()
            .find(|entry| entry.0 == statistics)
            .unwrap();
        assert!(
            large.1 <= small.1 + small.1 / 5,
            "statistics={statistics}: {small:?} → {large:?}"
        );
        assert!(
            large.2 > small.2 * 4,
            "unlimited selector must discriminate: {small:?} → {large:?}"
        );
    }
}
