use super::*;
use crate::fixture::ManySessions;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

// The original downward-subtree query is an independent membership/rank oracle.
fn downward() -> String {
    format!(
        "WITH hits AS MATERIALIZED ({}) {DOWNWARD}",
        crate::search::ranked_texts(crate::search::SnippetIndex::Prose, "?1", None, None)
    )
}

const DOWNWARD: &str = "SELECT id,rank FROM (
    SELECT s.id,s.last_activity_at,(SELECT min(h.rank) FROM hits h
      JOIN block b ON b.text_id=h.text_id JOIN message m ON m.pk=b.message_pk
      WHERE b.kind='text' AND (m.written_by IN ('human','model','unknown') OR m.prompt_kind='compaction_summary')
        AND m.duplicate_of IS NULL AND coalesce(m.on_active_path,1)=1
        AND m.session_pk IN (WITH RECURSIVE tree(pk) AS (
          SELECT s.pk UNION SELECT child.pk FROM session_link l
          JOIN session parent ON parent.id=l.parent_id JOIN tree ON tree.pk=parent.pk
          JOIN session child ON child.id=l.child_id
          WHERE l.kind IN ('spawn','fork','continuation','resume')
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
                text: Some("a longer plain report with one needle"),
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
        // Search reachability is bounded even when stored source links cycle.
        // This test reads query rows, independently of transcript layout indexes.
        conn.execute("INSERT INTO session_link(parent_id,child_id,kind,reason,evidence,input_pk)
          VALUES('test:deep','test:leaf','spawn',NULL,'fixture',(SELECT pk FROM input_ref LIMIT 1))", []).unwrap();
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
            .query(&downward(), &[crate::SqlValue::Text("\"needle\"".into())])
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
        // The oracle keeps the original per-row traversal, so it also proves
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
fn rare_ranked_pages_skip_a_catalog_sized_scope_probe() {
    let mut work = Vec::new();
    for unrelated in [100, 2000] {
        let (_dir, fixture) = archive(unrelated, false);
        let mut program =
            crate::query::compile("sessions where kind:main needle", &options_for_search())
                .unwrap();
        program.page_size = 1;
        assert!(matches!(
            &program.statements,
            crate::query::compile::Statements::Keyed(rows)
                if rows.search.as_ref().unwrap().scope.is_some()
        ));
        for analyzed in [false, true] {
            if analyzed {
                crate::archive::open_write(&fixture.archive.index_db(), 1024)
                    .unwrap()
                    .execute_batch("ANALYZE")
                    .unwrap();
            }
            let pool = fixture.reads().unwrap();
            let reader = pool.reader().unwrap();
            let (rows, steps) = counted(&reader, || pages(&reader, &program));
            let mut ids = rows
                .iter()
                .map(|row| row["id"].as_str().unwrap())
                .collect::<Vec<_>>();
            ids.sort();
            assert_eq!(ids, ["test:root-a", "test:root-b"]);
            work.push((unrelated, analyzed, steps));
        }
    }
    for analyzed in [false, true] {
        let small = work
            .iter()
            .find(|entry| entry.0 == 100 && entry.1 == analyzed)
            .unwrap();
        let large = work
            .iter()
            .find(|entry| entry.0 == 2000 && entry.1 == analyzed)
            .unwrap();
        assert!(
            large.2 <= small.2 + small.2 / 10,
            "rare page work: {small:?} -> {large:?}"
        );
    }
}

#[test]
fn session_text_search_remains_readable_with_build_indexes_deferred() {
    let (_dir, fixture) = archive(20, true);
    let write = crate::archive::open_write(&fixture.archive.index_db(), 1024).unwrap();
    for &index in crate::schema::BUILT_AT_END {
        write
            .execute_batch(&format!("DROP INDEX IF EXISTS {index}"))
            .unwrap();
    }
    drop(write);
    // The same independent membership/rank oracle still holds while a
    // first build has deferred its natural-key and secondary read indexes.
    read(&fixture, false);
}

fn many_hits(blocks: usize, output: bool) -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let mut sessions = Vec::new();
    for (id, kind, last) in [
        ("old", "main", 1000),
        ("new", "main", 2000),
        ("leaf", "subagent", 1000),
    ] {
        sessions.push(
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
    // A kept cycle must not change the best-score choice per session.
    fixture
        .link("test:leaf", "test:old", "spawn", None)
        .unwrap();
    for n in 0..=blocks {
        let session = if n == blocks {
            sessions[1]
        } else if n % 2 == 0 {
            sessions[0]
        } else {
            sessions[2]
        };
        let text = format!("needle token{n:06}");
        let pk = fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("test:hit-{n:06}"),
                    role: "assistant",
                    written_by: "model",
                    text: (!output).then_some(text.as_str()),
                    at: Some(if n == blocks { 2000 } else { 1000 }),
                    ..Default::default()
                },
            )
            .unwrap();
        if output {
            fixture
                .tool_call(
                    session,
                    pk,
                    crate::fixture::CallFacts {
                        tool: "Bash",
                        action: "shell",
                        outcome: "ok",
                        output: Some(&text),
                        at: Some(if n == blocks { 2000 } else { 1000 }),
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
            // The final cursor also has no weaker duplicate session left behind it.
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

fn shared_body_outside_scope(fixture: &mut Fixture, message: &str, count: usize) {
    fixture
        .many_sessions(ManySessions {
            prefix: "/synthetic/outside-search-scope",
            sessions: count,
            messages: 1,
            answered: 0,
            last: 1000,
        })
        .unwrap();
    let write = crate::archive::open_write(&fixture.archive.index_db(), 1024).unwrap();
    write
        .execute(
            "INSERT INTO block(message_pk,session_pk,position,kind,text_id,content_group,content_start,content_length,data_length) \
         SELECT m.pk,m.session_pk,0,source.kind,source.text_id,source.content_group,source.content_start,source.content_length,source.data_length \
         FROM message m JOIN session s ON s.pk=m.session_pk \
         CROSS JOIN block source JOIN message source_message ON source_message.pk=source.message_pk \
         WHERE s.id LIKE 'test:many-%' AND source_message.id=?1 \
         AND source.kind IN ('text','tool_result')",
            [message],
        )
        .unwrap();
}

fn search_message(
    fixture: &mut Fixture,
    session: i64,
    id: &str,
    at: Option<i64>,
    text: &str,
    output: bool,
) {
    let message = fixture
        .message(
            session,
            MessageFacts {
                id,
                at,
                role: "assistant",
                written_by: "model",
                text: (!output).then_some(text),
                ..Default::default()
            },
        )
        .unwrap();
    if output {
        fixture
            .tool_call(
                session,
                message,
                crate::fixture::CallFacts {
                    tool: "Bash",
                    action: "shell",
                    outcome: "ok",
                    at,
                    output: Some(text),
                    ..Default::default()
                },
            )
            .unwrap();
    }
}

#[test]
fn recency_scope_and_index_agree_across_batches_shared_children_and_unknown_times() {
    use crate::query::compile::Statements;
    for output in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::new(dir.path()).unwrap();
        let mut sessions = std::collections::BTreeMap::new();
        for (id, kind, last) in [
            ("a", "main", Some(5000)),
            ("b", "main", Some(4000)),
            ("c", "main", Some(3000)),
            ("d", "main", Some(2000)),
            ("e", "main", Some(2000)),
            ("f", "main", Some(500)),
            ("g", "main", None),
            ("child", "subagent", Some(1000)),
        ] {
            sessions.insert(
                id,
                fixture
                    .session(SessionFacts {
                        id: &format!("test:{id}"),
                        kind,
                        last,
                        ..Default::default()
                    })
                    .unwrap(),
            );
        }
        for parent in ["a", "d", "e"] {
            fixture
                .link(&format!("test:{parent}"), "test:child", "spawn", None)
                .unwrap();
        }
        // Kept cycles do not change best-source membership or batches.
        fixture.link("test:child", "test:a", "spawn", None).unwrap();
        search_message(
            &mut fixture,
            sessions["child"],
            "shared-hit",
            Some(1000),
            "needle",
            output,
        );
        search_message(
            &mut fixture,
            sessions["b"],
            "miss-b",
            Some(4000),
            "other",
            output,
        );
        search_message(
            &mut fixture,
            sessions["c"],
            "miss-c",
            Some(3000),
            "other",
            output,
        );
        search_message(
            &mut fixture,
            sessions["f"],
            "own-f",
            Some(500),
            "needle own",
            output,
        );
        search_message(
            &mut fixture,
            sessions["g"],
            "own-g",
            None,
            "needle own",
            output,
        );
        shared_body_outside_scope(&mut fixture, "shared-hit", 512);
        let identities = ["a", "b", "c", "d", "e", "f", "g"]
            .map(|id| format!("id:test:{id}"))
            .join(" OR ");
        let term = if output { "output:needle" } else { "needle" };
        for (query, page_size, expected_ids) in [
            (
                format!(
                    "sessions where kind:main ({identities}) {term} | sort by last_active desc"
                ),
                3,
                // Newest first is the default order read backwards: e and d
                // share a time, so the greater ID comes first.
                vec!["test:a", "test:e", "test:d", "test:f", "test:g"],
            ),
            (
                format!(
                    "messages where (id:shared-hit OR id:miss-b OR id:miss-c OR id:own-f OR id:own-g) {term} | sort by at desc"
                ),
                2,
                vec!["shared-hit", "own-f", "own-g"],
            ),
        ] {
            let mut program = crate::query::compile(&query, &options_for_search()).unwrap();
            program.page_size = page_size;
            let scope = match &program.statements {
                Statements::Keyed(rows) => rows
                    .search
                    .as_ref()
                    .unwrap()
                    .scope
                    .as_ref()
                    .unwrap()
                    .clone(),
                _ => panic!("direct recency sort stays keyed"),
            };
            assert!(scope.recent.is_some());
            let mut indexed = program.clone();
            if let Statements::Keyed(rows) = &mut indexed.statements {
                rows.search = None;
            }
            let mut scoped = indexed.clone();
            if let Statements::Keyed(rows) = &mut scoped.statements {
                rows.keys = scope.by_scope.clone();
            }
            for analyzed in [false, true] {
                if analyzed {
                    crate::archive::open_write(&fixture.archive.index_db(), 1024)
                        .unwrap()
                        .execute_batch("ANALYZE")
                        .unwrap();
                }
                let pool = fixture.reads().unwrap();
                let reader = pool.reader().unwrap();
                assert_eq!(
                    reader
                        .search_driver(
                            &scope.blocks,
                            &crate::selection::values(&program.params),
                            if output {
                                SnippetIndex::Output
                            } else {
                                SnippetIndex::Prose
                            },
                            "needle",
                            true
                        )
                        .unwrap(),
                    SearchDriver::Recency
                );
                let expected = reader.query_rows(&indexed).unwrap();
                assert_eq!(
                    expected
                        .iter()
                        .map(|row| row["id"].as_str().unwrap())
                        .collect::<Vec<_>>(),
                    expected_ids
                );
                for candidate in [&program, &scoped] {
                    let mut actual = pages(&reader, candidate);
                    assert_eq!(result(&actual), result(&expected));
                    assert_eq!(
                        reader.query_count(candidate).unwrap(),
                        Some((expected_ids.len() as u64, 0))
                    );
                    let mut expected = expected.clone();
                    crate::query::execute::snippets(&reader, candidate, &mut actual).unwrap();
                    crate::query::execute::snippets(&reader, &program, &mut expected).unwrap();
                    assert_eq!(
                        actual.iter().map(|row| &row["snippet"]).collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|row| &row["snippet"])
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn recent_page_skips_older_matching_sessions() {
    use crate::query::compile::Statements;
    let mut work = Vec::new();
    for outside in [512, 4096] {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::new(dir.path()).unwrap();
        for (id, at, text) in [
            ("newest", 5000, "needle"),
            ("runner-up", 4000, "needle second"),
        ] {
            let session = fixture
                .session(SessionFacts {
                    id: &format!("test:{id}"),
                    kind: "main",
                    last: Some(at),
                    ..Default::default()
                })
                .unwrap();
            search_message(
                &mut fixture,
                session,
                &format!("test:{id}-message"),
                Some(at),
                text,
                false,
            );
        }
        shared_body_outside_scope(&mut fixture, "test:newest-message", outside);
        for unit in ["sessions", "messages"] {
            let time = if unit == "sessions" {
                "last_active"
            } else {
                "at"
            };
            let mut program = crate::query::compile(
                &format!("{unit} where needle | sort by {time} desc"),
                &options_for_search(),
            )
            .unwrap();
            program.page_size = 1;
            let mut indexed = program.clone();
            if let Statements::Keyed(rows) = &mut indexed.statements {
                rows.search = None;
            }
            for analyzed in [false, true] {
                if analyzed {
                    crate::archive::open_write(&fixture.archive.index_db(), 1024)
                        .unwrap()
                        .execute_batch("ANALYZE")
                        .unwrap();
                }
                let pool = fixture.reads().unwrap();
                let reader = pool.reader().unwrap();
                let (actual, steps) =
                    counted(&reader, || reader.query_page(&program, None).unwrap().rows);
                let (expected, old_steps) =
                    counted(&reader, || reader.query_page(&indexed, None).unwrap().rows);
                assert_eq!(result(&actual), result(&expected));
                assert_eq!(
                    actual[0]["id"],
                    if unit == "sessions" {
                        "test:newest"
                    } else {
                        "test:newest-message"
                    }
                );
                work.push((outside, unit, analyzed, steps, old_steps));
            }
        }
    }
    for unit in ["sessions", "messages"] {
        for analyzed in [false, true] {
            let small = work
                .iter()
                .find(|row| row.0 == 512 && row.1 == unit && row.2 == analyzed)
                .unwrap();
            let large = work
                .iter()
                .find(|row| row.0 == 4096 && row.1 == unit && row.2 == analyzed)
                .unwrap();
            assert!(
                large.3 <= small.3 + small.3 / 5,
                "recency work: {small:?} -> {large:?}"
            );
            assert!(
                large.4 > small.4 * 4,
                "full-index control: {small:?} -> {large:?}"
            );
        }
    }
}

#[test]
fn scope_and_index_plans_agree_on_rows_sources_counts_and_continuations() {
    for output in [false, true] {
        let (_dir, mut fixture) = many_hits(32, output);
        shared_body_outside_scope(&mut fixture, "test:hit-000032", 512);
        for analyzed in [false, true] {
            if analyzed {
                crate::archive::open_write(&fixture.archive.index_db(), 1024)
                    .unwrap()
                    .execute_batch("ANALYZE")
                    .unwrap();
            }
            let pool = fixture.reads().unwrap();
            let reader = pool.reader().unwrap();
            let term = if output { "output:needle" } else { "needle" };
            for query in [
                format!("sessions where (id:test:old OR id:test:new) {term}"),
                format!("sessions where id:test:old {term}"),
                format!("sessions where {term} | where id:test:new"),
                format!("messages where {term} id:test:hit-000032"),
            ] {
                let mut program = crate::query::compile(&query, &options_for_search()).unwrap();
                program.page_size = 1;
                let by_scope = match &program.statements {
                    crate::query::compile::Statements::Keyed(rows) => rows
                        .search
                        .as_ref()
                        .unwrap()
                        .scope
                        .as_ref()
                        .unwrap()
                        .by_scope
                        .clone(),
                    _ => panic!("scoped search has keyed rows"),
                };
                let mut scoped = program.clone();
                if let crate::query::compile::Statements::Keyed(rows) = &mut scoped.statements {
                    rows.keys = by_scope;
                    rows.search = None;
                }
                let mut indexed = program.clone();
                if let crate::query::compile::Statements::Keyed(rows) = &mut indexed.statements {
                    rows.search = None;
                }
                let expected = reader.query_rows(&indexed).unwrap();
                for candidate in [&program, &scoped] {
                    assert_eq!(
                        result(&reader.query_rows(candidate).unwrap()),
                        result(&expected),
                        "whole selection: {query}"
                    );
                    let mut actual = pages(&reader, candidate);
                    assert_eq!(result(&actual), result(&expected), "{query}");
                    assert_eq!(
                        reader.query_count(candidate).unwrap(),
                        Some((expected.len() as u64, 0))
                    );
                    let mut expected = expected.clone();
                    crate::query::execute::snippets(&reader, candidate, &mut actual).unwrap();
                    crate::query::execute::snippets(&reader, &program, &mut expected).unwrap();
                    assert_eq!(
                        actual.iter().map(|row| &row["snippet"]).collect::<Vec<_>>(),
                        expected
                            .iter()
                            .map(|row| &row["snippet"])
                            .collect::<Vec<_>>(),
                        "{query}"
                    );
                }
            }
        }
    }
}

/// docs/search.md, The search leg: the hit-set size picks the plan, and
/// each plan is its own statement returning the rows, ranks and order of a
/// reference that ranks every hit at once.
#[test]
fn a_search_runs_each_plan_as_its_own_statement() {
    use crate::query::compile::Statements;
    for (shared, query, plan) in [
        (0, "sessions where id:test:new needle", SearchDriver::Index),
        (
            512,
            "sessions where id:test:new needle",
            SearchDriver::Scope,
        ),
        (
            512,
            "sessions where (id:test:old OR id:test:new) needle | sort by last_active desc",
            SearchDriver::Recency,
        ),
        (
            512,
            "messages where (id:test:hit-000031 OR id:test:hit-000032) needle | sort by at desc",
            SearchDriver::Recency,
        ),
        (
            512,
            "messages where id:test:hit-000032 needle",
            SearchDriver::Scope,
        ),
    ] {
        let (_dir, mut fixture) = many_hits(32, false);
        if shared > 0 {
            shared_body_outside_scope(&mut fixture, "test:hit-000032", shared);
        }
        let mut program = crate::query::compile(query, &options_for_search()).unwrap();
        program.page_size = 1;
        let Statements::Keyed(rows) = &program.statements else {
            panic!("a ranked search has keyed rows: {query}");
        };
        let search = rows.search.clone().expect("search plans");
        let scope = search.scope.as_ref().expect("a scope can drive");
        let mut reference = program.clone();
        if let Statements::Keyed(rows) = &mut reference.statements {
            rows.search = None;
        }
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader
                .search_driver(
                    &scope.blocks,
                    &crate::selection::values(&program.params),
                    search.index,
                    &search.expression,
                    scope.recent.is_some(),
                )
                .unwrap(),
            plan,
            "{query}"
        );
        let expected = reader.query_rows(&reference).unwrap();
        assert!(!expected.is_empty(), "{query}");
        assert_eq!(
            result(&pages(&reader, &program)),
            result(&expected),
            "{query}"
        );
        assert_eq!(
            reader.query_count(&program).unwrap(),
            Some((expected.len() as u64, 0)),
            "{query}"
        );
        // The index plan reads the best `prefix` texts, its own input after
        // the query's parameters; the whole index gives the reference.
        let mut params = crate::selection::values(&program.params);
        params.push(crate::SqlValue::Int(-1));
        let mut by_index = reader.query(&search.by_index, &params).unwrap();
        by_index.sort_by(|a, b| a["_pk"].as_i64().cmp(&b["_pk"].as_i64()));
        let mut whole = reader
            .query(&rows.keys, &crate::selection::values(&program.params))
            .unwrap();
        whole.sort_by(|a, b| a["_pk"].as_i64().cmp(&b["_pk"].as_i64()));
        assert_eq!(by_index, whole, "{query}");
    }
}

#[test]
fn a_narrow_search_scope_does_not_expand_shared_bodies_outside_it() {
    let mut work = Vec::new();
    for outside in [512, 4096] {
        let (_dir, mut fixture) = many_hits(32, false);
        shared_body_outside_scope(&mut fixture, "test:hit-000032", outside);
        let program = crate::query::compile("id:test:new needle", &options_for_search()).unwrap();
        let mut unlimited = program.clone();
        if let crate::query::compile::Statements::Keyed(rows) = &mut unlimited.statements {
            rows.search = None;
        }
        for analyzed in [false, true] {
            if analyzed {
                crate::archive::open_write(&fixture.archive.index_db(), 1024)
                    .unwrap()
                    .execute_batch("ANALYZE")
                    .unwrap();
            }
            let pool = fixture.reads().unwrap();
            let reader = pool.reader().unwrap();
            let (rows, steps) =
                counted(&reader, || reader.query_page(&program, None).unwrap().rows);
            let (old, old_steps) = counted(&reader, || {
                reader.query_page(&unlimited, None).unwrap().rows
            });
            assert_eq!(result(&rows), result(&old));
            assert_eq!(rows[0]["id"], "test:new");
            work.push((outside, analyzed, steps, old_steps));
        }
    }
    for analyzed in [false, true] {
        let small = work
            .iter()
            .find(|entry| entry.0 == 512 && entry.1 == analyzed)
            .unwrap();
        let large = work
            .iter()
            .find(|entry| entry.0 == 4096 && entry.1 == analyzed)
            .unwrap();
        assert!(
            large.2 <= small.2 + small.2 / 5,
            "scope work: {small:?} -> {large:?}"
        );
        assert!(
            large.3 > small.3 * 4,
            "full-index control: {small:?} -> {large:?}"
        );
    }
}

#[test]
fn ranked_pages_do_not_repeat_many_hit_sessions() {
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
        assert_eq!(
            rows.iter()
                .map(|row| row["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["test:new", "test:old"],
            "{blocks} blocks"
        );
        work.push(steps);
    }
    // A fourfold input grows the prefix fold roughly by N log N. The old
    // per-candidate session reconstruction exceeded this bound already at256.
    assert!(work[1] < work[0] * 7, "128→512 blocks: {work:?}");
}

#[test]
fn first_ranked_page_skips_worse_shared_body_sessions() {
    let mut work = Vec::new();
    for outside in [64, 512] {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::new(dir.path()).unwrap();
        let sessions = fixture
            .many_sessions(ManySessions {
                prefix: "/home/op/shared-body",
                sessions: outside,
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
            _ => panic!("search has keyed rows"),
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

fn assert_hit_sources(fixture: &Fixture, expected: &[(&str, &str)]) {
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut program = crate::query::compile("needle", &options_for_search()).unwrap();
    program.page_size = 1;
    let mut rows = pages(&reader, &program);
    crate::query::execute::snippets(&reader, &program, &mut rows).unwrap();
    let actual = rows
        .iter()
        .map(|row| {
            (
                row["id"].as_str().unwrap(),
                row["snippet"]["message"].as_str().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn relevance_ties_use_the_chosen_message_and_snippets_keep_its_source() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let active = fixture
        .session(SessionFacts {
            id: "test:activity",
            kind: "main",
            last: Some(9000),
            ..Default::default()
        })
        .unwrap();
    let matching = fixture
        .session(SessionFacts {
            id: "test:matching",
            kind: "main",
            last: Some(2000),
            ..Default::default()
        })
        .unwrap();
    for (session, id, text, at, author, prompt) in [
        (
            active,
            "test:old-hit",
            "needle",
            1000,
            "human",
            Some("prose"),
        ),
        (
            active,
            "test:activity-tail",
            "unrelated activity",
            9000,
            "model",
            None,
        ),
        (
            matching,
            "test:recent-hit",
            "needle",
            2000,
            "human",
            Some("prose"),
        ),
        (
            active,
            "test:harness-copy",
            "needle",
            10000,
            "harness",
            Some("harness"),
        ),
    ] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: if author == "model" {
                        "assistant"
                    } else {
                        "user"
                    },
                    written_by: author,
                    prompt_kind: prompt,
                    text: Some(text),
                    at: Some(at),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    assert_hit_sources(
        &fixture,
        &[
            ("test:matching", "test:recent-hit"),
            ("test:activity", "test:old-hit"),
        ],
    );
    let child = fixture
        .session(SessionFacts {
            id: "test:child",
            kind: "subagent",
            last: Some(3000),
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            child,
            MessageFacts {
                id: "test:child-hit",
                role: "assistant",
                written_by: "model",
                text: Some("needle"),
                at: Some(3000),
                ..Default::default()
            },
        )
        .unwrap();
    fixture
        .link("test:activity", "test:child", "spawn", None)
        .unwrap();
    assert_hit_sources(
        &fixture,
        &[
            ("test:activity", "test:child-hit"),
            ("test:matching", "test:recent-hit"),
        ],
    );
    fixture
        .message(
            active,
            MessageFacts {
                id: "test:summary-hit",
                role: "user",
                written_by: "harness",
                prompt_kind: Some("compaction_summary"),
                text: Some("needle"),
                at: Some(11000),
                ..Default::default()
            },
        )
        .unwrap();
    assert_hit_sources(
        &fixture,
        &[
            ("test:activity", "test:summary-hit"),
            ("test:matching", "test:recent-hit"),
        ],
    );
}

#[test]
fn snippet_reuse_does_not_rescan_many_hit_sessions() {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    let mut work = Vec::new();
    for blocks in [128, 512] {
        let (_dir, fixture) = many_hits(blocks, false);
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        let program = crate::query::compile("needle", &options_for_search()).unwrap();
        let rows = reader.query_rows(&program).unwrap();
        let mut fallback = rows.clone();
        for row in &mut fallback {
            row.as_object_mut().unwrap().remove("_hit_block");
        }
        let conn = reader.conn().unwrap();
        conn.authorizer(Some(|context: AuthContext<'_>| {
            if matches!(context.action,AuthAction::Read { table_name, .. }
                if table_name.starts_with("text_fts") || table_name.starts_with("output_fts"))
            {
                Authorization::Deny
            } else {
                Authorization::Allow
            }
        }))
        .unwrap();
        let (reused, steps) = counted(&reader, || {
            let mut rows = rows;
            crate::query::execute::snippets(&reader, &program, &mut rows).unwrap();
            rows
        });
        conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
            .unwrap();
        let (fallback, old_steps) = counted(&reader, || {
            crate::query::execute::snippets(&reader, &program, &mut fallback).unwrap();
            fallback
        });
        assert_eq!(
            reused.iter().map(|row| &row["snippet"]).collect::<Vec<_>>(),
            fallback
                .iter()
                .map(|row| &row["snippet"])
                .collect::<Vec<_>>()
        );
        work.push((steps, old_steps));
    }
    assert!(
        work[1].0 <= work[0].0 + work[0].0 / 5,
        "selected-source work: {work:?}"
    );
}

#[test]
fn ranking_reuses_one_score_for_the_rank_field() {
    let (_dir, fixture) = many_hits(32, false);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let program = crate::query::compile("needle", &options_for_search()).unwrap();
    let keys = match &program.statements {
        crate::query::compile::Statements::Keyed(rows) => &rows.keys,
        _ => panic!("ranked keyed rows"),
    };
    for row in reader
        .query(keys, &crate::selection::values(&program.params))
        .unwrap()
    {
        let score = &row["_rank"];
        assert!(score.as_f64().is_some_and(|score| score < 0.0));
        assert_eq!(
            row.as_object()
                .unwrap()
                .values()
                .filter(|value| *value == score)
                .count(),
            1,
            "the score is one stored value, reused by ordering and presentation: {row}"
        );
    }
    let mut expected = result(&reader.query_rows(&program).unwrap());
    expected.sort_by(|left, right| left.0.cmp(&right.0));
    for query in [
        "needle",
        "needle | sort by id",
        "needle | sort by last_active desc",
    ] {
        let projected = crate::query::compile(query, &options_for_search()).unwrap();
        let mut actual = reader.query_rows(&projected).unwrap();
        for row in &mut actual {
            present(&projected, row).unwrap();
        }
        // Explicit sort keys can change row order when scores tie. Every
        // identity must retain the same single stored score under each sort.
        let mut scores = result(&actual);
        scores.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(scores, expected, "{query}");
    }
}

#[test]
fn recency_sort_keeps_a_source_order_condition() {
    let (_dir, fixture) = many_hits(32, false);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let condition = "messages where before(messages where id:test:hit-000032) within 2s";
    for query in [
        condition.to_owned(),
        format!("{condition} | sort by at desc"),
    ] {
        let program = crate::query::compile(&query, &options_for_search()).unwrap();
        let rows = reader.query_rows(&program).unwrap();
        assert_eq!(rows.len(), 32, "{query}");
        assert!(
            rows.iter().all(|row| row["id"] != "test:hit-000032"),
            "{query}"
        );
    }
}
