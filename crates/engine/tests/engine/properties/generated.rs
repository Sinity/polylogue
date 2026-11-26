//! The stored view of generated Claude Code activity equals the view the
//! docs say Polylogue shows for it, however its files arrive: in any order,
//! whole or in appends cut at any byte, with restarts and index rebuilds
//! between the steps. Idempotent ingest, append equivalence, order
//! independence and rebuild equivalence are deliveries of that statement;
//! beside it, the search index and the record are compared with a delivery
//! of the files whole.

use polylogue_store::{Archive, ReadPool};
use polylogue_testkit::activity::{Part, ToolResult};
use polylogue_testkit::generate::{activity, delivered_activity};
use polylogue_testkit::{
    Activity, Home, Schedule, Step, Transcript, View, delete_index, store_with,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;
use std::fs;

/// What the search indexes hold, by content rather than row id: for each
/// index the hashes of the texts in it, and each term's document and
/// occurrence counts over every full-text file. A first build fills the
/// files while it composes and a live pass through the search queue; both
/// must give the same index.
fn search_index(w: &Home) -> Vec<serde_json::Value> {
    let archive = Archive::at(&w.root);
    let conn = polylogue_store::archive::open_read(&archive.index_db()).unwrap();
    let files = conn
        .prepare("SELECT name FROM pragma_database_list WHERE name LIKE 'search_%' ORDER BY name")
        .unwrap()
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    assert!(!files.is_empty());
    let mut out = Vec::new();
    for fts in ["text_fts", "output_fts"] {
        let mut keys = Vec::new();
        let mut vocabulary = Vec::new();
        for file in &files {
            keys.push(format!("SELECT rowid FROM {file}.{fts} WHERE rowid > 0"));
            let vocab = format!("vocab_{file}_{fts}");
            conn.execute_batch(&format!(
                "CREATE VIRTUAL TABLE temp.{vocab} USING fts5vocab({file}, {fts}, 'row')"
            ))
            .unwrap();
            vocabulary.push(format!("SELECT term, doc, cnt FROM temp.{vocab}"));
        }
        let mut rows = |sql: String| {
            let mut statement = conn.prepare(&sql).unwrap();
            let names: Vec<String> = statement
                .column_names()
                .iter()
                .map(|name| (*name).to_owned())
                .collect();
            let found = statement
                .query_map([], |row| {
                    let mut value = serde_json::Map::new();
                    for (n, name) in names.iter().enumerate() {
                        let cell = match row.get_ref(n)? {
                            rusqlite::types::ValueRef::Integer(i) => serde_json::json!(i),
                            rusqlite::types::ValueRef::Text(t) => {
                                serde_json::json!(String::from_utf8_lossy(t))
                            }
                            other => serde_json::json!(format!("{other:?}")),
                        };
                        value.insert(name.clone(), cell);
                    }
                    Ok(serde_json::Value::Object(value))
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            out.extend(found);
        };
        rows(format!(
            "SELECT '{fts}' AS idx, hex(t.hash) AS h FROM text t
             WHERE t.id IN ({}) ORDER BY h",
            keys.join(" UNION ALL ")
        ));
        rows(format!(
            "SELECT '{fts} terms' AS idx, term, sum(doc) AS doc, sum(cnt) AS cnt
             FROM ({}) GROUP BY term ORDER BY term",
            vocabulary.join(" UNION ALL ")
        ));
    }
    out
}

/// Row count of every archive.db table.
fn archive_row_counts(w: &Home) -> Vec<(String, i64)> {
    let pool = ReadPool::open(&Archive::at(&w.root)).unwrap();
    let r = pool.reader().unwrap();
    r.query(
        "SELECT name FROM a.sqlite_master WHERE type = 'table' ORDER BY name",
        &[],
    )
    .unwrap()
    .iter()
    .map(|row| {
        let name = row["name"].as_str().unwrap().to_owned();
        let n = r
            .query(&format!("SELECT COUNT(*) AS n FROM a.\"{name}\""), &[])
            .unwrap()[0]["n"]
            .as_i64()
            .unwrap();
        (name, n)
    })
    .collect()
}

/// Two resumes of one session, one copying it whole and one copying a
/// resume of it, store each record once: in the session it came from.
#[test]
fn a_copied_record_is_stored_once() {
    let at = 1_772_359_200_000;
    let mut parent = Transcript::new("5e551000-0000-4000-8000-000000000001", "/home/op/morrow");
    parent.prompt("a1", None, at, "parse the dates");
    parent.response(
        "a2",
        Some("a1"),
        at + 1_000,
        vec![
            Part::text("running them"),
            Part::call("toolu_1", "cargo test"),
        ],
    );
    parent.results(
        "a3",
        Some("a2"),
        at + 2_000,
        vec![ToolResult::error("toolu_1", "one test fails")],
    );
    parent.response("a4", Some("a3"), at + 3_000, vec![Part::text("fixed")]);
    let mut first = Transcript::resuming(
        "5e551000-0000-4000-8000-000000000002",
        "/home/op/morrow",
        0,
        4,
    );
    first.prompt("b1", Some("a4"), at + 90_000, "and the times");
    let mut second = Transcript::resuming(
        "5e551000-0000-4000-8000-000000000003",
        "/home/op/morrow",
        1,
        5,
    );
    second.prompt("c1", Some("b1"), at + 180_000, "grün");
    let activity = Activity {
        sessions: vec![parent, first, second],
    };
    let home = Home::empty();
    let files = activity.claude_code_files();
    home.deliver(&files, &Schedule::whole(&files));
    let view = View::stored(&home);
    assert_eq!(view, activity.expected());
    let stored: Vec<(&str, Vec<&str>)> = view
        .sessions
        .iter()
        .map(|s| {
            (
                s.id.as_str(),
                s.messages.iter().map(|m| m.id.as_str()).collect(),
            )
        })
        .collect();
    assert_eq!(
        stored,
        [
            (
                "claude-code:5e551000-0000-4000-8000-000000000001",
                vec![
                    "claude-code:a1",
                    "claude-code:a2",
                    "claude-code:a3",
                    "claude-code:a4"
                ]
            ),
            (
                "claude-code:5e551000-0000-4000-8000-000000000002",
                vec!["claude-code:b1"]
            ),
            (
                "claude-code:5e551000-0000-4000-8000-000000000003",
                vec!["claude-code:c1"]
            ),
        ]
    );
    assert_eq!(
        view.sessions[2].read,
        ["a1", "a2", "a3", "a4", "b1", "c1"].map(|id| format!("claude-code:{id}"))
    );
}

/// A file whose first line arrives in three appends, each stored before the
/// next, is stored as one record when it completes (docs/archive.md, an
/// incomplete suffix waits at its recorded byte length).
#[test]
fn a_first_line_written_in_three_pieces_is_stored() {
    let at = 1_772_359_200_000;
    let mut session = Transcript::new("5e551000-0000-4000-8000-000000000001", "/home/op/morrow");
    session.prompt("a1", None, at, "parse the dates");
    session.response("a2", Some("a1"), at + 1_000, vec![Part::text("done")]);
    let activity = Activity {
        sessions: vec![session],
    };
    let files = activity.claude_code_files();
    let end = files[0].bytes.len();
    let schedule = Schedule {
        chunk: polylogue_testkit::WHOLE_FILES,
        steps: vec![
            Step::Write { file: 0, end: 10 },
            Step::Store,
            Step::Write { file: 0, end: 20 },
            Step::Store,
            Step::Write { file: 0, end },
            Step::Store,
        ],
    };
    let home = Home::empty();
    drop(home.deliver(&files, &schedule));
    assert_eq!(View::stored(&home), activity.expected());
}

proptest! {
    // Seeds of past failures are kept beside this file; cargo runs
    // integration tests from the package directory.
    #![proptest_config(ProptestConfig {
        cases: 16,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(
            "tests/engine/properties/regressions.txt",
        ))),
        ..ProptestConfig::default()
    })]

    /// However the files arrive, the stored view is the expected one; another
    /// pass then reads nothing and changes nothing, and the search index is
    /// the one a delivery of the files whole builds.
    #[test]
    fn the_stored_view_is_the_expected_view((activity, files, schedule) in delivered_activity()) {
        let expected = activity.expected();
        let home = Home::empty();
        let (mut engine, _lock) = home.deliver(&files, &schedule);
        prop_assert_eq!(&View::stored(&home), &expected);
        let again = store_with(&mut engine, &[]);
        prop_assert_eq!((again.inputs_changed, again.bytes_read), (0, 0));
        prop_assert_eq!(&View::stored(&home), &expected);
        drop((engine, _lock));

        let whole = Home::empty();
        drop(whole.deliver(&files, &Schedule::whole(&files)));
        prop_assert_eq!(&View::stored(&whole), &expected);
        prop_assert_eq!(search_index(&home), search_index(&whole));
    }

    /// Deleting the index and storing again gives the expected view, even
    /// with the files gone, and adds no row to the record but the
    /// observation that each file went.
    #[test]
    fn a_rebuild_gives_the_expected_view_and_adds_nothing(activity in activity(), files_gone in any::<bool>()) {
        let files = activity.claude_code_files();
        let home = Home::empty();
        drop(home.deliver(&files, &Schedule::whole(&files)));
        let before = archive_row_counts(&home);
        if files_gone {
            for file in &files {
                fs::remove_file(home.path_of(file)).unwrap();
            }
        }
        delete_index(&home.root);
        home.store();
        prop_assert_eq!(View::stored(&home), activity.expected());
        let expected: Vec<(String, i64)> = before
            .into_iter()
            .map(|(table, n)| {
                let gone = if files_gone && table == "observation" { files.len() as i64 } else { 0 };
                (table, n + gone)
            })
            .collect();
        prop_assert_eq!(archive_row_counts(&home), expected);
    }
}
