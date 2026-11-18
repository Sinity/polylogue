use super::*;
use std::ops::ControlFlow;

fn copied_event(copy: &SessionId) -> Event {
    Event {
        session: Some(copy.clone()),
        kind: "permission_mode".into(),
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: None,
        range: Some((7, 8)),
        data: serde_json::json!({"mode":"plan"}),
    }
}

fn selected(w: &TestArchive, sessions: &[SessionId]) -> Vec<(String, Vec<Event>)> {
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    sessions
        .iter()
        .map(|session| {
            let mut events = Vec::new();
            reader
                .visit_event_records(session, 0, |record| {
                    events.push(record.event);
                    ControlFlow::Continue(())
                })
                .unwrap();
            (session.to_string(), events)
        })
        .collect()
}

fn move_events(restart_after_install: bool) -> Vec<(String, Vec<Event>)> {
    let mut w = test_archive();
    let copy = SessionId::parse("ai-studio:content-copy").unwrap();
    let known = SessionId::parse("ai-studio:drive-file").unwrap();
    let ambiguous = SessionId::parse("ai-studio:other-file").unwrap();
    let mut input = batch("copied", Route::Capture, 0, vec![], &[]);
    input.parse[0]
        .output
        .sessions
        .push(Session::new(copy.clone(), SessionKind::Main));
    input.parse[0].output.events.push(copied_event(&copy));
    w.store_reading(input, Some(&SOURCE_RECORD)).unwrap();
    w.writer.derive().unwrap();
    if restart_after_install {
        w.writer.stop();
        w.archive = w.archive.reopened();
        w.writer = Writer::start(&w.archive).unwrap();
    }
    let declare = |input: &str, id: &SessionId| {
        let mut input = batch(input, Route::Live, 0, vec![], &[]);
        let mut session = Session::new(id.clone(), SessionKind::Main);
        session.aliases.push(copy.clone());
        input.parse[0].output.sessions.push(session);
        input
    };
    w.store_reading(declare("native", &known), Some(&SOURCE_RECORD))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let moved = selected(&w, &[copy.clone(), known.clone()]);
    assert!(moved[0].1.is_empty());
    let mut expected = copied_event(&copy);
    expected.session = Some(known.clone());
    assert_eq!(moved[1].1, [expected]);
    w.store_reading(declare("ambiguous", &ambiguous), Some(&SOURCE_RECORD))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    w.writer.derive().unwrap();
    let released = selected(&w, &[copy.clone(), known, ambiguous]);
    assert_eq!(released[0].1, [copied_event(&copy)]);
    assert!(released[1].1.is_empty());
    assert!(released[2].1.is_empty());
    released
}

#[test]
fn installed_build_moves_copy_events_like_a_restarted_writer() {
    assert_eq!(move_events(false), move_events(true));
}

#[cfg(target_os = "linux")]
#[test]
fn build_statement_rollback_retains_file_spill_for_larger_updates() {
    use crate::ConnExt as _;

    fn writes() -> u64 {
        std::fs::read_to_string("/proc/thread-self/io")
            .unwrap()
            .lines()
            .find_map(|line| line.strip_prefix("syscw: "))
            .unwrap()
            .parse()
            .unwrap()
    }

    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&directory.path().join("archive")).unwrap();
    let mut conn = crate::archive::index_build::open_writer(&archive).unwrap();
    assert_eq!(
        conn.scalar::<String>("PRAGMA journal_mode", []).unwrap(),
        "persist"
    );
    assert_eq!(conn.scalar::<i64>("PRAGMA temp_store", []).unwrap(), 1);
    assert_eq!(conn.scalar::<i64>("PRAGMA page_size", []).unwrap(), 4096);
    assert_eq!(conn.scalar::<i64>("PRAGMA foreign_keys", []).unwrap(), 1);
    assert_eq!(
        conn.scalar::<i64>("PRAGMA cache_size", []).unwrap(),
        -archive.memory_budget().writer_cache_mib() * 1024
    );
    assert_eq!(
        conn.scalar::<i64>(
            "SELECT sqlite_compileoption_used('STMTJRNL_SPILL=131072')",
            []
        )
        .unwrap(),
        1
    );
    conn.execute_batch(
        "CREATE TABLE journal_parent(pk INTEGER PRIMARY KEY) STRICT;
         INSERT INTO journal_parent VALUES(1);
         CREATE TABLE journal_probe(pk INTEGER PRIMARY KEY,
             parent INTEGER NOT NULL REFERENCES journal_parent(pk),
             generation INTEGER NOT NULL CHECK(generation>=0), body BLOB NOT NULL) STRICT;
         WITH RECURSIVE ids(pk) AS (VALUES(1) UNION ALL SELECT pk+1 FROM ids WHERE pk<96)
         INSERT INTO journal_probe SELECT pk,1,0,zeroblob(3800) FROM ids;",
    )
    .unwrap();

    for (rows, spills) in [(18, false), (96, true)] {
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        // Each row occupies its own page. The first UPDATE writes those pages
        // into the outer transaction so the failing statement must save them.
        tx.execute("UPDATE journal_probe SET generation=1 WHERE pk<=?1", [rows])
            .unwrap();
        let mut update = tx.prepare_cached(
            "UPDATE journal_probe SET generation=CASE WHEN pk=?1 THEN -1 ELSE 2 END WHERE pk<=?2",
        ).unwrap();
        let before = writes();
        let error = update.execute([rows, rows]).unwrap_err();
        let requests = writes() - before;
        assert_eq!(
            error.sqlite_error_code(),
            Some(rusqlite::ErrorCode::ConstraintViolation)
        );
        assert_eq!(requests > 0, spills, "{rows} rows issued {requests} writes");
        assert_eq!(
            tx.scalar::<i64>("SELECT sum(generation) FROM journal_probe", [])
                .unwrap(),
            rows
        );
        assert!(!tx.is_autocommit());
        let error = tx
            .execute("INSERT INTO journal_probe VALUES(97,2,0,zeroblob(0))", [])
            .unwrap_err();
        assert_eq!(
            error.sqlite_error_code(),
            Some(rusqlite::ErrorCode::ConstraintViolation)
        );
        assert_eq!(update.execute([0, rows]).unwrap() as i64, rows);
        drop(update);
        assert_eq!(
            tx.scalar::<i64>("SELECT sum(generation) FROM journal_probe", [])
                .unwrap(),
            rows * 2
        );
        tx.rollback().unwrap();
        let reopened = crate::archive::open_read(&archive.index_db()).unwrap();
        assert_eq!(
            reopened
                .scalar::<i64>("SELECT sum(generation) FROM journal_probe", [])
                .unwrap(),
            0
        );
        assert_eq!(
            reopened
                .scalar::<i64>("SELECT count(*) FROM journal_probe", [])
                .unwrap(),
            96
        );
        assert_eq!(
            reopened
                .scalar::<i64>(
                    "SELECT count(*) FROM pragma_foreign_key_check('journal_probe')",
                    []
                )
                .unwrap(),
            0
        );
    }
}
