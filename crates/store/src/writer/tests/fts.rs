//! Deferred search retries keep parsed progress and the live indexer's ranking.
use super::*;
use crate::archive::{cache, open_write};
use std::path::Path;
use std::time::Instant;

const TEXTS: usize = 32_000;
const WORDS: usize = 64;

fn body(n: usize) -> String {
    use std::fmt::Write;
    let mut text = String::from("anchor cedar birch ");
    if n.is_multiple_of(5) {
        text.push_str("Łódź Straße ");
    }
    for word in 0..WORDS {
        let mixed = (n as u64 * 67 + word as u64).wrapping_mul(0x9e3779b97f4a7c15);
        write!(&mut text, "t{n:05x}{word:02x}{mixed:016x} ").unwrap();
    }
    text
}

fn membership(n: usize) -> i64 {
    match n % 7 {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => 3,
    }
}

fn corpus(texts: usize) -> TestArchive {
    let mut w = test_archive();
    w.store_reading(
        batch(
            "fts-progress",
            Route::Live,
            0,
            vec![row(
                0,
                text_message(
                    "fts-progress-message",
                    "fts-progress-session",
                    Role::User,
                    "preserved source progress",
                    None,
                ),
            )],
            &["fts-progress-session"],
        ),
        Some(&SOURCE_RECORD),
    )
    .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    w.writer.stop();
    let mut conn = open_write(&w.archive.index_db(), cache::INDEX_WRITER).unwrap();
    let tx = conn.transaction().unwrap();
    for n in 0..texts {
        let text = body(n);
        crate::text_rows::insert_alone(
            &tx,
            Some(1000 + n as i64),
            blake3::hash(text.as_bytes()).as_bytes(),
            &crate::encoding::normalized_text_hash(&text),
            &text,
            membership(n),
        );
    }
    tx.commit().unwrap();
    checkpoint(&conn);
    w
}

fn checkpoint(conn: &Connection) {
    let status: (i64, i64, i64) = conn
        .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!(status.0, 0, "fixture has no held read snapshot");
}

fn wal_bytes(path: &Path) -> u64 {
    std::fs::metadata(format!("{}-wal", path.display())).map_or(0, |m| m.len())
}

fn digest(conn: &Connection, sql: &str) -> String {
    use rusqlite::types::ValueRef;
    let mut h = blake3::Hasher::new();
    let mut query = conn.prepare(sql).unwrap();
    let columns = query.column_count();
    let mut rows = query.query([]).unwrap();
    while let Some(row) = rows.next().unwrap() {
        for column in 0..columns {
            match row.get_ref(column).unwrap() {
                ValueRef::Null => {
                    h.update(b"n");
                }
                ValueRef::Integer(n) => {
                    h.update(b"i");
                    h.update(&n.to_le_bytes());
                }
                ValueRef::Real(n) => {
                    h.update(b"r");
                    h.update(&n.to_bits().to_le_bytes());
                }
                ValueRef::Text(s) | ValueRef::Blob(s) => {
                    h.update(
                        if matches!(row.get_ref(column).unwrap(), ValueRef::Text(_)) {
                            b"t"
                        } else {
                            b"b"
                        },
                    );
                    h.update(&(s.len() as u64).to_le_bytes());
                    h.update(s);
                }
            }
        }
    }
    h.finalize().to_hex().to_string()
}

fn parsed(conn: &Connection) -> Vec<String> {
    [
        "SELECT * FROM parse_state ORDER BY input_pk,chain",
        "SELECT * FROM message ORDER BY pk",
        "SELECT * FROM message_source ORDER BY message_pk,input_pk,chain",
        "SELECT * FROM block ORDER BY pk",
        "SELECT * FROM message_version ORDER BY pk",
        "SELECT * FROM message_version_body ORDER BY version_pk",
        "SELECT * FROM message_block_selection ORDER BY version_pk",
        "SELECT key,value FROM meta WHERE key LIKE '%generation%' ORDER BY key",
    ]
    .map(|sql| digest(conn, sql))
    .into()
}

fn signature(conn: &Connection) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (table, vocab) in [
        ("text_fts", "fts_vocab_text"),
        ("output_fts", "fts_vocab_output"),
    ] {
        let mut hits = Vec::new();
        for term in ["anchor", "\"cedar birch\"", "lodz strasse", "t00001*"] {
            let sql = format!(
                "SELECT rowid,bm25({table}) FROM {table} WHERE {table} MATCH '{}' ORDER BY rowid",
                term.replace('\'', "''")
            );
            hits.push((term, digest(conn, &sql)));
        }
        let cutoff = digest(
            conn,
            &format!(
                "SELECT rowid,rank FROM {table} WHERE {table} MATCH 'anchor' ORDER BY rank LIMIT 31"
            ),
        );
        let count: i64 = conn
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
            .unwrap();
        out.insert(
            table.into(),
            serde_json::json!({"rows":count, "hits_bm25":hits, "rank_cutoff":cutoff,
            "vocabulary":digest(conn, &format!("SELECT term,doc,cnt FROM {vocab} ORDER BY term")),
            "docsize":digest(conn, &format!("SELECT id,sz FROM {table}_docsize ORDER BY id")),
            "settings":digest(conn, &format!("SELECT k,v FROM {table}_config ORDER BY k"))}),
        );
    }
    serde_json::Value::Object(out)
}

fn pending_signature(texts: usize) -> serde_json::Value {
    let normal = corpus(texts);
    let conn = open_write(&normal.archive.index_db(), cache::INDEX_WRITER).unwrap();
    conn.execute_batch("INSERT INTO search_pending(text_id,fts_table) SELECT id,1 FROM text WHERE indexed&1<>0; INSERT INTO search_pending(text_id,fts_table) SELECT id,2 FROM text WHERE indexed&2<>0;").unwrap();
    crate::text_rows::index_pending(&conn, || ()).unwrap();
    signature(&conn)
}

fn reopen(w: TestArchive) -> (TestArchive, std::result::Result<u64, String>) {
    let root = w.archive.root.clone();
    let TestArchive {
        writer,
        archive,
        _lock,
        _dir,
    } = w;
    drop(writer);
    drop(archive);
    drop(_lock);
    let (archive, lock) = Archive::open_or_create(&root).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let result = writer.derive().map_err(|e| e.to_string());
    (
        TestArchive {
            writer,
            archive,
            _lock: lock,
            _dir,
        },
        result,
    )
}

fn corruption() -> serde_json::Value {
    const ROWS: usize = 6000;
    let w = corpus(ROWS);
    let conn = open_write(&w.archive.index_db(), cache::INDEX_WRITER).unwrap();
    let before = parsed(&conn);
    crate::text_rows::replace_group_body(&conn, 1000 + ROWS as i64 - 2, b"\0");
    let failed = crate::text_rows::SearchBuild::start(
        &conn,
        &crate::cancellation::Cancellation::default(),
        8,
    )
    .unwrap()
    .finish(&conn)
    .is_err();
    let partial: i64 = conn
        .query_row("SELECT count(*) FROM text_fts", [], |r| r.get(0))
        .unwrap();
    let preserved = before == parsed(&conn) && w.archive.index_build().is_some();
    crate::text_rows::replace_group_body(
        &conn,
        1000 + ROWS as i64 - 2,
        &crate::text_rows::alone(&body(ROWS - 2)),
    );
    drop(conn);
    crate::fault::arm(
        "index.deferred",
        &w.archive.root.to_string_lossy(),
        crate::fault::Action::Fail,
        0,
        1,
    );
    let (w, first_resume) = reopen(w);
    let still_not_ready = w.archive.index_build().is_some();
    // This retry starts with a fully committed FTS build but incomplete
    // secondary indexes, exercising its statistics as well as partial rows.
    let (w, resumed) = reopen(w);
    crate::fault::clear();
    let conn = open_write(&w.archive.index_db(), cache::INDEX_WRITER).unwrap();
    let ready = w.archive.index_build().is_none();
    let recovered = signature(&conn);
    let preserved = preserved && before == parsed(&conn);
    drop(conn);
    drop(w);
    serde_json::json!({"failed":failed,"partial_fts_rows":partial,"parsed_preserved":preserved,
        "deferred_failure":first_resume.is_err(),"not_ready_after_deferred_failure":still_not_ready,
        "reopen_ok":resumed.is_ok(),"reopen_result":resumed,"ready":ready,"normal_pending_exact":recovered==pending_signature(ROWS)})
}

fn stopped() -> serde_json::Value {
    crate::fault::clear();
    let w = corpus(TEXTS);
    let path = w.archive.index_db();
    let conn = open_write(&path, cache::INDEX_WRITER).unwrap();
    let before = parsed(&conn);
    drop(conn);
    let stop = crate::cancellation::Cancellation::default();
    let stop_seen = stop.clone();
    let file = path.clone();
    let run = std::thread::spawn(move || {
        // A small page cache, so the one search transaction spills pages
        // before the stop; FTS settings are unchanged.
        let conn = open_write(&file, 8).unwrap();
        // SQLite owns the resolved filename used by the production fault point.
        crate::fault::arm(
            "search.before_commit",
            conn.path().unwrap(),
            crate::fault::Action::Hold,
            0,
            1,
        );
        let result = crate::text_rows::SearchBuild::start(&conn, &stop_seen, 8)
            .and_then(|build| build.finish(&conn));
        result.map_err(|e| e.to_string())
    });
    let guard = Instant::now();
    while !crate::fault::held() {
        if run.is_finished() {
            let result = run.join().unwrap();
            panic!("deferred search ended before its stop boundary: {result:?}");
        }
        if guard.elapsed() >= std::time::Duration::from_secs(30) {
            stop.cancel();
            crate::fault::release();
            let result = run.join().unwrap();
            panic!(
                "search copy did not reach its commit boundary: result={result:?}, wal_bytes={}",
                wal_bytes(&path)
            );
        }
        std::thread::yield_now();
    }
    let wal_at_stop = wal_bytes(&path);
    let reader = crate::archive::open_read(&path).unwrap();
    let visible_at_stop: i64 = reader
        .query_row("SELECT count(*) FROM text_fts", [], |r| r.get(0))
        .unwrap();
    drop(reader);
    let not_ready_at_stop = w.archive.index_build().is_some();
    let started = Instant::now();
    stop.cancel();
    crate::fault::release();
    let result = run.join().unwrap();
    let stop_seconds = started.elapsed().as_secs_f64();
    crate::fault::clear();
    let conn = open_write(&path, cache::INDEX_WRITER).unwrap();
    let partial: i64 = conn
        .query_row("SELECT count(*) FROM text_fts", [], |r| r.get(0))
        .unwrap();
    let preserved = before == parsed(&conn) && w.archive.index_build().is_some();
    drop(conn);
    let (w, resumed) = reopen(w);
    let conn = open_write(&w.archive.index_db(), cache::INDEX_WRITER).unwrap();
    let ready = w.archive.index_build().is_none();
    let preserved = preserved && before == parsed(&conn);
    let recovered = signature(&conn);
    drop(conn);
    drop(w);
    serde_json::json!({"interrupted":result.is_err(),"wal_at_stop":wal_at_stop,"stop_seconds":stop_seconds,
        "visible_at_stop":visible_at_stop,"reached_copy_commit_boundary":true,"not_ready_at_stop":not_ready_at_stop,
        "partial_fts_rows":partial,"parsed_preserved":preserved,
        "reopen_ok":resumed.is_ok(),"reopen_result":resumed,"ready":ready,"normal_pending_exact":recovered==pending_signature(TEXTS)})
}

#[test]
fn deferred_search_replay_matches_live_ranking_after_interruption() {
    let stopped = stopped();
    let corruption = corruption();
    eprintln!(
        "FTS_CONTROLS {}",
        serde_json::json!({"stopped":stopped,"corruption":corruption})
    );
    assert_eq!(stopped["interrupted"], true);
    // Both native trees have been copied but their one main-index transaction
    // has not committed (docs/architecture.md, Pipeline). Readers see neither.
    assert_eq!(stopped["reached_copy_commit_boundary"], true);
    assert_eq!(stopped["visible_at_stop"], 0);
    assert_eq!(stopped["partial_fts_rows"], 0);
    assert_eq!(stopped["not_ready_at_stop"], true);
    assert_eq!(stopped["parsed_preserved"], true);
    assert_eq!(stopped["ready"], true);
    assert_eq!(stopped["reopen_ok"], true);
    assert_eq!(stopped["normal_pending_exact"], true);
    assert_eq!(corruption["failed"], true);
    assert_eq!(corruption["partial_fts_rows"], 0);
    assert_eq!(corruption["parsed_preserved"], true);
    assert_eq!(corruption["ready"], true);
    assert_eq!(corruption["reopen_ok"], true);
    assert_eq!(corruption["deferred_failure"], true);
    assert_eq!(corruption["not_ready_after_deferred_failure"], true);
    assert_eq!(corruption["normal_pending_exact"], true);
}

#[test]
fn streamed_search_keeps_committed_membership_rank_positions_and_reopen() {
    use crate::text_rows::{SearchBuild, insert_alone};
    let dir = tempfile::tempdir().unwrap();
    let mut built = open_write(&dir.path().join("built.db"), 8).unwrap();
    let normal = open_write(&dir.path().join("normal.db"), 8).unwrap();
    for conn in [&built, &normal] {
        crate::schema::create_index(conn).unwrap();
    }
    let insert = |conn: &Connection, id: i64, bits: i64, text: &str| {
        insert_alone(
            conn,
            Some(id),
            blake3::hash(text.as_bytes()).as_bytes(),
            &crate::encoding::normalized_text_hash(text),
            text,
            bits,
        );
    };
    let stop = crate::cancellation::Cancellation::default();
    let mut build = SearchBuild::start(&built, &stop, 8).unwrap();
    build.capture(&built, None).unwrap();
    let before = signature(&built); // Cache both native virtual-table structures.
    for chunk in 0..4 {
        let tx = built.transaction().unwrap();
        for n in chunk * 64..(chunk + 1) * 64 {
            insert(&tx, 1000 + n as i64, membership(n), &body(n));
        }
        assert!(
            build.capture(&tx, None).is_err(),
            "uncommitted rows cannot enter the feed"
        );
        tx.commit().unwrap();
        build.capture(&built, None).unwrap();
    }
    assert_eq!(
        signature(&built),
        before,
        "temporary postings are not published"
    );
    let texts = crate::text_rows::TextIds::new(&built, true, 1 << 20).unwrap();
    let text_tx = texts.transaction();
    let tx = built.transaction().unwrap();
    insert(&tx, 900, 3, "anchor cedar birch Łódź Straße");
    crate::text_rows::remove_unindexed(&tx, &texts, 1004).unwrap();
    tx.execute("UPDATE text SET indexed=2 WHERE id=1001", [])
        .unwrap();
    tx.execute("UPDATE text SET indexed=3 WHERE id=1002", [])
        .unwrap();
    tx.commit().unwrap();
    build.capture(&built, text_tx.commit()).unwrap();
    build.capture(&built, None).unwrap();
    let text_tx = texts.transaction();
    let tx = built.transaction().unwrap();
    crate::text_rows::remove_unindexed(&tx, &texts, 1003).unwrap();
    insert(&tx, 9999, 3, "must never become searchable");
    tx.rollback().unwrap();
    drop(text_tx);
    build.capture(&built, None).unwrap();
    let oversized = format!("{} anchor cedar", "x".repeat((4 << 20) + 1));
    insert(&built, 20000, 3, &oversized);
    build.capture(&built, None).unwrap();
    build.finish(&built).unwrap();

    // The independent ordinary indexer sees only the hand-specified final
    // population, not the builder's notices or reconstructed native trees.
    for n in 0..256 {
        if n == 4 {
            continue;
        }
        let bits = match n {
            1 => 2,
            2 => 3,
            _ => membership(n),
        };
        insert(&normal, 1000 + n as i64, bits, &body(n));
    }
    insert(&normal, 900, 3, "anchor cedar birch Łódź Straße");
    insert(&normal, 20000, 3, &oversized);
    normal
        .execute_batch(
            "INSERT INTO search_pending SELECT id,1 FROM text WHERE indexed&1<>0;
        INSERT INTO search_pending SELECT id,2 FROM text WHERE indexed&2<>0;",
        )
        .unwrap();
    crate::text_rows::index_pending(&normal, || ()).unwrap();
    let reopened = crate::archive::open_read(&dir.path().join("built.db")).unwrap();
    let positions = |conn: &Connection, table: &str| {
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS temp.positions_{table}
            USING fts5vocab(main,{table},instance)"
        ))
        .unwrap();
        digest(
            conn,
            &format!(
                "SELECT term,doc,col,offset FROM temp.positions_{table}
            WHERE term IN ('anchor','cedar','lodz','strasse') ORDER BY term,doc,col,offset"
            ),
        )
    };
    for conn in [&built, &reopened] {
        assert_eq!(signature(conn), signature(&normal));
        for table in ["text_fts", "output_fts"] {
            assert_eq!(positions(conn, table), positions(&normal, table));
        }
    }

    // Installation must invalidate the original writer's cached FTS state,
    // including before reopen, and leave the ordinary incremental route usable.
    for conn in [&built, &normal] {
        insert(conn, 30000, 2, "anchor cedar birch maple");
        conn.execute("INSERT INTO search_pending VALUES(30000,2)", [])
            .unwrap();
        crate::text_rows::index_pending(conn, || ()).unwrap();
        crate::text_rows::remove(conn, 1003, membership(3)).unwrap();
    }
    for conn in [&built, &reopened] {
        assert_eq!(signature(conn), signature(&normal));
    }
    for conn in [&built, &normal] {
        insert(conn, 1003, membership(3), &body(3));
        for bit in [1, 2].into_iter().filter(|bit| membership(3) & bit != 0) {
            conn.execute("INSERT INTO search_pending VALUES(?1,?2)", [1003, bit])
                .unwrap();
        }
        crate::text_rows::index_pending(conn, || ()).unwrap();
    }
    for conn in [&built, &reopened] {
        assert_eq!(signature(conn), signature(&normal));
    }
}

#[test]
fn a_search_worker_can_build_before_finalization_without_holding_the_row_writer() {
    struct Release;
    impl Drop for Release {
        fn drop(&mut self) {
            crate::fault::release();
        }
    }
    let mut w = test_archive();
    let path = w.archive.index_db();
    crate::fault::arm(
        "search.build_batch",
        &path.to_string_lossy(),
        crate::fault::Action::Hold,
        0,
        1,
    );
    let release = Release;
    let reading = |name: &str| {
        batch(
            name,
            Route::Live,
            0,
            vec![row(
                0,
                text_message(name, "streaming-session", Role::User, name, None),
            )],
            &["streaming-session"],
        )
    };
    w.store_reading(reading("cedar"), Some(&SOURCE_RECORD))
        .unwrap();
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    while !crate::fault::held() && Instant::now() < deadline {
        std::thread::yield_now();
    }
    let held = crate::fault::held();
    if !held {
        drop(release);
        panic!("the first committed group never reached native search construction");
    }
    // The first native batch is still held. Another group can commit using
    // the real queue; the worker owns no private-index reader or writer lease.
    let second = w.store_reading(reading("maple"), Some(&SOURCE_RECORD));
    let still_building = w.archive.index_build().is_some();
    drop(release);
    assert!(second.is_ok(), "{second:?}");
    assert!(still_building);
    assert!(w.writer.flush().unwrap().errors.is_empty());
    w.writer.stop();
    w.archive = w.archive.reopened();
    w.writer = Writer::start(&w.archive).unwrap();
    w.writer.derive().unwrap();
    crate::fault::clear();
    assert!(w.archive.index_build().is_none());
    let conn = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let texts: i64 = conn
        .query_row(
            "SELECT count(*) FROM text_fts WHERE text_fts MATCH 'cedar OR maple'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(texts, 2);
}

#[test]
fn an_early_search_failure_is_reported_before_an_explicit_build_retry() {
    let w = test_archive();
    crate::fault::arm(
        "search.build_batch",
        &w.archive.index_db().to_string_lossy(),
        crate::fault::Action::Fail,
        0,
        1,
    );
    for name in ["cedar", "maple", "birch", "walnut"] {
        w.store_reading(
            batch(
                name,
                Route::Live,
                0,
                vec![row(
                    0,
                    text_message(name, "failed-search", Role::User, name, None),
                )],
                &["failed-search"],
            ),
            Some(&SOURCE_RECORD),
        )
        .unwrap();
    }
    let report = w.writer.flush().unwrap();
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.contains("search.build_batch")),
        "{:?}",
        report.errors
    );
    // Ordinary row turns must not silently rebuild a failed worker's prefix.
    // The finalizer owns reporting its exact result before a new build retries.
    let error = w.writer.derive().unwrap_err();
    assert!(error.to_string().contains("search.build_batch"), "{error}");
    assert!(w.archive.index_build().is_some());
    w.writer.derive().unwrap();
    crate::fault::clear();
    assert!(w.archive.index_build().is_none());
    let conn = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let texts: i64 = conn.query_row(
        "SELECT count(*) FROM text_fts WHERE text_fts MATCH 'cedar OR maple OR birch OR walnut'",
        [], |row| row.get(0),
    ).unwrap();
    assert_eq!(texts, 4);
}

#[test]
fn a_staged_chain_replacement_keeps_final_population_search_during_a_build() {
    let replaced = test_archive();
    let fresh = test_archive();
    let reading = |body: &str| {
        batch(
            "fts-chain",
            Route::Live,
            0,
            vec![row(
                0,
                text_message("selected", "fts-session", Role::User, body, None),
            )],
            &["fts-session"],
        )
    };
    replaced
        .store_reading(reading("cedar cedar birch"), Some(&SOURCE_RECORD))
        .unwrap();
    assert!(replaced.writer.flush().unwrap().errors.is_empty());
    assert!(replaced.archive.index_build().is_some());
    let mut next = reading("maple maple birch");
    let mut staged = ChainReading::new().unwrap();
    staged.push(&mut next.parse[0]).unwrap();
    replaced
        .writer
        .replace_chain(next.input, 1, next.length, staged)
        .unwrap();
    replaced.writer.derive().unwrap();
    fresh
        .store_reading(reading("maple maple birch"), Some(&SOURCE_RECORD))
        .unwrap();
    fresh.writer.derive().unwrap();
    let left = crate::archive::open_read(&replaced.archive.index_db()).unwrap();
    let right = crate::archive::open_read(&fresh.archive.index_db()).unwrap();
    for (table, vocab) in [
        ("text_fts", "fts_vocab_text"),
        ("output_fts", "fts_vocab_output"),
    ] {
        let ranked = format!(
            "SELECT hex(t.hash),bm25({table}) FROM {table} JOIN text t ON t.id={table}.rowid WHERE {table} MATCH 'cedar OR maple' ORDER BY hex(t.hash)"
        );
        assert_eq!(digest(&left, &ranked), digest(&right, &ranked));
        let words = format!("SELECT term,doc,cnt FROM {vocab} ORDER BY term");
        assert_eq!(digest(&left, &words), digest(&right, &words));
    }
    assert!(replaced.archive.index_build().is_none());
}
