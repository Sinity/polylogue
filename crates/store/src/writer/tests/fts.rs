//! Deferred search retries retain parsed progress and the live indexer's ranking.
use super::*;
use crate::archive::{cache, open_write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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
        tx.execute(
            "INSERT INTO text(id, hash, body, indexed) VALUES(?1,?2,?3,?4)",
            params![
                1000 + n as i64,
                blake3::hash(text.as_bytes()).as_bytes().as_slice(),
                crate::encoding::pack_text(&text).unwrap(),
                membership(n)
            ],
        )
        .unwrap();
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
        "SELECT * FROM message_projection ORDER BY version_pk",
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
        let count: i64 = conn
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
            .unwrap();
        out.insert(
            table.into(),
            serde_json::json!({"rows":count, "hits_bm25":hits,
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
    conn.execute(
        "UPDATE text SET body=X'00' WHERE id=?1",
        [1000 + ROWS as i64 - 2],
    )
    .unwrap();
    let failed = crate::text_rows::index_marked(&conn).is_err();
    let partial: i64 = conn
        .query_row("SELECT count(*) FROM text_fts", [], |r| r.get(0))
        .unwrap();
    let preserved = before == parsed(&conn) && crate::schema::building(&conn).unwrap();
    conn.execute(
        "UPDATE text SET body=?1 WHERE id=?2",
        params![
            crate::encoding::pack_text(&body(ROWS - 2)).unwrap(),
            1000 + ROWS as i64 - 2
        ],
    )
    .unwrap();
    drop(conn);
    crate::fault::arm(
        "index.deferred",
        &w.archive.root.to_string_lossy(),
        crate::fault::Action::Fail,
        0,
        1,
    );
    let (w, first_resume) = reopen(w);
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let still_not_ready = crate::schema::building(&index).unwrap();
    drop(index);
    // This retry starts with a fully committed FTS build but incomplete
    // secondary indexes, exercising its statistics as well as partial rows.
    let (w, resumed) = reopen(w);
    crate::fault::clear();
    let conn = open_write(&w.archive.index_db(), cache::INDEX_WRITER).unwrap();
    let ready = !crate::schema::building(&conn).unwrap();
    let recovered = signature(&conn);
    let preserved = preserved && before == parsed(&conn);
    drop(conn);
    drop(w);
    serde_json::json!({"failed":failed,"partial_fts_rows":partial,"parsed_preserved":preserved,
        "deferred_failure":first_resume.is_err(),"not_ready_after_deferred_failure":still_not_ready,
        "reopen_ok":resumed.is_ok(),"ordinary_reopen":resumed,"ready":ready,"normal_pending_exact":recovered==pending_signature(ROWS)})
}

fn stopped() -> serde_json::Value {
    crate::fault::clear();
    let w = corpus(TEXTS);
    let path = w.archive.index_db();
    let conn = open_write(&path, cache::INDEX_WRITER).unwrap();
    let before = parsed(&conn);
    drop(conn);
    let stop = WriterStop::default();
    let stop_seen = stop.clone();
    let commits = Arc::new(AtomicU64::new(0));
    let postings = Arc::new(AtomicU64::new(0));
    let committed = commits.clone();
    let posted = postings.clone();
    let file = path.clone();
    let run = std::thread::spawn(move || {
        // Force natural FTS flushing and WAL spill with a finite synthetic set.
        // This connection's page cache is smaller; FTS settings are unchanged.
        let conn = open_write(&file, 8).unwrap();
        // SQLite owns the resolved filename used by the production fault point.
        crate::fault::arm(
            "search.indexed",
            conn.path().unwrap(),
            crate::fault::Action::Hold,
            24_000,
            1,
        );
        conn.commit_hook(Some(move || {
            committed.fetch_add(1, Ordering::Relaxed);
            false
        }))
        .unwrap();
        conn.update_hook(Some(
            move |_: rusqlite::hooks::Action, _: &str, table: &str, id: i64| {
                if (table == "text_fts_data" || table == "output_fts_data") && id > 10 {
                    posted.fetch_add(1, Ordering::Relaxed);
                }
            },
        ))
        .unwrap();
        conn.progress_handler(1000, Some(move || stop_seen.requested()))
            .unwrap();
        let result = crate::text_rows::index_marked(&conn);
        conn.progress_handler(0, None::<fn() -> bool>).unwrap();
        result.map_err(|e| e.to_string())
    });
    let guard = Instant::now();
    while !crate::fault::held() {
        if run.is_finished() {
            let result = run.join().unwrap();
            panic!("deferred search ended before its stop boundary: {result:?}");
        }
        if guard.elapsed() >= std::time::Duration::from_secs(30) {
            stop.request();
            crate::fault::release();
            let result = run.join().unwrap();
            panic!(
                "deferred search did not reach its stop boundary: result={result:?}, commits={}, postings={}, wal_bytes={}",
                commits.load(Ordering::Relaxed),
                postings.load(Ordering::Relaxed),
                wal_bytes(&path)
            );
        }
        std::thread::yield_now();
    }
    let wal_at_stop = wal_bytes(&path);
    let commits_at_stop = commits.load(Ordering::Relaxed);
    let postings_at_stop = postings.load(Ordering::Relaxed);
    let observer = crate::archive::open_read(&path).unwrap();
    let not_ready_at_stop = crate::schema::building(&observer).unwrap();
    drop(observer);
    let started = Instant::now();
    stop.request();
    crate::fault::release();
    let result = run.join().unwrap();
    let stop_seconds = started.elapsed().as_secs_f64();
    crate::fault::clear();
    let conn = open_write(&path, cache::INDEX_WRITER).unwrap();
    let partial: i64 = conn
        .query_row("SELECT count(*) FROM text_fts", [], |r| r.get(0))
        .unwrap();
    let preserved = before == parsed(&conn) && crate::schema::building(&conn).unwrap();
    drop(conn);
    let (w, resumed) = reopen(w);
    let conn = open_write(&w.archive.index_db(), cache::INDEX_WRITER).unwrap();
    let ready = !crate::schema::building(&conn).unwrap();
    let preserved = preserved && before == parsed(&conn);
    let recovered = signature(&conn);
    drop(conn);
    drop(w);
    serde_json::json!({"interrupted":result.is_err(),"wal_at_stop":wal_at_stop,"stop_seconds":stop_seconds,
        "commits_at_stop":commits_at_stop,"postings_at_stop":postings_at_stop,"not_ready_at_stop":not_ready_at_stop,
        "partial_fts_rows":partial,"parsed_preserved":preserved,
        "reopen_ok":resumed.is_ok(),"ordinary_reopen":resumed,"ready":ready,"normal_pending_exact":recovered==pending_signature(TEXTS)})
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
    assert!(stopped["wal_at_stop"].as_u64().unwrap() > 32);
    assert!(stopped["postings_at_stop"].as_u64().unwrap() > 0);
    assert_eq!(stopped["not_ready_at_stop"], true);
    assert_eq!(stopped["parsed_preserved"], true);
    assert_eq!(stopped["ready"], true);
    assert_eq!(stopped["reopen_ok"], true);
    assert_eq!(stopped["normal_pending_exact"], true);
    assert_eq!(corruption["failed"], true);
    assert_eq!(corruption["parsed_preserved"], true);
    assert_eq!(corruption["ready"], true);
    assert_eq!(corruption["reopen_ok"], true);
    assert_eq!(corruption["deferred_failure"], true);
    assert_eq!(corruption["not_ready_after_deferred_failure"], true);
    assert_eq!(corruption["normal_pending_exact"], true);
}
