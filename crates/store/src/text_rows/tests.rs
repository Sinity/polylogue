use super::*;
use crate::archive::{cache, open_write};
use crate::fixture::Fixture;

fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |row| row.get(0)).unwrap()
}

fn index_of(table: &str) -> crate::search::SnippetIndex {
    match table {
        "text_fts" => crate::search::SnippetIndex::Prose,
        _ => crate::search::SnippetIndex::Output,
    }
}

/// The entries of `table` over every full-text file.
fn entries(conn: &Connection, table: &str) -> i64 {
    matching(conn, table, None).len() as i64
}

/// The keys of the entries in `table` matching `expression` (all entries
/// when `None`), over every full-text file, in key order.
fn matching(conn: &Connection, table: &str, expression: Option<&str>) -> Vec<i64> {
    let sql = match expression {
        Some(expression) => crate::search::matching_texts(
            index_of(table),
            &format!("'{}'", expression.replace('\'', "''")),
        ),
        None => (0..crate::search::files::FILES)
            .map(|n| format!("SELECT rowid AS text_id FROM search_{n}.{table} WHERE rowid > 0"))
            .collect::<Vec<_>>()
            .join(" UNION ALL "),
    };
    conn.prepare(&format!("SELECT text_id FROM ({sql}) ORDER BY text_id"))
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

/// Fills the full-text files of the index `conn` has open from its texts'
/// membership, as a build's settling does when its composition ends.
/// The files are attached afterwards, as the installed index's are.
fn fill(conn: &Connection) -> Result<()> {
    let filled = (|| {
        let mut fill = SearchFill::start(conn, 64 << 20)?;
        fill.settle(conn)?;
        while !fill.wait(std::time::Duration::from_secs(1))? {}
        Ok(())
    })();
    crate::search::files::attach(conn).unwrap();
    filled
}

/// A connection to a build's file, whose full-text files a fill writes, or
/// to an installed index, whose files upkeep writes.
fn index(dir: &std::path::Path, built: bool) -> Connection {
    if built {
        let conn = open_write(&dir.join("build.db"), cache::SETUP).unwrap();
        crate::schema::ensure_build_index(&conn).unwrap();
        conn
    } else {
        let fixture = Fixture::new(&dir.join("archive")).unwrap();
        open_write(&fixture.archive.index_db(), cache::SETUP).unwrap()
    }
}

/// Stores `body` as a member of the search index `bit` names, as a block
/// holding it makes it: queued for its entry in an installed index, only
/// marked in a build, which writes the entries from the marks.
fn text(conn: &Connection, body: &str, bit: i64, queued: bool) -> i64 {
    let id = intern(conn, body).unwrap();
    if bit != 0 {
        if queued {
            mark_searched(conn, id, bit).unwrap();
        } else {
            conn.execute(
                "UPDATE text SET indexed = indexed | ?2 WHERE id = ?1",
                params![id, bit],
            )
            .unwrap();
        }
    }
    id
}

#[test]
fn repeated_text_is_stored_once_and_indexed_in_both_searches() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let tx = conn.transaction().unwrap();
    let id = text(&tx, "walrus", 1, true);
    assert_eq!(text(&tx, "walrus", 1, true), id);
    assert_eq!(text(&tx, "walrus", 2, true), id);
    tx.commit().unwrap();
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 2);
    assert_eq!(index_pending(&conn).unwrap(), 2);
    assert_eq!(matching(&conn, "text_fts", Some("walrus")).len() as i64, 1);
    assert_eq!(
        matching(&conn, "output_fts", Some("walrus")).len() as i64,
        1
    );
    assert_eq!(text(&conn, "walrus", 1, true), id);
    assert_eq!(index_pending(&conn).unwrap(), 0);
}

#[test]
fn removal_and_rollback_do_not_leave_search_work() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    {
        let tx = conn.transaction().unwrap();
        text(&tx, "rolled back", 1, true);
    }
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 0);
    assert_eq!(index_pending(&conn).unwrap(), 0);
    let tx = conn.transaction().unwrap();
    let id = text(&tx, "removed", 1, true);
    remove(&tx, id).unwrap();
    tx.commit().unwrap();
    // Its queue row settles to no entry.
    assert_eq!(index_pending(&conn).unwrap(), 1);
    assert_eq!(entries(&conn, "text_fts"), 0);
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 0);
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 0);
}

#[test]
fn a_failed_decode_keeps_the_pending_entry() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let id = text(&conn, "walrus", 1, true);
    replace_group_body(&conn, id, b"damaged");
    assert!(index_pending(&conn).is_err());
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 1);
    assert_eq!(entries(&conn, "text_fts"), 0);
}

#[test]
fn pending_byte_limit_commits_complete_entries_and_retry_keeps_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let first = text(&conn, "walrus", 1, true);
    let second = text(&conn, "narwhal", 2, true);
    text(&conn, "dolphin", 1, true);
    replace_group_body(&conn, second, b"\0");
    assert_eq!(pending_batch(&conn, 1).unwrap(), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 2);
    assert!(index_pending(&conn).is_err());
    assert_eq!(entries(&conn, "text_fts"), 1);
    assert_eq!(entries(&conn, "output_fts"), 0);
    assert_eq!(matching(&conn, "text_fts", None)[0], first);
    replace_group_body(&conn, second, &alone("narwhal"));
    assert_eq!(index_pending(&conn).unwrap(), 2);
    assert_eq!(index_pending(&conn).unwrap(), 0);
    assert_eq!(entries(&conn, "text_fts"), 2);
    assert_eq!(
        matching(&conn, "output_fts", Some("narwhal")).len() as i64,
        1
    );
}

#[test]
fn a_later_decode_failure_rolls_back_earlier_search_entries_in_its_transaction() {
    for built in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let conn = index(dir.path(), built);
        text(&conn, "walrus", 1, !built);
        let second = text(&conn, "narwhal", 1, !built);
        replace_group_body(&conn, second, b"\0");
        if built {
            assert!(fill(&conn).is_err());
        } else {
            assert!(index_pending(&conn).is_err());
            assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 2);
        }
        assert_eq!(entries(&conn, "text_fts"), 0);
    }
}

#[test]
fn build_and_pending_searches_keep_ascii_unicode_and_search_kinds() {
    for built in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut conn = index(dir.path(), built);
        let tx = conn.transaction().unwrap();
        for n in 0..SEARCH_ROWS + 3 {
            text(&tx, &format!("walrus{n}"), 1, !built);
        }
        let shared = "Łódź Straße Øresund";
        let id = text(&tx, shared, 1, !built);
        assert_eq!(text(&tx, shared, 2, !built), id);
        text(&tx, "notindexed", 0, !built);
        tx.commit().unwrap();
        if built {
            fill(&conn).unwrap();
        } else {
            assert_eq!(index_pending(&conn).unwrap(), (SEARCH_ROWS + 5) as u64);
        }
        assert_eq!(entries(&conn, "text_fts"), SEARCH_ROWS + 4);
        assert_eq!(entries(&conn, "output_fts"), 1);
        assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 0);
        for table in ["text_fts", "output_fts"] {
            assert_eq!(matching(&conn, table, Some("lodz strasse oresund")), [id]);
            assert!(matching(&conn, table, Some("notindexed")).is_empty());
        }
        assert_eq!(
            matching(
                &conn,
                "text_fts",
                Some(format!("walrus{}", SEARCH_ROWS + 2).as_str())
            )
            .len(),
            1
        );
    }
}

#[test]
fn a_text_is_one_row_and_a_removed_texts_key_is_never_given_again() {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    let first = intern(&conn, "shared synthetic text").unwrap();
    assert_eq!(intern(&conn, "shared synthetic text").unwrap(), first);
    let prepared = PreparedTexts::pack(["shared synthetic text"]).unwrap();
    let hash = *blake3::hash(b"shared synthetic text").as_bytes();
    assert_eq!(resolve(&conn, &hash, None).unwrap(), first);
    assert_eq!(resolve(&conn, &hash, prepared.get(&hash)).unwrap(), first);
    let unknown = *blake3::hash(b"never stored").as_bytes();
    assert!(resolve(&conn, &unknown, None).is_err());
    // The newest key leaves; the next text gets a new one.
    remove(&conn, first).unwrap();
    let second = intern(&conn, "another synthetic text").unwrap();
    assert!(second > first, "{second} reused {first}");
}

/// A block, document block or tool call holding `text`, written as
/// composition writes it.
fn block(conn: &Connection, pk: i64, kind: &str, text: i64) {
    conn.execute(
        "INSERT INTO block(pk,message_pk,session_pk,position,kind,text_id,content_group,content_start,content_length)
         VALUES (?1,1,1,?1,?2,?3,0,0,0)",
        params![pk, kind, text],
    )
    .unwrap();
}

#[test]
fn search_membership_follows_the_blocks_that_hold_a_text() {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    // The rows stand alone: no session or message holds them.
    conn.pragma_update(None, "foreign_keys", false).unwrap();
    let id = intern(&conn, "synthetic walrus output").unwrap();
    let entries = |conn: &Connection| {
        (
            matching(conn, "text_fts", Some("walrus")).len() as i64,
            matching(conn, "output_fts", Some("walrus")).len() as i64,
        )
    };
    let group = |conn: &Connection| -> String {
        conn.query_row(
            &format!(
                "SELECT indexed || ':' || coalesce((SELECT group_concat(fts_table) FROM
                     (SELECT fts_table FROM search_pending WHERE text_id={id} ORDER BY fts_table)), '')
                 FROM text WHERE id={id}"
            ),
            [],
            |r| r.get(0),
        )
        .unwrap()
    };
    // A tool result joins the output index and is queued once.
    block(&conn, 1, "tool_result", id);
    block(&conn, 2, "tool_call", id);
    assert_eq!(group(&conn), "2:2");
    // A prose block joins the prose index too.
    block(&conn, 3, "text", id);
    assert_eq!(group(&conn), "3:1,2");
    assert_eq!(index_pending(&conn).unwrap(), 2);
    assert_eq!(group(&conn), "3:");
    assert_eq!(entries(&conn), (1, 1));
    // The output index keeps it while one tool block holds it.
    conn.execute("DELETE FROM block WHERE pk=1", []).unwrap();
    assert_eq!((group(&conn), entries(&conn)), ("3:".into(), (1, 1)));
    // Its last tool block leaving queues the entry's removal.
    conn.execute("DELETE FROM block WHERE pk=2", []).unwrap();
    assert_eq!((group(&conn), entries(&conn)), ("1:2".into(), (1, 1)));
    assert_eq!(index_pending(&conn).unwrap(), 1);
    assert_eq!((group(&conn), entries(&conn)), ("1:".into(), (1, 0)));
    // A document block holds it for the prose index after its last prose block.
    conn.execute(
        "INSERT INTO document_block(document_pk,position,kind,body,text_id) VALUES (1,0,'text',x'00',?1)",
        [id],
    )
    .unwrap();
    conn.execute("DELETE FROM block WHERE pk=3", []).unwrap();
    assert_eq!((group(&conn), entries(&conn)), ("1:".into(), (1, 0)));
    conn.execute("DELETE FROM document_block", []).unwrap();
    assert_eq!((group(&conn), entries(&conn)), ("0:1".into(), (1, 0)));
    assert_eq!(index_pending(&conn).unwrap(), 1);
    assert_eq!((group(&conn), entries(&conn)), ("0:".into(), (0, 0)));
    // A text that leaves before its entry is written keeps one queue row,
    // which settles to no entry.
    block(&conn, 4, "tool_result", id);
    assert_eq!(group(&conn), "2:2");
    conn.execute("DELETE FROM block WHERE pk=4", []).unwrap();
    assert_eq!(group(&conn), "0:2");
    assert_eq!(index_pending(&conn).unwrap(), 1);
    assert_eq!((group(&conn), entries(&conn)), ("0:".into(), (0, 0)));
}

#[test]
fn the_trigram_index_follows_tool_calls() {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.pragma_update(None, "foreign_keys", false).unwrap();
    let matches = |pattern: &str| {
        count(
            &conn,
            &format!(
                "SELECT count(*) FROM tool_call_trigram WHERE tool_call_trigram MATCH '{pattern}'"
            ),
        )
    };
    conn.execute(
        "INSERT INTO tool_call(pk,session_pk,call_block_pk,tool,command,path,outcome)
         VALUES (1,1,1,'shell','cargo walrus','/home/op/narwhal.rs','ok')",
        [],
    )
    .unwrap();
    assert_eq!((matches("command:walrus"), matches("path:narwhal")), (1, 1));
    conn.execute("UPDATE tool_call SET command='git beluga' WHERE pk=1", [])
        .unwrap();
    assert_eq!(
        (matches("command:walrus"), matches("command:beluga")),
        (0, 1)
    );
    conn.execute("DELETE FROM tool_call WHERE pk=1", [])
        .unwrap();
    assert_eq!((matches("command:beluga"), matches("path:narwhal")), (0, 0));
    conn.execute(
        "INSERT INTO tool_call_trigram(tool_call_trigram, rank) VALUES ('integrity-check', 1)",
        [],
    )
    .unwrap();
}

#[cfg(feature = "faults")]
mod durability;
mod search_fill;

#[test]
fn search_work_follows_the_queue_instead_of_text_history() {
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };
    const HISTORY: u64 = 8192;
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let tx = conn.transaction().unwrap();
    for number in 0..HISTORY {
        text(&tx, &format!("historical text {number}"), 0, true);
    }
    tx.commit().unwrap();
    // A completed bulk build has history and an empty queue when analyzed.
    conn.execute_batch("ANALYZE").unwrap();
    let id = text(&conn, "walrus", 1, true);
    assert_eq!(text(&conn, "walrus", 2, true), id);
    // A text removed before indexing must not strand the queue or enter FTS.
    conn.execute(
        "INSERT INTO search_pending(text_id,fts_table) VALUES (?1,1)",
        [i64::MAX],
    )
    .unwrap();
    let work = Arc::new(AtomicU64::new(0));
    let seen = Arc::clone(&work);
    conn.progress_handler(
        100,
        Some(move || {
            seen.fetch_add(100, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    assert_eq!(index_pending(&conn).unwrap(), 3);
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    // Even one VM instruction per unrelated historical text exceeds the
    // complete budget for two queued entries and the orphan's removal.
    assert!(
        work.load(Ordering::Relaxed) < HISTORY,
        "search drain visited history: {} VM instructions",
        work.load(Ordering::Relaxed)
    );
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 0);
    for table in ["text_fts", "output_fts"] {
        assert_eq!(matching(&conn, table, Some("walrus")), [id]);
    }
    assert_eq!(index_pending(&conn).unwrap(), 0);
}

#[test]
fn computed_entries_apply_only_to_the_queue_and_text_they_were_read_from() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let walrus = text(&conn, "walrus", 1, true);
    let narwhal = text(&conn, "narwhal", 2, true);
    // The writer indexed the queue itself after the search thread read it.
    let stale = pending_entries(&conn, SEARCH_BYTES).unwrap();
    assert_eq!(index_pending(&conn).unwrap(), 2);
    assert_eq!(apply_entries(&conn, &stale).unwrap(), 0);
    assert_eq!(entries(&conn, "text_fts"), 1);
    assert_eq!(entries(&conn, "output_fts"), 1);

    // A text removed and its key given to other content after the read.
    let orca = text(&conn, "orca", 1, true);
    let stale = pending_entries(&conn, SEARCH_BYTES).unwrap();
    remove(&conn, orca).unwrap();
    insert_alone(
        &conn,
        Some(orca),
        b"beluga-hash",
        b"beluga-norm",
        "beluga",
        1,
    );
    conn.execute(
        "INSERT OR IGNORE INTO search_pending(text_id,fts_table) VALUES (?1,1)",
        [orca],
    )
    .unwrap();
    assert_eq!(apply_entries(&conn, &stale).unwrap(), 0);
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 1);
    assert_eq!(index_pending(&conn).unwrap(), 1);
    assert_eq!(matching(&conn, "text_fts", Some("beluga"))[0], orca);
    assert_eq!(matching(&conn, "text_fts", Some("orca")).len() as i64, 0);
    assert_eq!(matching(&conn, "text_fts", Some("walrus"))[0], walrus);
    assert_eq!(matching(&conn, "output_fts", Some("narwhal"))[0], narwhal);
}
