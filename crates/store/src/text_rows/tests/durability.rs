//! A crash between the full-text files' commit and the queue's leaves the
//! queue, and applying it again gives the same entries.
use super::*;
use crate::fault::{self, Action};

#[test]
fn a_crash_between_a_file_commit_and_the_queue_is_completed_by_a_flush() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let path = fixture.archive.index_db();
    let conn = open_write(&path, cache::SETUP).unwrap();
    let walrus = text(&conn, "walrus tusk", 1, true);
    let narwhal = text(&conn, "narwhal tusk", 1, true);
    let orca = text(&conn, "orca", 2, true);
    assert_eq!(index_pending(&conn).unwrap(), 3);
    let ranked = |conn: &Connection| -> Vec<(i64, f64)> {
        conn.prepare(&format!(
            "SELECT text_id, rank FROM ({}) ORDER BY rank, text_id",
            crate::search::ranked_texts(crate::search::SnippetIndex::Prose, "'tusk'", None, None)
        ))
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
    };
    // One text joins, one leaves its index, and the files commit both
    // before the process dies.
    let beluga = text(&conn, "beluga tusk tusk", 1, true);
    remove(&conn, orca).unwrap();
    fault::clear();
    fault::arm("search.before_queue", "", Action::Fail, 0, 1);
    assert!(index_pending(&conn).is_err());
    fault::clear();
    drop(conn);
    assert_eq!(
        matching(&open_write(&path, cache::SETUP).unwrap(), "text_fts", None).len(),
        3
    );

    let conn = open_write(&path, cache::SETUP).unwrap();
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 2);
    assert_eq!(index_pending(&conn).unwrap(), 2);
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 0);
    assert_eq!(matching(&conn, "text_fts", None), [walrus, narwhal, beluga]);
    assert!(matching(&conn, "output_fts", None).is_empty());

    // The same texts written once rank the same.
    let other = tempfile::tempdir().unwrap();
    let once = Fixture::new(&other.path().join("archive")).unwrap();
    let once = open_write(&once.archive.index_db(), cache::SETUP).unwrap();
    for body in ["walrus tusk", "narwhal tusk"] {
        text(&once, body, 1, true);
    }
    let removed = text(&once, "orca", 2, true);
    text(&once, "beluga tusk tusk", 1, true);
    remove(&once, removed).unwrap();
    index_pending(&once).unwrap();
    assert_eq!(ranked(&conn), ranked(&once));
    assert_eq!(ranked(&conn).len(), 3);
}
