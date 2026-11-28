use super::*;

fn facts(inode: u64, size: u64, mtime: i128) -> Option<FileFacts> {
    Some(FileFacts {
        device: 1,
        inode,
        size,
        mtime_ns: mtime,
        ctime_ns: mtime,
    })
}

#[test]
fn classifies_every_kind_of_change() {
    assert_eq!(classify(None, facts(1, 10, 1)), Change::First);
    assert_eq!(
        classify(facts(1, 10, 1), facts(1, 10, 1)),
        Change::Unchanged
    );
    assert_eq!(
        classify(facts(1, 10, 1), facts(1, 25, 2)),
        Change::Appended { from: 10 }
    );
    assert_eq!(classify(facts(1, 10, 1), facts(1, 5, 2)), Change::Rewritten);
    assert_eq!(
        classify(facts(1, 10, 1), facts(2, 30, 2)),
        Change::Rewritten
    );
    assert_eq!(classify(facts(1, 10, 1), None), Change::Disappeared);
    assert_eq!(classify(None, None), Change::Unchanged);
    // A remount can renumber the device. The caller must compare bytes
    // before identifying the file as a continuation.
    let remounted = facts(1, 10, 1).map(|f| FileFacts { device: 7, ..f });
    assert_eq!(classify(facts(1, 10, 1), remounted), Change::Rewritten);
    let touched = remounted.map(|f| FileFacts { ctime_ns: 2, ..f });
    assert_eq!(classify(facts(1, 10, 1), touched), Change::Rewritten);
}

/// The segment is cut back to its committed end when a writer opens, so an
/// end recorded behind the one already committed is refused and the stored
/// end stays.
#[test]
fn the_committed_segment_end_only_grows() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::ensure_archive(&conn).unwrap();
    let end = |conn: &rusqlite::Connection| -> i64 {
        conn.query_row("SELECT committed_end FROM segment", [], |row| row.get(0))
            .unwrap()
    };
    record_committed_end(&conn, 100).unwrap();
    record_committed_end(&conn, 100).unwrap();
    record_committed_end(&conn, 250).unwrap();
    assert_eq!(end(&conn), 250);
    assert!(matches!(
        record_committed_end(&conn, 120),
        Err(crate::StoreError::Internal(_))
    ));
    assert_eq!(end(&conn), 250);
}
