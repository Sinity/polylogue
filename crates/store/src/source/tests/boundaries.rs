use super::super::*;
use crate::capture_read::SourceStatus;
use crate::input_rows::{Extent, InputRows, ObservationRow, Revision};

// These rows model source commits; byte comparison belongs to the sampler.
fn append(conn: &Connection, input: &str, id: &str, seq: i64, from: i64, end: i64) {
    let hash = polylogue_model::BlobHash::of(id.as_bytes());
    conn.execute(
        "INSERT INTO blob(hash,size,stored_size,storage) VALUES(?1,?2,?2,x'')",
        rusqlite::params![&hash.0[..], end - from],
    )
    .unwrap();
    InputRows {
        input: input.into(),
        origin: "synthetic".into(),
        capture_read: None,
        revision: Some(Revision {
            number: 1,
            produced: "first".into(),
            chain: 1,
            length: end,
        }),
        extents: vec![Extent {
            chain: 1,
            offset: from,
            length: end - from,
            blob: hash,
            blob_offset: 0,
        }],
        observation: Some(ObservationRow {
            id: id.into(),
            machine: "machine".into(),
            host: "host".into(),
            path: input.into(),
            path_bytes: Some(input.as_bytes().to_vec()),
            class: "appended".into(),
            revision: Some(1),
            chain: None,
            length: Some(end),
            device: None,
            inode: None,
            size: Some(end),
            mtime_ns: None,
            ctime_ns: None,
            polylogue_version: "test".into(),
            seen_at: seq,
            details_json: None,
            container: None,
        }),
    }
    .insert(conn, seq)
    .unwrap();
    conn.execute(
        "INSERT INTO meta(key,value) VALUES('write_counter',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [seq.to_string()],
    ).unwrap();
}

#[test]
fn source_rows_without_a_commit_position_stay_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = crate::Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let conn = Connection::open(archive.archive_db()).unwrap();
    append(&conn, "native", "first", 1, 0, 4);
    conn.execute("DELETE FROM meta WHERE key='write_counter'", [])
        .unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let (status, observation) = reader
        .source_observation(&crate::capture_read::SourceSelector {
            origin: "synthetic".into(),
            machine: "machine".into(),
            path_bytes: b"native".to_vec(),
        })
        .unwrap();
    assert_eq!(status, SourceStatus::MissingEvidence);
    assert!(observation.is_some());
}
