use super::super::*;
use crate::input_rows::{Extent, InputRows, ObservationRow, Revision};
use crate::primary::SourceStatus;

// These rows model source admissions; byte comparison belongs to the sampler.
fn append(conn: &Connection, input: &str, id: &str, seq: i64, from: i64, end: i64) {
    let hash = polylogue_model::BlobHash::of(id.as_bytes());
    conn.execute(
        "INSERT INTO blob(hash,size,stored_size,storage) VALUES(?1,?2,?2,x'')",
        rusqlite::params![&hash.0[..], end - from],
    )
    .unwrap();
    InputRows {
        input: input.into(),
        format: "synthetic".into(),
        acquisition: None,
        revision: Some(Revision {
            number: 1,
            produced: "first".into(),
            chain: 1,
            length: end,
        }),
        extent: Some(Extent {
            chain: 1,
            offset: from,
            length: end - from,
            blob: hash,
            blob_offset: 0,
        }),
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
fn pre_io_boundary_excludes_later_admissions_and_keeps_the_original_subject_head() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = crate::Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let conn = Connection::open(archive.archive_db()).unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let empty = pool.reader().unwrap().source_boundary().unwrap().unwrap();
    append(&conn, "native", "first", 1, 0, 4);
    let boundary = pool.reader().unwrap().source_boundary().unwrap().unwrap();
    append(&conn, "unrelated", "other", 2, 0, 4);
    {
        let reader = pool.reader().unwrap();
        assert!(
            reader
                .source_subject_at(&empty, "native")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            reader
                .source_subject_at(&boundary, "native")
                .unwrap()
                .unwrap()
                .observation,
            "first"
        );
        let mut points = Vec::new();
        reader
            .visit_revisions_at_length_before(&boundary, "native", 4, |point| {
                points.push(point);
                Ok(false)
            })
            .unwrap();
        assert_eq!(points.len(), 1);
    }
    append(&conn, "native", "later", 3, 4, 8);
    {
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader
                .source_subject_at(&boundary, "native")
                .unwrap()
                .unwrap()
                .observation,
            "first"
        );
        let mut points = Vec::new();
        reader
            .visit_revisions_at_length_before(&boundary, "native", 8, |point| {
                points.push(point);
                Ok(false)
            })
            .unwrap();
        assert!(points.is_empty());
    }
    archive.advance_generation_for_test().unwrap();
    assert!(matches!(
        pool.reader()
            .unwrap()
            .source_subject_at(&boundary, "native"),
        Err(crate::StoreError::ScopeChanged)
    ));
}

#[test]
fn source_rows_without_a_recorded_admission_counter_remain_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = crate::Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let conn = Connection::open(archive.archive_db()).unwrap();
    append(&conn, "native", "first", 1, 0, 4);
    conn.execute("DELETE FROM meta WHERE key='write_counter'", [])
        .unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    assert!(reader.source_boundary().unwrap().is_none());
    let (status, observation) = reader
        .source_observation(&crate::primary::SourceSelector {
            origin: "synthetic".into(),
            machine: "machine".into(),
            path_bytes: b"native".to_vec(),
        })
        .unwrap();
    assert_eq!(status, SourceStatus::MissingEvidence);
    assert!(observation.is_some());
}
