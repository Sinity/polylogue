use super::super::*;
use crate::capture_read::{CaptureRead, RawRange, ReadRoute, SourceSelector, SourceStatus};
use crate::input_rows::{InputRows, ObservationRow};

fn observation(id: &str, bytes: Vec<u8>) -> ObservationRow {
    ObservationRow {
        id: id.into(),
        machine: "machine".into(),
        host: "host".into(),
        path_bytes: bytes,
        seen: crate::input_rows::Seen::Deleted,
        device: None,
        inode: None,
        size: None,
        mtime_ns: None,
        ctime_ns: None,
        polylogue_version: "test".into(),
        seen_at: 1,
        details_json: None,
        container: None,
    }
}
fn rows(id: &str, bytes: Vec<u8>) -> InputRows {
    InputRows {
        capture_read: None,
        input: id.into(),
        origin: "synthetic".into(),
        revision: None,
        chain: None,
        extents: Vec::new(),
        observation: Some(observation(id, bytes)),
    }
}
fn selector(bytes: &[u8]) -> SourceSelector {
    SourceSelector {
        origin: "synthetic".into(),
        machine: "machine".into(),
        path_bytes: bytes.to_vec(),
    }
}
fn database(primary: bool) -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(if primary {
        schema().unwrap().ddl.as_str()
    } else {
        crate::schema::ARCHIVE.as_str()
    })
    .unwrap();
    conn
}

#[test]
fn colliding_display_paths_keep_distinct_raw_paths() {
    for primary in [false, true] {
        let conn = database(primary);
        let paths = [b"/home/op/\xff".as_slice(), b"/home/op/\xfe".as_slice()];
        assert_eq!(
            String::from_utf8_lossy(paths[0]),
            String::from_utf8_lossy(paths[1])
        );
        for (n, path) in paths.iter().enumerate() {
            rows(&format!("input-{n}"), path.to_vec())
                .insert(&conn, 1)
                .unwrap();
        }
        for (n, path) in paths.iter().enumerate() {
            let (status, source) = Source::confirmed_source(&conn, &selector(path)).unwrap();
            assert_eq!(status, SourceStatus::Ready);
            let source = source.unwrap();
            assert_eq!(source.input, format!("input-{n}"));
            assert_eq!(source.observation.path_bytes.as_slice(), *path);
        }
    }
}

#[test]
fn source_read_keeps_body_when_extent_is_trimmed() {
    for primary in [false, true] {
        let conn = database(primary);
        let source = Source::of(&conn).unwrap();
        let source_read = CaptureRead {
            version: 1,
            route: ReadRoute::NativeCapture,
            source: selector(b"/home/op/\xff"),
            range: RawRange { start: 4, end: 12 },
            hash: polylogue_model::BlobHash::of(b"abcdefgh"),
            observed_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
            generation: None,
            raw_total: None,
            previous_revision: None,
            chunks: Vec::new(),
        };
        conn.execute(
            &source
                .sql("INSERT INTO @source.blob(hash,size,stored_size,storage) VALUES(?1,8,8,x'')"),
            [&source_read.hash.0[..]],
        )
        .unwrap();
        let mut row = rows("native", source_read.source.path_bytes.clone());
        row.capture_read = Some(source_read.clone());
        row.revision = Some(crate::input_rows::Revision {
            number: 1,
            produced: crate::input_rows::Produced::First,
            chain: 1,
            length: 12,
            seen_at: Some(1),
            mtime_ns: None,
            ctime_ns: None,
        });
        row.extents = vec![crate::input_rows::Extent {
            chain: 1,
            offset: 8,
            length: 4,
            blob: source_read.hash,
            blob_offset: 4,
        }];
        // The capture read checks the trimmed extent; the extent row keeps
        // no copy of it, since nothing reads one.
        row.insert(&conn, 7).unwrap();
        row.extents.clear();
        row.revision = None;
        row.observation.as_mut().unwrap().id = "overlap".into();
        row.observation.as_mut().unwrap().details_json = Some("{\"import\":\"kept\"}".into());
        row.insert(&conn, 8).unwrap();
        let details: String = conn
            .query_row(
                &source.sql("SELECT details_json FROM @source.observation WHERE id='overlap'"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        let details: serde_json::Value = serde_json::from_str(&details).unwrap();
        assert_eq!(details["import"], "kept");
        assert_eq!(
            serde_json::from_value::<CaptureRead>(details["capture_read"].clone()).unwrap(),
            source_read
        );
        row.observation = None;
        let refused = "the capture read does not describe these source rows; this is a Polylogue bug; report it with the request ID";
        assert_eq!(row.insert(&conn, 9).unwrap_err().to_string(), refused);
        row.extents = vec![crate::input_rows::Extent {
            chain: 1,
            offset: 8,
            length: 4,
            blob: source_read.hash,
            blob_offset: 0,
        }];
        assert_eq!(row.insert(&conn, 9).unwrap_err().to_string(), refused);
        let count: i64 = conn
            .query_row(
                &source.sql("SELECT count(*) FROM @source.extent"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
}

#[test]
fn source_snapshot_and_observation_keep_the_same_observation() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = crate::Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let conn = Connection::open(archive.archive_db()).unwrap();
    let path = b"/home/op/\xff";
    rows("native", path.to_vec()).insert(&conn, 1).unwrap();
    conn.execute(
        "INSERT INTO meta(key,value) VALUES('write_counter','1')",
        [],
    )
    .unwrap();
    drop(conn);
    let pool = crate::ReadPool::open(&archive).unwrap();
    let snapshot = pool
        .reader()
        .unwrap()
        .source_snapshot(selector(path))
        .unwrap();
    assert_eq!(snapshot.status, SourceStatus::Ready);
    assert_eq!(
        snapshot
            .confirmed
            .as_ref()
            .unwrap()
            .observation
            .path_bytes
            .as_slice(),
        path.as_slice()
    );
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader.source_observation(&selector(path)).unwrap().1,
        snapshot.confirmed
    );
    assert_eq!(
        reader
            .known_input_at("machine", path)
            .unwrap()
            .unwrap()
            .path_bytes
            .as_slice(),
        path.as_slice()
    );
    assert!(
        reader
            .known_input_at("machine", b"/home/op/\xfe")
            .unwrap()
            .is_none()
    );
}

#[test]
fn source_history_checks_observations_at_a_fixed_position() {
    for primary in [false, true] {
        let conn = database(primary);
        let path = b"/home/op/\xff";
        rows("first", path.to_vec()).insert(&conn, 2).unwrap();
        rows("later", path.to_vec()).insert(&conn, 6).unwrap();
        for at in [2, 4] {
            let (status, first) =
                Source::confirmed_source_at(&conn, &selector(path), Some(at)).unwrap();
            assert_eq!(status, SourceStatus::Ready);
            assert_eq!(first.unwrap().observation.id, "first");
        }
        assert_eq!(
            Source::confirmed_source(&conn, &selector(path))
                .unwrap()
                .1
                .unwrap()
                .observation
                .id,
            "later"
        );
    }
}
