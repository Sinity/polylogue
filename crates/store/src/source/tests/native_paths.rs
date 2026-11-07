use super::super::*;
use crate::input_rows::{InputRows, ObservationRow};
use crate::primary::{
    AcquisitionReceipt, AcquisitionRoute, RawRange, SourceSelector, SourceStatus,
};

fn observation(id: &str, bytes: Option<Vec<u8>>) -> ObservationRow {
    ObservationRow {
        id: id.into(),
        machine: "machine".into(),
        host: "host".into(),
        path: "/home/op/�".into(),
        path_bytes: bytes,
        class: "first".into(),
        revision: None,
        chain: None,
        length: None,
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
fn rows(id: &str, bytes: Option<Vec<u8>>) -> InputRows {
    InputRows {
        acquisition: None,
        input: id.into(),
        format: "synthetic".into(),
        revision: None,
        extent: None,
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
        &schema().unwrap().ddl
    } else {
        crate::schema::ARCHIVE
    })
    .unwrap();
    conn
}

#[test]
fn colliding_display_paths_keep_distinct_exact_native_identities() {
    for primary in [false, true] {
        let conn = database(primary);
        let paths = [b"/home/op/\xff".as_slice(), b"/home/op/\xfe".as_slice()];
        assert_eq!(
            String::from_utf8_lossy(paths[0]),
            String::from_utf8_lossy(paths[1])
        );
        for (n, path) in paths.iter().enumerate() {
            rows(&format!("input-{n}"), Some(path.to_vec()))
                .insert(&conn, 1)
                .unwrap();
        }
        for (n, path) in paths.iter().enumerate() {
            let (status, source) = Source::confirmed_source(&conn, &selector(path)).unwrap();
            assert_eq!(status, SourceStatus::Ready);
            let source = source.unwrap();
            assert_eq!(source.input, format!("input-{n}"));
            assert_eq!(source.observation.path_bytes.as_deref(), Some(*path));
        }
        rows("legacy", None).insert(&conn, 2).unwrap();
        assert_eq!(
            Source::confirmed_source(&conn, &selector(paths[0]))
                .unwrap()
                .0,
            SourceStatus::MissingEvidence
        );
        rows("legacy-2", None).insert(&conn, 3).unwrap();
        assert_eq!(
            Source::confirmed_source(&conn, &selector(paths[0]))
                .unwrap()
                .0,
            SourceStatus::Ambiguous
        );
        rows("exact-again", Some(paths[0].to_vec()))
            .insert(&conn, 4)
            .unwrap();
        assert_eq!(
            Source::confirmed_source(&conn, &selector(paths[0]))
                .unwrap()
                .0,
            SourceStatus::Ready
        );
    }
}

#[test]
fn acquisition_evidence_keeps_original_body_when_an_extent_is_trimmed_or_absent() {
    for primary in [false, true] {
        let conn = database(primary);
        let source = Source::of(&conn).unwrap();
        let receipt = AcquisitionReceipt {
            version: 1,
            route: AcquisitionRoute::NativeCapture,
            source: selector(b"/home/op/\xff"),
            range: RawRange { start: 4, end: 12 },
            hash: polylogue_model::BlobHash::of(b"abcdefgh"),
            observed_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
            generation: None,
            raw_total: None,
            raw_predecessor: None,
        };
        conn.execute(
            &source
                .sql("INSERT INTO @source.blob(hash,size,stored_size,storage) VALUES(?1,8,8,x'')"),
            [&receipt.hash.0[..]],
        )
        .unwrap();
        let mut row = rows("native", Some(receipt.source.path_bytes.clone()));
        row.acquisition = Some(receipt.clone());
        row.revision = Some(crate::input_rows::Revision {
            number: 1,
            produced: "first".into(),
            chain: 1,
            length: 12,
        });
        row.extent = Some(crate::input_rows::Extent {
            chain: 1,
            offset: 8,
            length: 4,
            blob: receipt.hash,
            blob_offset: 4,
        });
        row.insert(&conn, 7).unwrap();
        let packed: Vec<u8> = conn
            .query_row(
                &source.sql("SELECT acquisition FROM @source.extent"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            crate::encoding::unpack_json::<AcquisitionReceipt>(&packed).unwrap(),
            receipt
        );
        row.extent = None;
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
            serde_json::from_value::<AcquisitionReceipt>(details["acquisition"].clone()).unwrap(),
            receipt
        );
        row.observation = None;
        assert!(row.insert(&conn, 9).is_err());
        row.extent = Some(crate::input_rows::Extent {
            chain: 1,
            offset: 8,
            length: 4,
            blob: receipt.hash,
            blob_offset: 0,
        });
        assert!(row.insert(&conn, 9).is_err());
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
fn queued_source_snapshot_and_reader_keep_the_same_exact_observation() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = crate::Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let conn = Connection::open(archive.archive_db()).unwrap();
    let path = b"/home/op/\xff";
    rows("native", Some(path.to_vec()))
        .insert(&conn, 1)
        .unwrap();
    conn.execute(
        "INSERT INTO meta(key,value) VALUES('write_counter','1')",
        [],
    )
    .unwrap();
    drop(conn);
    let writer = crate::Writer::start(&archive).unwrap();
    let snapshot = writer
        .source_snapshot(selector(path), "synthetic:1")
        .unwrap();
    assert_eq!(snapshot.basis.status, SourceStatus::Ready);
    assert_eq!(snapshot.basis.confirmed.as_deref(), Some("native"));
    assert_eq!(snapshot.basis.pending, None);
    assert_eq!(
        snapshot
            .confirmed
            .as_ref()
            .unwrap()
            .observation
            .path_bytes
            .as_deref(),
        Some(path.as_slice())
    );
    let pool = crate::ReadPool::open(&archive).unwrap();
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
            .as_deref(),
        Some(path.as_slice())
    );
    assert!(
        reader
            .known_input_at("machine", b"/home/op/\xfe")
            .unwrap()
            .is_none()
    );
}

#[test]
fn source_history_checks_only_observations_admitted_at_the_original_cut() {
    for primary in [false, true] {
        let conn = database(primary);
        let path = b"/home/op/\xff";
        rows("first", Some(path.to_vec())).insert(&conn, 2).unwrap();
        rows("legacy", None).insert(&conn, 4).unwrap();
        rows("later", Some(path.to_vec())).insert(&conn, 6).unwrap();
        let (status, first) = Source::confirmed_source_at(&conn, &selector(path), Some(2)).unwrap();
        assert_eq!(status, SourceStatus::Ready);
        assert_eq!(first.unwrap().observation.id, "first");
        assert_eq!(
            Source::confirmed_source_at(&conn, &selector(path), Some(4))
                .unwrap()
                .0,
            SourceStatus::MissingEvidence
        );
        assert_eq!(
            Source::confirmed_source(&conn, &selector(path))
                .unwrap()
                .1
                .unwrap()
                .observation
                .id,
            "later"
        );
        assert!(Source::observation_matches(&conn, &selector(path), "first", Some(2)).unwrap());
        assert!(
            !Source::observation_matches(&conn, &selector(b"other"), "first", Some(2)).unwrap()
        );
        assert!(!Source::observation_matches(&conn, &selector(path), "later", Some(2)).unwrap());
    }
}
