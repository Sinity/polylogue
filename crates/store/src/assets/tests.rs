//! Range delivery reads the current chain, including bytes in several new blobs.
use super::*;
use crate::{Archive, Batch, Change, InputKey, Observation, ReadPool, Writer};
use polylogue_model::BlobHash;
use std::sync::Arc;

#[test]
fn segmented_asset_ranges_and_hashes_are_exact_and_old_selections_refuse_rewrites() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let input = InputKey {
        id: "synthetic-segmented-asset".into(),
        format: "export-file".into(),
        arrival_rank: 0,
    };
    let sender = writer.sender().unwrap();
    let store = Arc::new(archive.blobs().unwrap());
    let observation = Observation {
        machine: "synthetic".into(),
        host: "synthetic".into(),
        path: "/synthetic/asset.bin".into(),
        path_bytes: b"/synthetic/asset.bin".to_vec(),
        facts: None,
        seen_at_ms: 1,
        import: None,
        container: None,
    };
    let original = "prefix λ🙂 suffix".as_bytes();
    let split = 8; // Inside the UTF-8 sequence; this API delivers bytes.
    for (start, end, continues) in [(0, split, false), (split, original.len(), true)] {
        let (new_blob, file) = archive.prepare_blob(&original[start..end]).unwrap();
        let mut stored = sender
            .store_group(
                vec![crate::staging::PreparedBatch {
                    batch: Batch {
                        required_disposition: None,
                        input: input.clone(),
                        change: Change::First,
                        observation: Some(observation.clone()),
                        continues,
                        new_blob: Some(new_blob),
                        blobs: vec![],
                        length: end as u64,
                        parse: vec![],
                        acquisition: None,
                        basis: None,
                        release: Default::default(),
                    },
                    files: file.into_iter().collect(),
                }],
                Arc::clone(&store),
            )
            .unwrap();
        stored.pop().expect("one captured source chunk").unwrap();
    }
    assert!(writer.flush().unwrap().errors.is_empty());
    let pool = ReadPool::open(&archive).unwrap();
    let asset = {
        let reader = pool.reader().unwrap();
        let source = reader.asset_source(&input.id).unwrap().unwrap();
        assert!(
            source.blob.is_none(),
            "segmented bytes must not pretend to be one stored blob"
        );
        let asset = AssetMatch {
            input: input.id.clone(),
            chain: source.chain,
            blob: source.blob,
            source_version: source.version,
            size: source.length,
            members: vec![],
            names: vec![],
            media_types: vec![],
            declared_sizes: vec![],
        };
        assert_eq!(
            reader.attachment_hash(&asset).unwrap(),
            BlobHash::of(original)
        );
        assert_eq!(
            reader.attachment_range(&asset, 2..13).unwrap(),
            &original[2..13]
        );
        assert_eq!(
            reader.attachment_range(&asset, 0..asset.size).unwrap(),
            original
        );
        assert!(
            reader
                .attachment_range(&asset, asset.size..asset.size + 1)
                .is_err()
        );
        asset
    };
    let replacement = b"replacement body";
    let (new_blob, file) = archive.prepare_blob(replacement).unwrap();
    let mut stored = sender
        .store_group(
            vec![crate::staging::PreparedBatch {
                batch: Batch {
                    required_disposition: None,
                    input: input.clone(),
                    change: Change::Rewritten,
                    observation: Some(Observation {
                        seen_at_ms: 2,
                        ..observation
                    }),
                    continues: false,
                    new_blob: Some(new_blob),
                    blobs: vec![],
                    length: replacement.len() as u64,
                    parse: vec![],
                    acquisition: None,
                    basis: None,
                    release: Default::default(),
                },
                files: file.into_iter().collect(),
            }],
            Arc::clone(&store),
        )
        .unwrap();
    stored.pop().expect("one captured rewrite").unwrap();
    assert!(writer.flush().unwrap().errors.is_empty());
    let reader = pool.reader().unwrap();
    assert!(reader.attachment_hash(&asset).is_err());
    assert!(reader.attachment_range(&asset, 0..asset.size).is_err());
}

// These fixtures exercise source metadata discovery, not byte admission or availability.
fn member_scan_fixture(prefix: &str) -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(&format!(
        "CREATE TABLE {prefix}observation(id TEXT PRIMARY KEY,input TEXT NOT NULL,details_json TEXT);
         CREATE TABLE {prefix}observation_container(observation_id TEXT,container_input TEXT);"
    )).unwrap();
    for (id, input, namespace, member) in [
        ("root-a", "archive", "scope-a", "original.zip"),
        ("member-a", "asset", "scope-a", "assets/a.bin"),
        ("root-b", "archive", "scope-b", "received.zip"),
        ("member-b", "other", "scope-b", "assets/b.bin"),
    ] {
        let details = serde_json::json!({"import":{
            "version":1,"import_id":namespace,"member":member
        }});
        conn.execute(
            &format!("INSERT INTO {prefix}observation VALUES (?1,?2,?3)"),
            params![id, input, details.to_string()],
        )
        .unwrap();
    }
    conn.execute(
        &format!("INSERT INTO {prefix}observation_container VALUES ('member-a','archive')"),
        [],
    )
    .unwrap();
    conn.execute(
        &format!("INSERT INTO {prefix}observation VALUES ('no-import','unrelated',NULL)"),
        [],
    )
    .unwrap();
    conn
}

#[test]
fn member_discovery_replays_selected_inputs_without_borrowing_another_import_namespace() {
    for prefix in ["", "source_"] {
        let conn = member_scan_fixture(prefix);
        let mut all = Vec::new();
        let through = visit_import_members(&conn, prefix, 0, None, |source| {
            all.push((
                source.observation,
                source.input,
                source.membership.import_id,
            ));
            Ok(())
        })
        .unwrap();
        assert_eq!(through, Some(5));
        assert_eq!(
            all,
            vec![
                (
                    "member-a".to_owned(),
                    "asset".to_owned(),
                    "scope-a".to_owned()
                ),
                (
                    "root-b".to_owned(),
                    "archive".to_owned(),
                    "scope-b".to_owned()
                ),
                (
                    "member-b".to_owned(),
                    "other".to_owned(),
                    "scope-b".to_owned()
                ),
            ]
        );
        assert_eq!(
            visit_import_members(&conn, prefix, 5, None, |_| panic!(
                "nothing is newer than the scan cursor"
            ))
            .unwrap(),
            None
        );
        let mut replayed = Vec::new();
        visit_import_members(&conn, prefix, 0, Some("archive"), |source| {
            replayed.push((source.observation, source.membership.import_id));
            Ok(())
        })
        .unwrap();
        assert_eq!(replayed, vec![("root-b".to_owned(), "scope-b".to_owned())]);
        let mut replayed = Vec::new();
        visit_import_members(&conn, prefix, 0, Some("asset"), |source| {
            replayed.push(source.observation);
            Ok(())
        })
        .unwrap();
        assert_eq!(replayed, vec!["member-a"]);
    }
}

#[test]
fn member_discovery_filters_before_decoding_and_propagates_visitor_refusal() {
    let conn = member_scan_fixture("source_");
    conn.execute(
        "UPDATE source_observation SET details_json=?1 WHERE id='member-b'",
        [r#"{"import":{"version":99,"import_id":"scope-b","member":"assets/b.bin"}}"#],
    )
    .unwrap();
    let mut selected = Vec::new();
    visit_import_members(&conn, "source_", 0, Some("asset"), |source| {
        selected.push(source.observation);
        Ok(())
    })
    .unwrap();
    assert_eq!(selected, vec!["member-a"]);
    assert!(matches!(
        visit_import_members(&conn, "source_", 0, Some("other"), |_| Ok(())),
        Err(StoreError::Invalid(_))
    ));
    let mut visited = Vec::new();
    let error = visit_import_members(&conn, "source_", 0, None, |source| {
        visited.push(source.observation);
        Err(StoreError::Conflict("synthetic visitor refusal".into()))
    });
    assert!(matches!(error, Err(StoreError::Conflict(_))));
    assert_eq!(visited, vec!["member-a"]);
}

#[test]
fn import_members_and_asset_rows_move_their_import_counters() {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::ensure_index(&conn).unwrap();
    let counter = |import: &str| -> Option<i64> {
        conn.query_row(
            "SELECT changed FROM import_change WHERE import_id=?1",
            [import],
            |row| row.get(0),
        )
        .optional()
        .unwrap()
    };
    for (observation, import) in [("one", "import-a"), ("two", "import-b")] {
        let member = ObservedMember {
            observation: observation.into(),
            input: "shared-input".into(),
            membership: crate::ImportMember {
                version: 1,
                import_id: import.into(),
                member: format!("{observation}.bin"),
                authorship: None,
                format: None,
            },
        };
        project_member(&conn, &member, None, None).unwrap();
    }
    assert_eq!(
        (counter("import-a"), counter("import-b")),
        (Some(1), Some(1))
    );
    count_input_change(&conn, "shared-input").unwrap();
    assert_eq!(
        (counter("import-a"), counter("import-b")),
        (Some(2), Some(2))
    );
    count_input_change(&conn, "unrelated-input").unwrap();
    assert_eq!(
        (counter("import-a"), counter("import-b")),
        (Some(2), Some(2))
    );
}
