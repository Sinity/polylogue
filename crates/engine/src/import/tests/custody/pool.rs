use super::*;

#[test]
fn watched_pool_rejection_does_not_publish_unowned_raw_or_decoded_bytes() {
    for image in [false, true] {
        let world = ClaudeHome::new();
        let path = world.file("pool-custody.jsonl");
        let decoded = vec![b'A'; polylogue_store::blobs::INLINE_LIMIT + 1024];
        let bytes = if image {
            let encoded = "QUFB".repeat(decoded.len() / 3 + 1);
            format!(
                "{}\n",
                serde_json::json!({"type":"user","uuid":"pool-image","sessionId":"pool-custody","message":{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":encoded}}]}})
            )
        } else {
            claude_user("pool-user", "pool-custody", &"x".repeat(decoded.len()))
        };
        std::fs::write(&path, &bytes).unwrap();
        let raw = hashed_blobs::Hash::of(bytes.as_bytes());
        let (mut engine, _lock) = engine(&world, 1 << 20);
        let input = engine.location_id(&path);
        let conn = rusqlite::Connection::open(world.root.join("archive.db")).unwrap();
        conn.execute_batch(&format!("CREATE TRIGGER reject_pool BEFORE INSERT ON input WHEN NEW.id='{input}' BEGIN SELECT RAISE(ABORT,'synthetic pool first admission rejection'); END;")).unwrap();
        let pass = engine.converge().unwrap();
        assert!(!pass.errors.is_empty());
        assert!(
            !engine.blobs.contains(&raw),
            "rejected raw pool bytes have no canonical owner"
        );
        assert_eq!(world.count("SELECT count(*) AS n FROM a.blob"), 0);
        conn.execute_batch("DROP TRIGGER reject_pool").unwrap();
        let pass = engine.converge().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
}

fn image_record(id: &str, session: &str, marker: &str, base64: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({"type":"user","uuid":id,"sessionId":session,"message":{"role":"user","content":[{"type":"text","text":marker},{"type":"image","source":{"type":"base64","media_type":"image/png","data":base64}}]}})
    )
}
fn image_payload(letter: char) -> (String, Vec<u8>) {
    let (word, bytes) = match letter {
        'A' => ("QUFB", b"AAA".as_slice()),
        _ => ("QkJC", b"BBB".as_slice()),
    };
    let count = polylogue_store::blobs::INLINE_LIMIT / 3 + 1024;
    (word.repeat(count), bytes.repeat(count))
}
fn assert_blob(engine: &Engine, hash: polylogue_model::BlobHash, expected: &[u8]) {
    use std::io::Read;
    let mut reader = engine
        .blobs
        .read_plain(
            &hashed_blobs::Hash(hash.0),
            Some(expected.len() as u64),
            || Ok(()),
        )
        .unwrap();
    let mut actual = Vec::new();
    reader.read_to_end(&mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn decoded_ownership_survives_failed_index_admission_and_provider_free_rebuild() {
    let world = ClaudeHome::new();
    let (encoded, decoded) = image_payload('A');
    let path = world.file("decoded-custody.jsonl");
    std::fs::write(
        &path,
        image_record(
            "image-user",
            "decoded-custody",
            "synthetic decoded content",
            &encoded,
        ),
    )
    .unwrap();
    let hash = polylogue_model::BlobHash::of(&decoded);
    let original;
    {
        let (mut engine, _lock) = engine(&world, 1 << 20);
        let index = rusqlite::Connection::open(world.root.join("index.db")).unwrap();
        index.execute_batch("CREATE TRIGGER reject_decoded_message BEFORE INSERT ON message BEGIN SELECT RAISE(ABORT,'synthetic derived admission refusal'); END;").unwrap();
        let pass = engine.converge().unwrap();
        assert!(!pass.errors.is_empty());
        assert_eq!(
            world.count("SELECT count(*) AS n FROM a.extracted_blob WHERE state='present'"),
            1
        );
        assert_eq!(world.count("SELECT count(*) AS n FROM message"), 0);
        assert_blob(&engine, hash, &decoded);
        original = world
            .reads()
            .reader()
            .unwrap()
            .query("SELECT * FROM a.extracted_blob", &[])
            .unwrap();
    }
    std::fs::remove_file(path).unwrap();
    polylogue_testkit::delete_index(&world.root);
    let (mut engine, _lock) = engine(&world, 1 << 20);
    let pass = engine.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(world.count("SELECT count(*) AS n FROM message"), 1);
    assert_eq!(
        world
            .reads()
            .reader()
            .unwrap()
            .query("SELECT * FROM a.extracted_blob", &[])
            .unwrap(),
        original
    );
    assert_blob(&engine, hash, &decoded);
}

#[test]
fn input_excision_without_an_index_preserves_another_decoded_source_owner() {
    let world = ClaudeHome::new();
    let (encoded, decoded) = image_payload('A');
    let first = world.file("first-image.jsonl");
    let second = world.file("second-image.jsonl");
    std::fs::write(
        &first,
        image_record(
            "first-image",
            "first-image",
            "First supplied image",
            &encoded,
        ),
    )
    .unwrap();
    std::fs::write(
        &second,
        image_record(
            "second-image",
            "second-image",
            "Second supplied image",
            &encoded,
        ),
    )
    .unwrap();
    let input;
    let hash = polylogue_model::BlobHash::of(&decoded);
    {
        let (mut engine, _lock) = engine(&world, 1 << 20);
        let pass = engine.converge().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        input = engine.location_id(&first);
        assert_eq!(world.count("SELECT count(*) AS n FROM a.extracted_blob"), 2);
    }
    std::fs::remove_file(first).unwrap();
    std::fs::remove_file(second).unwrap();
    polylogue_testkit::delete_index(&world.root);
    let (mut engine, _lock) = engine(&world, 1 << 20);
    engine
        .excise(ExciseRequest {
            id: "remove-first-decoded-owner".into(),
            scope: ExciseScope::Input(input),
            reason: None,
            dry_run: false,
        })
        .unwrap();
    assert_eq!(world.count("SELECT count(*) AS n FROM a.extracted_blob"), 1);
    assert_blob(&engine, hash, &decoded);
    let pass = engine.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(world.count("SELECT count(*) AS n FROM message"), 1);
    assert_blob(&engine, hash, &decoded);
}

#[test]
fn a_stale_decode_cannot_reinstall_bytes_after_same_length_source_excision() {
    use polylogue_store::{
        Batch, Change, InputKey,
        retention::{Extraction, PreparedBatch},
    };
    let world = ClaudeHome::new();
    let (encoded, decoded) = image_payload('A');
    let (other_encoded, other_decoded) = image_payload('B');
    let path = world.file("stale-image.jsonl");
    std::fs::write(
        &path,
        image_record(
            "stale-image",
            "stale-image",
            "Synthetic stale image",
            &encoded,
        ),
    )
    .unwrap();
    std::fs::write(
        world.file("surviving-image.jsonl"),
        image_record("survivor", "survivor", "Synthetic survivor", &other_encoded),
    )
    .unwrap();
    let (mut engine, _lock) = engine(&world, 1 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    let input = engine.location_id(&path);
    let row = engine
        .reads
        .reader()
        .unwrap()
        .query(
            "SELECT * FROM a.extracted_blob WHERE input=?1",
            &[polylogue_store::SqlValue::Text(input.clone())],
        )
        .unwrap()
        .pop()
        .unwrap();
    let chain = row["chain"].as_i64().unwrap();
    let range = row["source_start"].as_u64().unwrap()..row["source_end"].as_u64().unwrap();
    let source_bytes = std::fs::read(&path).unwrap();
    let source_hash =
        polylogue_model::BlobHash::of(&source_bytes[range.start as usize..range.end as usize]);
    let (piece, file) = engine.archive.prepare_piece(&[], &decoded).unwrap();
    let hash = piece.hash;
    let pending = PreparedBatch {
        batch: Batch {
            input: InputKey {
                id: input.clone(),
                format: "claude-code".into(),
                route_rank: 0,
            },
            change: Change::Reparse { chain },
            observation: None,
            continues: false,
            piece: None,
            blobs: vec![piece],
            length: source_bytes.len() as u64,
            parse: Vec::new(),
            release: Default::default(),
        },
        files: file.into_iter().collect(),
        extracted: vec![Extraction {
            range,
            source_hash,
            parser_version: row["parser_version"].as_u64().unwrap() as u32,
            blob: hash,
        }],
    };
    let before = source_bytes.len();
    engine
        .excise(ExciseRequest {
            id: "remove-decoded-source-value".into(),
            scope: ExciseScope::Text(encoded),
            reason: None,
            dry_run: false,
        })
        .unwrap();
    let retained = engine
        .reads
        .reader()
        .unwrap()
        .retained_input(&input)
        .unwrap()
        .unwrap();
    assert_eq!(retained.length, before as u64);
    assert!(!engine.blobs.contains(&hashed_blobs::Hash(hash.0)));
    let result = engine
        .writer
        .sender()
        .unwrap()
        .retain_group(vec![pending], std::sync::Arc::clone(&engine.blobs))
        .unwrap();
    assert!(result[0].is_err());
    assert!(
        !engine.blobs.contains(&hashed_blobs::Hash(hash.0)),
        "stale decoded stage became public before source validation"
    );
    assert_blob(
        &engine,
        polylogue_model::BlobHash::of(&other_decoded),
        &other_decoded,
    );
}

#[cfg(feature = "faults")]
#[test]
fn pending_watched_groups_retry_exact_bytes_without_substituting_an_edited_provider() {
    use polylogue_store::fault::{self, Action};
    for edited in [false, true] {
        let world = ClaudeHome::new();
        let (encoded, decoded) = image_payload('A');
        let (next_encoded, _) = image_payload('B');
        let path = world.file("pending-pool.jsonl");
        let original = image_record(
            "pending-image",
            "pending-image",
            "Original pending image",
            &encoded,
        );
        std::fs::write(&path, &original).unwrap();
        let raw = hashed_blobs::Hash::of(original.as_bytes());
        let image = hashed_blobs::Hash::of(&decoded);
        let input;
        {
            let (mut engine, _lock) = engine(&world, 1 << 20);
            input = engine.location_id(&path);
            fault::arm("import.reserved", &input, Action::Fail, 0, 1);
            assert!(!engine.converge().unwrap().errors.is_empty());
            assert_eq!(
                world.count("SELECT count(*) AS n FROM a.extracted_blob WHERE state='writing'"),
                1
            );
            assert!(!engine.blobs.contains(&raw));
            assert!(!engine.blobs.contains(&image));
        }
        if edited {
            std::fs::write(
                &path,
                image_record(
                    "edited-image",
                    "edited-image",
                    "Edited pending image",
                    &next_encoded,
                ),
            )
            .unwrap();
        }
        let (mut engine, _lock) = engine(&world, 1 << 20);
        for _ in 0..2 {
            let pass = engine.converge().unwrap();
            assert!(pass.errors.is_empty(), "{edited}: {:?}", pass.errors);
        }
        assert_eq!(engine.blobs.contains(&raw), !edited);
        assert_eq!(engine.blobs.contains(&image), !edited);
        if edited {
            assert_eq!(world.count("SELECT count(*) AS n FROM a.revision"), 2);
            assert_eq!(
                world.count("SELECT count(*) AS n FROM a.extracted_blob WHERE state='writing'"),
                1
            );
        } else {
            assert_eq!(
                world.count("SELECT count(*) AS n FROM a.extracted_blob WHERE state='writing'"),
                0
            );
            assert_blob(&engine, polylogue_model::BlobHash(image.0), &decoded);
        }
    }
}
