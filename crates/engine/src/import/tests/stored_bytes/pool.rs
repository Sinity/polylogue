use super::*;

#[test]
fn refused_watched_bytes_install_no_file() {
    for image in [false, true] {
        let home = ClaudeHome::new();
        let path = home.file("pool-stored.jsonl");
        let decoded = vec![b'A'; polylogue_store::blobs::INLINE_LIMIT + 1024];
        let bytes = if image {
            let encoded = "QUFB".repeat(decoded.len() / 3 + 1);
            format!(
                "{}\n",
                serde_json::json!({"type":"user","uuid":"pool-image","sessionId":"pool-stored","message":{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":encoded}}]}})
            )
        } else {
            claude_user("pool-user", "pool-stored", &"x".repeat(decoded.len()))
        };
        std::fs::write(&path, &bytes).unwrap();
        let raw = hashed_blobs::Hash::of(bytes.as_bytes());
        let (mut engine, _lock) = engine(&home, 1 << 20);
        let input = engine.location_id(&path);
        let conn = rusqlite::Connection::open(home.root.join("archive.db")).unwrap();
        conn.execute_batch(&format!("CREATE TRIGGER reject_pool BEFORE INSERT ON input WHEN NEW.id='{input}' BEGIN SELECT RAISE(ABORT,'synthetic pool first write rejection'); END;")).unwrap();
        let pass = engine.run_pass().unwrap();
        assert!(!pass.errors.is_empty());
        assert!(
            !engine.blobs.contains(&raw),
            "rejected pool bytes have no archive row"
        );
        assert_eq!(home.count("SELECT count(*) AS n FROM a.blob"), 0);
        conn.execute_batch("DROP TRIGGER reject_pool").unwrap();
        let pass = engine.run_pass().unwrap();
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
fn decoded_bytes_survive_a_failed_index_write_and_a_rebuild_without_the_provider_file() {
    let home = ClaudeHome::new();
    let (encoded, decoded) = image_payload('A');
    let path = home.file("decoded-stored.jsonl");
    std::fs::write(
        &path,
        image_record(
            "image-user",
            "decoded-stored",
            "synthetic decoded content",
            &encoded,
        ),
    )
    .unwrap();
    let hash = polylogue_model::BlobHash::of(&decoded);
    {
        let (mut engine, _lock) = engine(&home, 1 << 20);
        let index = rusqlite::Connection::open(home.root.join("index.db")).unwrap();
        index.execute_batch("CREATE TRIGGER reject_decoded_message BEFORE INSERT ON message BEGIN SELECT RAISE(ABORT,'synthetic derived write refusal'); END;").unwrap();
        let pass = engine.run_pass().unwrap();
        assert!(!pass.errors.is_empty());
        assert_eq!(home.count("SELECT count(*) AS n FROM message"), 0);
        assert_blob(&engine, hash, &decoded);
    }
    std::fs::remove_file(path).unwrap();
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, _lock) = engine(&home, 1 << 20);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(home.count("SELECT count(*) AS n FROM message"), 1);
    assert_blob(&engine, hash, &decoded);
}

#[test]
fn input_excision_without_an_index_keeps_bytes_another_input_decodes() {
    let home = ClaudeHome::new();
    let (encoded, decoded) = image_payload('A');
    let first = home.file("first-image.jsonl");
    let second = home.file("second-image.jsonl");
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
        let (mut engine, _lock) = engine(&home, 1 << 20);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        input = engine.location_id(&first);
    }
    std::fs::remove_file(first).unwrap();
    std::fs::remove_file(second).unwrap();
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, _lock) = engine(&home, 1 << 20);
    engine
        .excise(ExciseRequest {
            id: "remove-first-decoded-owner".into(),
            target: ExciseTarget::Input(input),
            reason: None,
            dry_run: false,
        })
        .unwrap();
    assert_blob(&engine, hash, &decoded);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(home.count("SELECT count(*) AS n FROM message"), 1);
    assert_blob(&engine, hash, &decoded);
}

#[cfg(feature = "faults")]
#[test]
fn bytes_left_writing_are_read_again_and_an_edited_file_starts_a_new_chain() {
    use polylogue_store::fault::{self, Action};
    for edited in [false, true] {
        let home = ClaudeHome::new();
        let (encoded, decoded) = image_payload('A');
        let (next_encoded, _) = image_payload('B');
        let path = home.file("pending-pool.jsonl");
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
            let (mut engine, _lock) = engine(&home, 1 << 20);
            input = engine.location_id(&path);
            fault::arm("import.install", &input, Action::Fail, 0, 1);
            assert!(!engine.run_pass().unwrap().errors.is_empty());
            assert_eq!(
                home.count("SELECT count(*) AS n FROM a.blob WHERE state='writing'"),
                1
            );
            assert!(!engine.blobs.contains(&raw));
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
        let (mut engine, _lock) = engine(&home, 1 << 20);
        for _ in 0..2 {
            let pass = engine.run_pass().unwrap();
            assert!(pass.errors.is_empty(), "{edited}: {:?}", pass.errors);
        }
        assert_eq!(engine.blobs.contains(&raw), !edited);
        assert!(engine.blobs.contains(&image));
        if edited {
            assert_eq!(home.count("SELECT count(*) AS n FROM a.revision"), 2);
            assert_eq!(
                home.count("SELECT count(*) AS n FROM a.blob WHERE state='writing'"),
                1
            );
        } else {
            assert_eq!(
                home.count("SELECT count(*) AS n FROM a.blob WHERE state='writing'"),
                0
            );
            assert_blob(&engine, polylogue_model::BlobHash(image.0), &decoded);
        }
    }
}
