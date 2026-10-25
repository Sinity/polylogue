use super::*;

#[test]
fn shared_document_without_record_ranges_blanks_only_complete_long_values() {
    use polylogue_model::export::Document;
    let text = format!(
        "private ☺ complete quoted value \"{}\"",
        "synthetic ".repeat(7000)
    );
    let kept = format!("keep this larger value containing [{text}] unchanged");
    let exporter = ClaudeHome::new();
    exporter.append(
        "removed.jsonl",
        &(user("u1", "removed", &text) + &user("u2", "removed", "ok")),
    );
    exporter.append("kept.jsonl", &user("k1", "kept", &kept));
    let (mut source, _lock) = exporter.engine(64 << 20);
    assert!(source.converge().unwrap().errors.is_empty());
    let original = Document::new(vec![
        exporter.export("claude-code:removed"),
        exporter.export("claude-code:kept"),
    ]);
    let file = exporter.dir.path().join("shared.json");
    fs::write(&file, original.to_json().unwrap().replace('☺', "\\u263a")).unwrap();
    let original_bytes = fs::read(&file).unwrap();
    let needle = serde_json::to_string(&text)
        .unwrap()
        .replace('☺', "\\u263a");
    let ranges: Vec<_> = std::str::from_utf8(&original_bytes)
        .unwrap()
        .match_indices(&needle)
        .map(|(at, _)| (at + 1, at + needle.len() - 1))
        .collect();
    assert!(!ranges.is_empty());
    let mut expected_bytes = original_bytes.clone();
    for &(start, end) in &ranges {
        expected_bytes[start..end].fill(b' ');
    }
    let w = ClaudeHome::new();
    let (mut engine, lock) = w.engine(64 << 20);
    engine.import(std::slice::from_ref(&file)).unwrap();
    let index = polylogue_store::archive::open_write(
        &Archive::at(&w.root).index_db(),
        polylogue_store::archive::cache::SETUP,
    )
    .unwrap();
    // These records have no known end offset, so excision takes its fallback path.
    index.execute("UPDATE message_source SET source_end=source_start WHERE message_pk IN (SELECT m.pk FROM message m JOIN session s ON s.pk=m.session_pk WHERE s.id='claude-code:removed')", []).unwrap();
    drop(index);
    let archive = Archive::at(&w.root);
    let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
    {
        let archive_conn = rusqlite::Connection::open(archive.archive_db()).unwrap();
        let hash: Vec<u8> = archive_conn
            .query_row("SELECT DISTINCT blob FROM extent", [], |row| row.get(0))
            .unwrap();
        // Inject malformed stored metadata only in this synthetic archive.
        archive_conn.execute_batch("PRAGMA foreign_keys=OFF").unwrap();
        archive_conn
            .execute("UPDATE extent SET blob=x'01'", [])
            .unwrap();
        let failed = engine.excise(ExciseRequest {
            id: "malformed-source-hash".into(),
            scope: ExciseScope::parse("session:claude-code:removed").unwrap(),
            reason: None,
            dry_run: false,
        });
        archive_conn
            .execute("UPDATE extent SET blob=?1", [hash])
            .unwrap();
        archive_conn.execute_batch("PRAGMA foreign_keys=ON").unwrap();
        assert!(
            failed
                .unwrap_err()
                .to_string()
                .contains("invalid source blob hash")
        );
        assert_eq!(
            archive_conn
                .query_row("SELECT count(*) FROM excision", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
    for dry in [true, false] {
        let done = excise(&mut engine, "session:claude-code:removed", dry);
        assert_eq!(done.inputs_kept.len(), 1);
        assert!(done.texts_left >= 1, "the short 'ok' value remains");
        assert_eq!(
            done.inputs_overwritten.values().sum::<u64>(),
            ranges.len() as u64
        );
        assert_eq!(w.texts("kept"), std::slice::from_ref(&kept));
        let archive_conn = rusqlite::Connection::open(archive.archive_db()).unwrap();
        let hash: Vec<u8> = archive_conn
            .query_row("SELECT DISTINCT blob FROM extent", [], |r| r.get(0))
            .unwrap();
        let hash = polylogue_model::BlobHash(hash.try_into().unwrap());
        let raw = polylogue_store::blobs::read(&archive_conn, &store, &hash).unwrap();
        assert_eq!(
            raw,
            if dry {
                &original_bytes
            } else {
                &expected_bytes
            }
            .as_slice()
        );
        if !dry {
            let source_hash = blake3::hash(&original_bytes).to_hex().to_string();
            let expected_spans: Vec<_> = ranges
                .iter()
                .map(|&(start, end)| serde_json::json!([source_hash, start, end - start]))
                .collect();
            let expected = serde_json::to_string(&expected_spans).unwrap();
            let recorded:Vec<String>=archive_conn.prepare("SELECT spans_json FROM observation WHERE class='excised' AND spans_json IS NOT NULL").unwrap().query_map([],|row|row.get(0)).unwrap().collect::<rusqlite::Result<_>>().unwrap();
            assert!(!recorded.is_empty());
            assert!(
                recorded.iter().all(|value| value == &expected),
                "{recorded:?}"
            );
        }
        let document = Document::from_json(std::str::from_utf8(&raw).unwrap()).unwrap();
        let removed_text = document.sessions[0].messages[0].text();
        if dry {
            assert_eq!(removed_text, text);
        } else {
            assert!(removed_text.trim().is_empty());
        }
        assert_eq!(document.sessions[0].messages[1].text(), "ok");
        assert_eq!(document.sessions[1].messages[0].text(), kept);
    }
    drop(engine);
    drop(lock);
    fs::remove_file(file).unwrap();
    delete_index(&w.root);
    let (mut restored, _lock) = w.engine(64 << 20);
    assert!(restored.converge().unwrap().errors.is_empty());
    assert_eq!(w.texts("kept"), [kept]);
    assert!(w.texts("removed").is_empty());
}
