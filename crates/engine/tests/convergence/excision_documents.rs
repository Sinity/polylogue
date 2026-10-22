use super::*;

#[test]
fn shared_document_without_record_ranges_blanks_only_complete_long_values() {
    use polylogue_model::export::Document;
    let text = format!(
        "private ☺ complete quoted value \"{}\"",
        "synthetic ".repeat(7000)
    );
    let kept = format!("keep this larger value containing [{text}] unchanged");
    let donor = ClaudeHome::new();
    donor.append(
        "removed.jsonl",
        &(user("u1", "removed", &text) + &user("u2", "removed", "ok")),
    );
    donor.append("kept.jsonl", &user("k1", "kept", &kept));
    let (mut source, _lock) = donor.engine(64 << 20);
    assert!(source.converge().unwrap().errors.is_empty());
    let original = Document::new(vec![
        donor.export("claude-code:removed"),
        donor.export("claude-code:kept"),
    ]);
    let file = donor.dir.path().join("shared.json");
    fs::write(&file, original.to_json().unwrap().replace('☺', "\\u263a")).unwrap();
    let w = ClaudeHome::new();
    let (mut engine, lock) = w.engine(64 << 20);
    engine.import(std::slice::from_ref(&file)).unwrap();
    let index = polylogue_store::archive::open_write(
        &Archive::at(&w.root).index_db(),
        polylogue_store::archive::cache::SETUP,
    )
    .unwrap();
    // The supported legacy fallback: these records have no known end offset.
    index.execute("UPDATE message_source SET source_end=source_start WHERE session_pk=(SELECT pk FROM session WHERE id='claude-code:removed')", []).unwrap();
    drop(index);
    let archive = Archive::at(&w.root);
    let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
    for dry in [true, false] {
        let done = excise(&mut engine, "session:claude-code:removed", dry);
        assert_eq!(done.inputs_kept.len(), 1);
        assert!(done.texts_left >= 1, "the short 'ok' value remains");
        assert!(done.inputs_overwritten.values().sum::<u64>() > 0);
        assert_eq!(w.texts("kept"), std::slice::from_ref(&kept));
        let canonical = rusqlite::Connection::open(archive.archive_db()).unwrap();
        let hash: Vec<u8> = canonical
            .query_row("SELECT DISTINCT blob FROM extent", [], |r| r.get(0))
            .unwrap();
        let hash = polylogue_model::BlobHash(hash.try_into().unwrap());
        let raw = polylogue_store::blobs::read(&canonical, &store, &hash).unwrap();
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
