use super::*;
use polylogue_testkit::{ClaudeHome, claude_user, delete_index};

fn message_sources(w: &ClaudeHome) -> Vec<polylogue_model::Json> {
    let pool = ReadPool::open(&Archive::at(&w.root)).unwrap();
    pool.reader().unwrap().query("SELECT m.id,r.id AS input,s.chain,s.source_start,s.source_end,hex(s.version_hash) AS version FROM message_source s JOIN message m ON m.pk=s.message_pk JOIN input_ref r ON r.pk=s.input_pk ORDER BY m.id,r.id,s.chain,s.source_start",&[]).unwrap()
}

#[test]
fn immutable_imports_sharing_a_path_rebuild_every_source_and_never_become_watchers() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("same.jsonl");
    let (mut e, lock) = engine(&w, 64);
    for (id, body) in [
        ("one", "first stored import"),
        ("two", "second stored import"),
    ] {
        std::fs::write(&path, claude_user(id, "s", body)).unwrap();
        assert!(
            e.import_with_options(
                std::slice::from_ref(&path),
                &ImportOptions {
                    format: Some("claude-code".into()),
                    explain: false,
                    ..Default::default()
                }
            )
            .unwrap()
            .pass
            .errors
            .is_empty()
        );
    }
    let before = message_sources(&w);
    assert_eq!(before.len(), 2);
    assert_eq!(w.texts("s").len(), 2);
    drop(e);
    drop(lock);
    std::fs::remove_file(&path).unwrap();
    delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 64);
    let first = e.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{first:?}");
    assert_eq!(message_sources(&w), before);
    assert_eq!(w.texts("s").len(), 2);
    assert!(e.inputs.is_empty());
    let quiet = e.run_pass().unwrap();
    assert!(quiet.errors.is_empty());
    assert_eq!(quiet.bytes_read, 0);
    assert_eq!(quiet.inputs_changed, 0);
    let pool = ReadPool::open(&e.archive).unwrap();
    let r = pool.reader().unwrap();
    assert_eq!(
        r.query(
            "SELECT count(DISTINCT input) AS n FROM a.revision WHERE produced='imported'",
            &[]
        )
        .unwrap()[0]["n"],
        2
    );
    assert!(
        r.known_watched_inputs(&e.config.machine)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        r.query(
            "SELECT count(*) AS n FROM a.observation WHERE class='deleted'",
            &[]
        )
        .unwrap()[0]["n"],
        0
    );
}

#[test]
fn an_import_observation_does_not_replace_mutable_source_discovery_state() {
    let w = ClaudeHome::new();
    w.append(
        "live.jsonl",
        &claude_user("one", "s", "original native text"),
    );
    let path = w.home.join(".claude/projects/-proj/live.jsonl");
    let (mut e, lock) = engine(&w, 64);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert!(
        e.import(std::slice::from_ref(&path))
            .unwrap()
            .pass
            .errors
            .is_empty()
    );
    drop(e);
    drop(lock);
    std::fs::write(
        &path,
        claude_user("one", "s", "changed native text with new bytes"),
    )
    .unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(w.texts("s"), ["changed native text with new bytes"]);
    let pool = ReadPool::open(&e.archive).unwrap();
    let r = pool.reader().unwrap();
    let watched = r.known_watched_inputs(&e.config.machine).unwrap();
    assert_eq!(watched.len(), 1);
    let imported = r
        .query(
            "SELECT DISTINCT input FROM a.revision WHERE produced='imported'",
            &[],
        )
        .unwrap();
    assert_eq!(imported.len(), 1);
    assert_ne!(watched[0].input_id, imported[0]["input"]);
}

#[test]
fn refused_import_can_be_imported_under_its_declared_structure() {
    let w = ClaudeHome::new();
    let unknown = w.dir.path().join("renamed.json");
    let standard = w.dir.path().join("prod-grok-backend.json");
    let bytes = format!(
        r#"{{"metadata":"{}","conversations":[{{"conversation":{{"id":"recognized-later"}},"responses":[{{"responseId":"one","sender":"human","message":"stored later parse"}}]}}]}}"#,
        "synthetic ".repeat(12)
    );
    std::fs::write(&unknown, &bytes).unwrap();
    std::fs::write(&standard, &bytes).unwrap();
    let id = blake3::hash(bytes.as_bytes()).to_hex().to_string();
    let (mut e, lock) = engine(&w, 1024);
    let opaque = e.import(std::slice::from_ref(&unknown)).unwrap();
    assert_eq!(opaque.refused.len(), 1);
    assert!(opaque.imported.is_empty());
    let prior = register(&w.root);
    let parsed = e.import(std::slice::from_ref(&standard)).unwrap();
    assert!(parsed.pass.errors.is_empty(), "{:?}", parsed.pass.errors);
    assert_eq!(parsed.imported.as_slice(), std::slice::from_ref(&standard));
    assert_eq!(
        w.count("SELECT count(*) AS n FROM session WHERE id='grok:recognized-later'"),
        1
    );
    assert_eq!(
        e.reads
            .reader()
            .unwrap()
            .query(
                "SELECT format FROM a.input WHERE id=?1",
                &[polylogue_store::SqlValue::Text(id.clone())]
            )
            .unwrap()[0]["format"],
        "grok"
    );
    let after = register(&w.root);
    assert_eq!(
        after[..prior.len()],
        prior[..],
        "recognition must not change register rows already written"
    );
    // A later refused input leaves the declared parser path intact.
    e.import(std::slice::from_ref(&unknown)).unwrap();
    drop(e);
    drop(lock);
    std::fs::remove_file(unknown).unwrap();
    std::fs::remove_file(standard).unwrap();
    delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 1024);
    let report = e.run_pass().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM session WHERE id='grok:recognized-later'"),
        1
    );
    assert_eq!(
        e.reads
            .reader()
            .unwrap()
            .query(
                "SELECT format FROM a.input WHERE id=?1",
                &[polylogue_store::SqlValue::Text(id)]
            )
            .unwrap()[0]["format"],
        "grok"
    );
}

#[test]
fn unknown_bytes_are_refused_without_archive_rows() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("opaque.dat");
    std::fs::write(&path, b"unparsed binary\0\xff").unwrap();
    let (mut e, _lock) = engine(&w, 1024);
    let report = e.import(std::slice::from_ref(&path)).unwrap();
    assert_eq!(report.refused.len(), 1);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 0);
}

#[test]
fn a_conflicting_new_parsed_origin_is_refused_without_changing_stored_rows() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("prod-grok-backend.json");
    std::fs::write(&path, br#"{"conversations":[]}"#).unwrap();
    let (mut e, _lock) = engine(&w, 1024);
    assert!(
        e.import(std::slice::from_ref(&path))
            .unwrap()
            .pass
            .errors
            .is_empty()
    );
    let before = w.count("SELECT count(*) AS n FROM a.observation");
    let mut captured = Captured::stream(&e.archive, std::fs::File::open(&path).unwrap()).unwrap();
    captured.member = Some(polylogue_store::ImportMember {
        version: 1,
        import_id: "competing".into(),
        authorship: None,
        member: "conversations.json".into(),
        format: None,
    });
    let mut report = ImportReport::default();
    e.import_captured(
        &path,
        polylogue_sources::origin::by_name("chatgpt").unwrap(),
        captured,
        None,
        &mut report,
    )
    .unwrap();
    assert!(
        report
            .pass
            .errors
            .iter()
            .any(|error| error.contains("already parsed as another origin"))
    );
    assert_eq!(w.count("SELECT count(*) AS n FROM a.observation"), before);
}

/// Every revision, extent and observation row, in write order.
fn register(root: &std::path::Path) -> Vec<String> {
    let conn = rusqlite::Connection::open(root.join("archive.db")).unwrap();
    let mut rows = Vec::new();
    for sql in [
        "SELECT input, number, produced, chain, length FROM revision ORDER BY commit_seq, input, number",
        "SELECT input, chain, offset, length, hex(blob) FROM extent ORDER BY input, chain, offset",
        "SELECT id, input, class, revision, path FROM observation ORDER BY rowid",
    ] {
        let mut statement = conn.prepare(sql).unwrap();
        let columns = statement.column_count();
        let mut query = statement.query([]).unwrap();
        while let Some(row) = query.next().unwrap() {
            let values: Vec<String> = (0..columns)
                .map(|i| format!("{:?}", row.get_ref(i).unwrap()))
                .collect();
            rows.push(values.join(" "));
        }
    }
    rows
}

#[test]
fn ai_studio_chunks_read_and_rebuild_from_stored_inputs_without_empty_sessions() {
    fn read(w: &ClaudeHome) -> serde_json::Value {
        let pool = ReadPool::open(&Archive::at(&w.root)).unwrap();
        let reader = pool.reader().unwrap();
        let id = polylogue_model::SessionId::parse("ai-studio:Chat").unwrap();
        serde_json::json!({
            "messages": reader.messages(&id, false).unwrap(),
            "sessions": reader.query("SELECT id FROM session WHERE stub=0 ORDER BY id", &[]).unwrap(),
            "sources": reader.query("SELECT m.id,s.source_start,s.source_end FROM message_source s JOIN message m ON m.pk=s.message_pk ORDER BY m.id", &[]).unwrap(),
        })
    }
    let w = ClaudeHome::new();
    let path = w.dir.path().join("Chat.json");
    let empty = w.dir.path().join("Empty.json");
    let bytes = serde_json::to_vec(&serde_json::json!({
        "runSettings": {"model": "models/test"},
        "systemInstruction": {"text": "Be brief"},
        "chunkedPrompt": {"chunks": [
            {"role":"user", "id":"ask", "text":"question"},
            {"role":"model", "isThought":true, "text":"thinking", "tokenCount":7},
            {"role":"model", "id":"reply", "text":"answer", "tokenCount":3,
             "inlineImage":{"mimeType":"image/png", "data":"aGk="}},
            {"role":"unknown", "text":"unreadable"}
        ]}
    }))
    .unwrap();
    std::fs::write(&path, &bytes).unwrap();
    std::fs::write(&empty, br#"{"runSettings":{},"systemInstruction":{"text":"header"},"chunkedPrompt":{"chunks":[]}}"#).unwrap();
    let (mut e, lock) = engine(&w, 64);
    let imported = e.import(&[path.clone(), empty.clone()]).unwrap();
    assert!(imported.pass.errors.is_empty(), "{imported:?}");
    let before = read(&w);
    assert_eq!(
        before["sessions"],
        serde_json::json!([{"id":"ai-studio:Chat"}])
    );
    let messages = before["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 4);
    assert!(messages[2].to_string().contains("thinking"));
    assert!(messages[3].to_string().contains("answer"));
    for row in before["sources"].as_array().unwrap() {
        if row["id"].as_str().unwrap().ends_with("#system:0:") {
            continue;
        }
        let start = row["source_start"].as_u64().unwrap() as usize;
        let end = row["source_end"].as_u64().unwrap() as usize;
        let chunk: serde_json::Value = serde_json::from_slice(&bytes[start..end]).unwrap();
        assert!(matches!(chunk["role"].as_str(), Some("user" | "model")));
    }
    drop(e);
    drop(lock);
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(empty).unwrap();
    delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 64);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(read(&w), before);
}
