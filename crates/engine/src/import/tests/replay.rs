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
            e.import(std::slice::from_ref(&path))
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
    let first = e.converge().unwrap();
    assert!(first.errors.is_empty(), "{first:?}");
    assert_eq!(message_sources(&w), before);
    assert_eq!(w.texts("s").len(), 2);
    assert!(e.inputs.is_empty());
    let quiet = e.converge().unwrap();
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
    assert!(e.converge().unwrap().errors.is_empty());
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
    let pass = e.converge().unwrap();
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
fn recognized_import_after_opaque_capture_rebuilds_from_the_recognized_observation() {
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
    assert_eq!(opaque.imported.as_slice(), std::slice::from_ref(&unknown));
    let prior = register(&w.root);
    let parsed = e.import(std::slice::from_ref(&standard)).unwrap();
    assert!(parsed.pass.errors.is_empty(), "{:?}", parsed.pass.errors);
    assert_eq!(
        parsed.already_stored.as_slice(),
        std::slice::from_ref(&standard)
    );
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
        "export-file"
    );
    let after = register(&w.root);
    assert_eq!(
        after[..prior.len()],
        prior[..],
        "recognition must not change register rows already written"
    );
    // A later opaque acquisition must not discard the captured parser path.
    e.import(std::slice::from_ref(&unknown)).unwrap();
    drop(e);
    drop(lock);
    std::fs::remove_file(unknown).unwrap();
    std::fs::remove_file(standard).unwrap();
    delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 1024);
    let report = e.converge().unwrap();
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
        "export-file"
    );
}

#[test]
fn explicit_kept_only_acquisitions_replay_without_errors() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("opaque.dat");
    std::fs::write(&path, b"synthetic unparsed binary\0\xff").unwrap();
    let (mut e, lock) = engine(&w, 1024);
    let report = e.import(std::slice::from_ref(&path)).unwrap();
    assert_eq!(report.unrecognized.as_slice(), std::slice::from_ref(&path));
    assert_eq!(report.imported.as_slice(), std::slice::from_ref(&path));
    let again = e.import(std::slice::from_ref(&path)).unwrap();
    assert_eq!(again.unrecognized.as_slice(), std::slice::from_ref(&path));
    assert_eq!(again.already_stored.as_slice(), std::slice::from_ref(&path));
    drop(e);
    drop(lock);
    std::fs::remove_file(path).unwrap();
    delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 1024);
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty());
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
        acquisition: "competing".into(),
        member: "conversations.json".into(),
        format: None,
        container: None,
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
