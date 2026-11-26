use super::*;
use std::io::Read;

fn hook(ordinal: usize) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(&serde_json::json!({
        "at_ns": (1_792_400_000_123_456_789_u64 + ordinal as u64).to_string(),
        "event": "PreToolUse", "provider": "claude-code", "harness_pid": 4242,
        "session_id": "synthetic-hooks",
        "payload": {"session_id":"synthetic-hooks", "tool_name":"Bash",
                    "tool_input":{"command":format!("printf synthetic-{ordinal}")}}
    }))
    .unwrap();
    bytes.push(b'\n');
    bytes
}

fn tar(path: &Path, members: &[(String, Vec<u8>)]) {
    let mut archive = tar::Builder::new(std::fs::File::create(path).unwrap());
    for (name, bytes) in members {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o600);
        header.set_cksum();
        archive
            .append_data(&mut header, name, bytes.as_slice())
            .unwrap();
    }
    archive.finish().unwrap();
}

fn membership(home: &polylogue_testkit::ClaudeHome) -> Vec<(String, i64, Vec<u8>, String)> {
    let db = rusqlite::Connection::open(home.root.join("archive.db")).unwrap();
    db.prepare("SELECT c.container_input,c.member_position,c.path_bytes,o.input FROM observation_container c JOIN observation o ON o.id=c.observation_id ORDER BY c.member_position,o.input")
        .unwrap().query_map([], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?)))
        .unwrap().collect::<rusqlite::Result<_>>().unwrap()
}

#[test]
fn hook_carrier_batches_keep_bytes_duplicate_paths_members_and_repeat() {
    let home = polylogue_testkit::ClaudeHome::new();
    let path = home.dir.path().join("hooks.tar");
    let mut members: Vec<_> = (0..259)
        .map(|i| (format!("polylogue/hooks/{i:04}.ndjson"), hook(i)))
        .collect();
    // The same input at distinct paths and distinct inputs at one path must
    // retain separate member provenance without concurrent chain readers.
    members[64].1 = members[63].1.clone();
    members[129].0 = members[128].0.clone();
    members[1].0 = "polylogue/hooks/two words-żółw.ndjson".into();
    tar(&path, &members);
    let original = std::fs::read(&path).unwrap();
    let parent = blake3::hash(&original).to_hex().to_string();
    let expected: Vec<_> = members
        .iter()
        .enumerate()
        .map(|(i, (name, bytes))| {
            (
                parent.clone(),
                i as i64,
                name.as_bytes().to_vec(),
                blake3::hash(bytes).to_hex().to_string(),
            )
        })
        .collect();
    let unique: BTreeSet<_> = members
        .iter()
        .map(|(_, bytes)| blake3::hash(bytes).to_hex().to_string())
        .collect();
    let (mut engine, _lock) = engine(&home, 64);
    let options = ImportOptions {
        origin: Some("hook-spool".into()),
        ..Default::default()
    };
    let first = engine
        .import_with_options(std::slice::from_ref(&path), &options)
        .unwrap();
    assert!(first.pass.errors.is_empty(), "{:?}", first.pass.errors);
    assert_eq!(membership(&home), expected);
    assert_eq!(
        home.count("SELECT count(*) AS n FROM event WHERE kind='hook_pre_tool_use'"),
        unique.len() as i64
    );
    for bytes in std::iter::once(&original).chain(members.iter().map(|(_, bytes)| bytes)) {
        let id = blake3::hash(bytes).to_hex().to_string();
        let chain = engine
            .reads
            .reader()
            .unwrap()
            .stored_chain(&id, 1, bytes.len() as u64)
            .unwrap();
        let mut restored = Vec::new();
        chain
            .reader(Arc::clone(&engine.blobs))
            .read_to_end(&mut restored)
            .unwrap();
        assert_eq!(restored, *bytes);
    }
    let counter =
        home.count("SELECT CAST(value AS INTEGER) AS n FROM a.meta WHERE key='write_counter'");
    let repeat = engine
        .import_with_options(std::slice::from_ref(&path), &options)
        .unwrap();
    assert!(repeat.pass.errors.is_empty(), "{:?}", repeat.pass.errors);
    assert_eq!(membership(&home), expected);
    assert_eq!(
        home.count("SELECT count(*) AS n FROM event WHERE kind='hook_pre_tool_use'"),
        unique.len() as i64
    );
    assert_eq!(
        home.count("SELECT CAST(value AS INTEGER) AS n FROM a.meta WHERE key='write_counter'"),
        counter
    );
}

fn spool_tree(root: &Path) {
    let write = |path: &str, text: &str| {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    write(
        "hooks/carriers/codex/2026-09-19/4242.ndjson",
        "{\"event_id\":\"0002\",\"event_type\":\"SessionStart\",\"observed_at_ms\":1789815097000,\"payload\":{\"session_id\":\"t1\",\"hook_event_name\":\"SessionStart\"}}\n",
    );
    write(
        "hooks/claude-code-s1.jsonl",
        "{\"event_type\":\"Stop\",\"session_id\":\"s1\",\"timestamp\":\"2026-08-14T00:00:00Z\"}\n",
    );
    write("hooks/.carrier-drain.lock", "");
    write(
        "browser-capture/gemini/files.json",
        &serde_json::json!({"polylogue_capture_kind":"browser_llm_session", "schema_version":1,
            "provenance":{"captured_at":"2026-08-01T00:00:00Z"},
            "session":{"provider":"gemini", "provider_session_id":"files",
                "turns":[{"provider_turn_id":"m1", "role":"user", "text":"a captured turn"}]}})
        .to_string(),
    );
}

/// Hook lines in the older shapes (dated carrier files with `event_id`,
/// per-session files with `event_type`) and a `browser_llm_session` v1
/// capture are record shapes of their parsers, read when named.
#[test]
fn older_hook_lines_and_browser_captures_import_when_named() {
    let options = |format: &str| ImportOptions {
        origin: Some(format.into()),
        ..Default::default()
    };
    let w = polylogue_testkit::ClaudeHome::new();
    let old = w.dir.path().join("old");
    spool_tree(&old);
    let (mut engine, _lock) = engine(&w, 64);
    // Unnamed, a spool is no declared export structure: nothing is stored.
    let unnamed = engine.import(&[old.join("hooks")]).unwrap();
    assert!(unnamed.imported.is_empty(), "{unnamed:?}");
    assert_eq!(unnamed.refused.len(), 3);

    let hooks = engine
        .import_with_options(&[old.join("hooks")], &options("hook-spool"))
        .unwrap();
    // Every spool file is parsed; the lock file beside them is kept as a
    // member of the named package, as any member no parser reads is.
    assert_eq!(hooks.imported.len(), 3, "{hooks:?}");
    assert!(hooks.refused.is_empty(), "{hooks:?}");
    let lock = hooks
        .members
        .iter()
        .find(|member| member.path.ends_with(".carrier-drain.lock"))
        .unwrap();
    assert_eq!(lock.class, "asset");
    assert!(hooks.pass.errors.is_empty(), "{:?}", hooks.pass.errors);
    let captures = engine
        .import_with_options(&[old.join("browser-capture")], &options("browser-capture"))
        .unwrap();
    assert_eq!(captures.imported.len(), 1, "{captures:?}");
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM a.input WHERE origin = 'hook-spool'"),
        2
    );
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM a.input WHERE origin = 'browser-capture'"),
        1
    );
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM session WHERE id = 'gemini-app:files'"),
        1
    );

    // Importing the same trees again stores nothing new.
    let again = engine
        .import_with_options(&[old.join("hooks")], &options("hook-spool"))
        .unwrap();
    assert!(again.pass.errors.is_empty(), "{:?}", again.pass.errors);
    assert!(again.imported.is_empty(), "{again:?}");
    assert_eq!(again.already_stored.len(), 3, "{again:?}");
}
