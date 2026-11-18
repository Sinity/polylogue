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
    db.prepare("SELECT c.container_input,c.member_index,c.path_bytes,o.input FROM observation_container c JOIN observation o ON o.id=c.observation_id ORDER BY c.member_index,o.input")
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
        format: Some("hook-spool".into()),
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
