use super::*;
use polylogue_store::predecessor::fixture::{self, VectorKey};
use polylogue_testkit::{ClaudeHome, claude_assistant, claude_user};
use rusqlite::OpenFlags;

fn options(format: &str) -> ImportOptions {
    ImportOptions {
        format: Some(format.into()),
        ..Default::default()
    }
}

fn read_only(path: &Path) -> rusqlite::Connection {
    rusqlite::Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap()
}

fn strings(conn: &rusqlite::Connection, sql: &str) -> Vec<Vec<String>> {
    let mut statement = conn.prepare(sql).unwrap();
    let columns = statement.column_count();
    statement
        .query_map([], |row| {
            (0..columns)
                .map(|i| {
                    Ok(match row.get_ref(i)? {
                        rusqlite::types::ValueRef::Null => "NULL".to_owned(),
                        rusqlite::types::ValueRef::Integer(n) => n.to_string(),
                        rusqlite::types::ValueRef::Text(t) => {
                            String::from_utf8_lossy(t).into_owned()
                        }
                        other => format!("{other:?}"),
                    })
                })
                .collect()
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn authored_records_carried() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("old").join("user.db");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    fixture::user_db(&path);
    let before = std::fs::read(&path).unwrap();
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);

    let report = engine
        .import_with_options(std::slice::from_ref(&path), &options(USER_DB))
        .unwrap()
        .predecessor
        .unwrap();
    let assertions = report.classes["assertions"];
    assert_eq!(
        (assertions.found, assertions.stored, assertions.refused),
        (12, 11, 1)
    );
    assert_eq!(report.classes["pass"].stored, 1);
    assert_eq!(report.classes["annotation_schemas"].kept, 1);
    assert_eq!(report.classes["result_set_members"].kept, 2);
    assert_eq!(report.refused.len(), 1);
    assert_eq!(report.refused[0].row, fixture::MARK);
    assert_eq!(
        report.refused[0].reason,
        "assertion kind mark has no counterpart here"
    );
    // Every row, the refused one included, stays readable in the rows input.
    let rows_input = report.rows_input.clone().unwrap();
    assert!(
        engine
            .reads
            .reader()
            .unwrap()
            .input_progress(&rows_input)
            .unwrap()
            .is_some()
    );

    let archive = read_only(&engine.archive.archive_db());
    let pass = format!("predecessor-user-db:{rows_input}");
    let stored = strings(
        &archive,
        "SELECT id, root, kind, key, author, via, machine FROM assertion ORDER BY id",
    );
    assert_eq!(stored.len(), 12);
    let row = |id: &str| stored.iter().find(|row| row[0] == id).unwrap().clone();
    assert_eq!(
        row(fixture::CANDIDATE),
        [
            fixture::CANDIDATE,
            fixture::CANDIDATE,
            "question",
            "duplicate:1",
            &format!("script:{}", fixture::DETECTOR),
            "import",
            "test-machine"
        ]
    );
    assert_eq!(row(fixture::PROMOTED)[1], fixture::ACCEPTED);
    assert_eq!(row(fixture::PROMOTED)[4], "unattributed");
    assert_eq!(
        row(fixture::ACCEPT)[2..5],
        ["judgment", "accepted", "unattributed"]
    );
    assert_eq!(row(fixture::REJECT)[2..4], ["judgment", "rejected"]);
    assert_eq!(row(fixture::BLOCKER)[2], "problem");
    assert_eq!(
        row(fixture::FINDING)[2..5],
        ["finding", "finding.followups", "script:run:followups-1"]
    );
    assert_eq!(
        row(fixture::NOTE)[2..5],
        [
            "note",
            "NULL",
            "agent:codex:00000000-0000-7000-8000-00000000c0de"
        ]
    );
    assert_eq!(
        row(&pass)[2..5],
        ["pass", "NULL", "script:polylogue-import"]
    );
    // The promotion revised the detector's request, so only it is current.
    let current = strings(
        &archive,
        &format!(
            "SELECT id, current FROM revision_state WHERE id IN ('{}', '{}') ORDER BY id",
            fixture::ACCEPTED,
            fixture::PROMOTED
        ),
    );
    assert_eq!(
        current,
        [[fixture::PROMOTED, "1"], [fixture::ACCEPTED, "0"]]
    );
    let origins = strings(
        &archive,
        &format!(
            "SELECT count(*) FROM link WHERE role = 'origin' AND ref = 'assertion:{pass}@{pass}'"
        ),
    );
    assert_eq!(origins, [["11"]]);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "user.db is only read"
    );

    // Importing the same store again changes nothing.
    let again = engine
        .import_with_options(std::slice::from_ref(&path), &options(USER_DB))
        .unwrap()
        .predecessor
        .unwrap();
    let assertions = again.classes["assertions"];
    assert_eq!(
        (assertions.stored, assertions.present, assertions.refused),
        (0, 11, 1)
    );
    assert_eq!(again.classes["pass"].present, 1);
    assert_eq!(
        strings(&archive, "SELECT count(*) FROM assertion"),
        [["12"]]
    );
}

#[test]
fn a_file_that_is_not_the_predecessors_user_db_is_refused_whole() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("user.db");
    rusqlite::Connection::open(&path)
        .unwrap()
        .execute_batch("CREATE TABLE notes (id TEXT)")
        .unwrap();
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    let error = engine
        .import_with_options(&[path], &options(USER_DB))
        .unwrap_err()
        .to_string();
    assert!(error.contains("not the predecessor's user.db"), "{error}");
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 0);
}

const MODEL: &str = "voyage-4";
const KEY: &str = "voyage/voyage-4/4/document/unit";

fn indexed_texts(w: &ClaudeHome, long: &str) {
    w.append(
        "s1.jsonl",
        &claude_user("u1", "s1", "the first text that was embedded"),
    );
    w.append(
        "s1.jsonl",
        &claude_assistant("a1", "s1", "a reply that was embedded too"),
    );
    w.append("s1.jsonl", &claude_user("u2", "s1", long));
}

fn addressed(text: &str) -> VectorKey {
    VectorKey::Address(address(MODEL, text))
}

fn embedded(engine: &Engine) -> Vec<Vec<String>> {
    let conn = read_only(&engine.archive.embeddings_db());
    let mut statement = conn
        .prepare("SELECT text_hash, model, chunk, dims, vector, embedded_at FROM embedded ORDER BY text_hash")
        .unwrap();
    statement
        .query_map([], |row| {
            Ok(vec![
                hex(&row.get::<_, Vec<u8>>(0)?),
                row.get(1)?,
                row.get::<_, i64>(2)?.to_string(),
                row.get::<_, i64>(3)?.to_string(),
                hex(&row.get::<_, Vec<u8>>(4)?),
                row.get::<_, i64>(5)?.to_string(),
            ])
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

fn bytes(values: &[f32]) -> String {
    hex(&values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>())
}

#[test]
fn predecessor_vectors_import_by_recipe_and_text_hash() {
    let w = ClaudeHome::new();
    let long = "x".repeat(polylogue_store::vectors::CHUNK_CHARS + 1);
    indexed_texts(&w, &long);
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);

    let store = w.dir.path().join("embeddings.db");
    fixture::vector_store(
        &store,
        MODEL,
        &[
            (
                addressed("the first text that was embedded"),
                vec![1.0, 0.0, 0.0, 0.0],
            ),
            (
                addressed("a reply that was embedded too"),
                vec![0.0, 1.0, 0.0, 0.0],
            ),
            (addressed(&long), vec![0.0, 0.0, 1.0, 0.0]),
            (
                addressed("a text this archive never held"),
                vec![0.0, 0.0, 0.0, 1.0],
            ),
        ],
        1_783_000_000_000,
    );
    let report = engine
        .import_with_options(std::slice::from_ref(&store), &options(VECTORS))
        .unwrap()
        .predecessor
        .unwrap();
    assert_eq!((report.vectors, report.vectors_matched), (4, 2));
    assert_eq!(report.classes["text key"].stored, 2);
    assert_eq!(
        report.unmatched,
        BTreeMap::from([
            (
                "its text is longer than one chunk here, so it is embedded in parts".to_owned(),
                1
            ),
            ("no text here has this request's address".to_owned(), 1),
        ])
    );
    let mut unmatched = report.unmatched_addresses.clone();
    unmatched.sort();
    let mut expected = vec![
        hex(&address(MODEL, &long)),
        hex(&address(MODEL, "a text this archive never held")),
    ];
    expected.sort();
    assert_eq!(unmatched, expected);

    // Each vector is stored under the raw text's hash, the full embedding
    // key, chunk 0, with its values and purchase time unchanged.
    let mut want = vec![
        vec![
            blake3::hash(b"the first text that was embedded")
                .to_hex()
                .to_string(),
            KEY.to_owned(),
            "0".into(),
            "4".into(),
            bytes(&[1.0, 0.0, 0.0, 0.0]),
            "1783000000000".into(),
        ],
        vec![
            blake3::hash(b"a reply that was embedded too")
                .to_hex()
                .to_string(),
            KEY.to_owned(),
            "0".into(),
            "4".into(),
            bytes(&[0.0, 1.0, 0.0, 0.0]),
            "1783000000000".into(),
        ],
    ];
    want.sort();
    assert_eq!(embedded(&engine), want);
}

#[test]
fn unmatched_predecessor_vectors_are_reported() {
    let w = ClaudeHome::new();
    indexed_texts(&w, "short");
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    engine.run_pass().unwrap();
    // A store keyed by message identity proves no text: every vector stays
    // in the source file and is counted.
    let store = w.dir.path().join("by-message.db");
    fixture::vector_store(
        &store,
        MODEL,
        &[
            (
                VectorKey::Message("claude-code-session:s1:u1".into()),
                vec![1.0, 0.0],
            ),
            (
                VectorKey::Message("codex-session:t1:msg-3".into()),
                vec![0.0, 1.0],
            ),
        ],
        0,
    );
    let report = engine
        .import_with_options(std::slice::from_ref(&store), &options(VECTORS))
        .unwrap()
        .predecessor
        .unwrap();
    assert_eq!((report.vectors, report.vectors_matched), (2, 0));
    assert_eq!(
        report.unmatched,
        BTreeMap::from([(
            "keyed by the predecessor's message identity; the embedded text cannot be proven"
                .to_owned(),
            2
        )])
    );
    assert!(embedded(&engine).is_empty());
}

#[test]
fn repeat_vector_import_adds_nothing() {
    let w = ClaudeHome::new();
    indexed_texts(&w, "short");
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    engine.run_pass().unwrap();
    let store = w.dir.path().join("embeddings.db");
    fixture::vector_store(
        &store,
        MODEL,
        &[(
            addressed("the first text that was embedded"),
            vec![1.0, 0.0, 0.0, 0.0],
        )],
        5,
    );
    let first = engine
        .import_with_options(std::slice::from_ref(&store), &options(VECTORS))
        .unwrap()
        .predecessor
        .unwrap();
    assert_eq!(first.classes["text key"].stored, 1);
    let held = embedded(&engine);
    let again = engine
        .import_with_options(std::slice::from_ref(&store), &options(VECTORS))
        .unwrap()
        .predecessor
        .unwrap();
    assert_eq!(again.classes["text key"].present, 1);
    assert_eq!(again.classes["text key"].stored, 0);
    assert_eq!(embedded(&engine), held);

    // A different vector bought for the same text is refused, never
    // replacing the one held.
    let other = w.dir.path().join("other.db");
    fixture::vector_store(
        &other,
        MODEL,
        &[(
            addressed("the first text that was embedded"),
            vec![0.0, 1.0, 0.0, 0.0],
        )],
        6,
    );
    let different = engine
        .import_with_options(std::slice::from_ref(&other), &options(VECTORS))
        .unwrap()
        .predecessor
        .unwrap();
    assert_eq!(different.classes["text key"].refused, 1);
    assert_eq!(embedded(&engine), held);
}

#[test]
fn imported_vectors_survive_index_rebuild() {
    let w = ClaudeHome::new();
    indexed_texts(&w, "short");
    let store = w.dir.path().join("embeddings.db");
    fixture::vector_store(
        &store,
        MODEL,
        &[(
            addressed("a reply that was embedded too"),
            vec![0.0, 1.0, 0.0, 0.0],
        )],
        5,
    );
    let held = {
        let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
        engine.run_pass().unwrap();
        engine
            .import_with_options(std::slice::from_ref(&store), &options(VECTORS))
            .unwrap();
        embedded(&engine)
    };
    assert_eq!(held.len(), 1);
    polylogue_testkit::delete_index(&w.root);
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(embedded(&engine), held);
    // The rebuilt index holds the text the vector is keyed by.
    let texts = engine
        .reads
        .reader()
        .unwrap()
        .prose_texts(0, 100, None)
        .unwrap();
    assert!(
        texts
            .iter()
            .any(|text| hex(&text.hash) == held[0][0]
                && text.text == "a reply that was embedded too")
    );
}

#[test]
fn the_address_is_the_predecessors_request_hash() {
    // Expected values computed with the predecessor's own recipe in Python
    // (hashlib and unicodedata), not with this code.
    assert_eq!(
        hex(&address("voyage-4", "hello")),
        "a4054bd18e1d989e3a07d9f552e4d68be5e409130cb6b66f540f1552ca7d4397"
    );
    // The predecessor addressed the NFC form; callers pass it normalized.
    let nfc: String = "Cafe\u{301}".nfc().collect();
    assert_eq!(
        hex(&address("voyage-4", &nfc)),
        "a6ad6a55c3e713108184d39d7d7a5161f8134ced709ac9f96e9bb60b29181687"
    );
}

fn spool_tree(root: &Path) {
    let write = |path: &str, text: &str| {
        let path = root.join(path);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    };
    write(
        "hooks/pending/2026-08-14/0001.json",
        "{\"event_id\":\"0001\",\"event_type\":\"PreToolUse\",\"observed_at_ms\":1786665985000,\"payload\":{\"session_id\":\"s1\",\"hook_event_name\":\"PreToolUse\",\"cwd\":\"/home/op\"}}\n",
    );
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

#[test]
fn predecessor_spool_tree_imports_every_hook_and_capture_file() {
    let w = ClaudeHome::new();
    let old = w.dir.path().join("old");
    spool_tree(&old);
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    // Unnamed, a spool is no declared export structure: nothing is stored.
    let unnamed = engine.import(&[old.join("hooks")]).unwrap();
    assert!(unnamed.imported.is_empty(), "{unnamed:?}");
    assert_eq!(unnamed.refused.len(), 4);

    let hooks = engine
        .import_with_options(&[old.join("hooks")], &options("hook-spool"))
        .unwrap();
    // Every spool file is parsed; the lock file beside them is kept as a
    // member of the named package, as any member no parser reads is.
    assert_eq!(hooks.imported.len(), 4, "{hooks:?}");
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
        w.count("SELECT COUNT(*) AS n FROM a.input WHERE format = 'hook-spool'"),
        3
    );
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM a.input WHERE format = 'browser-capture'"),
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
    assert_eq!(again.already_stored.len(), 4, "{again:?}");
}
