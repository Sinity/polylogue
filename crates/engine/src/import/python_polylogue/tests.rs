use super::*;
use polylogue_store::python_polylogue::fixture::{self, VectorKey};
use polylogue_testkit::{Home, claude_assistant, claude_user};
use rusqlite::OpenFlags;

fn options(format: &str) -> ImportOptions {
    ImportOptions {
        origin: Some(format.into()),
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

const MODEL: &str = "voyage-4";
const KEY: &str = "voyage/voyage-4/4/document/unit";

fn indexed_texts(w: &Home, long: &str) {
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
fn python_vectors_import_by_recipe_and_text_hash() {
    for prepared in [false, true] {
        let w = Home::new();
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
        if prepared {
            let conn = rusqlite::Connection::open(&store).unwrap();
            conn.execute_batch("PRAGMA application_id=0x504c5650; PRAGMA user_version=1; ALTER TABLE message_embeddings_meta ADD COLUMN request_text TEXT;").unwrap();
            for text in [
                "the first text that was embedded",
                "a reply that was embedded too",
                &long,
                "a text this archive never held",
            ] {
                conn.execute("UPDATE message_embeddings_meta SET request_text=?1 WHERE embedding_input_hash=?2", rusqlite::params![text,address(MODEL,text).as_slice()]).unwrap();
            }
        }
        let report = engine
            .import_with_options(std::slice::from_ref(&store), &options(VECTORS))
            .unwrap()
            .python_import
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
}
#[test]
fn vector_addresses_match_all_current_raw_unicode_spellings_and_repeat() {
    for prepared in [false, true] {
        let w = Home::new();
        let nfc = "Café";
        let decomposed = "Cafe\u{301}";
        w.append("unicode.jsonl", &claude_user("composed", "unicode", nfc));
        w.append(
            "unicode.jsonl",
            &claude_user("decomposed", "unicode", decomposed),
        );
        let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        let store = w.dir.path().join("unicode-vectors.db");
        fixture::vector_store(
            &store,
            MODEL,
            &[(addressed(nfc), vec![1.0, 0.0, 0.0, 0.0])],
            1_783_000_000_000,
        );
        if prepared {
            let conn = rusqlite::Connection::open(&store).unwrap();
            conn.execute_batch("PRAGMA application_id=0x504c5650; PRAGMA user_version=1; ALTER TABLE message_embeddings_meta ADD COLUMN request_text TEXT;").unwrap();
            conn.execute("UPDATE message_embeddings_meta SET request_text=?1", [nfc])
                .unwrap();
        }
        let source = std::fs::read(&store).unwrap();
        let report = engine
            .import_with_options(std::slice::from_ref(&store), &options(VECTORS))
            .unwrap()
            .python_import
            .unwrap();
        assert_eq!((report.vectors, report.vectors_matched), (1, 1));
        assert_eq!(report.classes["text key"].stored, 2);
        let mut hashes = embedded(&engine)
            .into_iter()
            .map(|row| row[0].clone())
            .collect::<Vec<_>>();
        hashes.sort();
        let mut expected = [nfc, decomposed]
            .map(|text| blake3::hash(text.as_bytes()).to_hex().to_string())
            .to_vec();
        expected.sort();
        assert_eq!(
            hashes, expected,
            "one normalized request can have two raw text keys"
        );
        let before = embedded(&engine);
        let repeat = engine
            .import_with_options(std::slice::from_ref(&store), &options(VECTORS))
            .unwrap()
            .python_import
            .unwrap();
        assert_eq!(
            (
                repeat.classes["text key"].stored,
                repeat.classes["text key"].present
            ),
            (0, 2)
        );
        assert_eq!(embedded(&engine), before);
        assert_eq!(std::fs::read(&store).unwrap(), source);
    }
}

#[test]
fn unmatched_python_vectors_are_reported() {
    let w = Home::new();
    indexed_texts(&w, "short");
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    engine.run_pass().unwrap();
    // A store keyed by message identity checks no text: every vector stays
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
        .python_import
        .unwrap();
    assert_eq!((report.vectors, report.vectors_matched), (2, 0));
    assert_eq!(
        report.unmatched,
        BTreeMap::from([(
            "keyed by Python Polylogue's message identity; the embedded text cannot be checked"
                .to_owned(),
            2
        )])
    );
    assert!(embedded(&engine).is_empty());
}

#[test]
fn repeat_vector_import_adds_nothing() {
    let w = Home::new();
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
        .python_import
        .unwrap();
    assert_eq!(first.classes["text key"].stored, 1);
    let held = embedded(&engine);
    let again = engine
        .import_with_options(std::slice::from_ref(&store), &options(VECTORS))
        .unwrap()
        .python_import
        .unwrap();
    assert_eq!(again.classes["text key"].present, 1);
    assert_eq!(again.classes["text key"].stored, 0);
    assert_eq!(embedded(&engine), held);

    // Equal purchase times cannot order conflicting paid values.
    // Restore refuses the conflict rather than choosing an arrival order.
    let other = w.dir.path().join("other.db");
    fixture::vector_store(
        &other,
        MODEL,
        &[(
            addressed("the first text that was embedded"),
            vec![0.0, 1.0, 0.0, 0.0],
        )],
        5,
    );
    let different = engine
        .import_with_options(std::slice::from_ref(&other), &options(VECTORS))
        .unwrap()
        .python_import
        .unwrap();
    assert_eq!(different.classes["text key"].refused, 1);
    assert_eq!(embedded(&engine), held);
}

#[test]
fn imported_vectors_survive_index_rebuild() {
    let w = Home::new();
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
fn address_is_python_polylogue_request_hash() {
    // Expected values computed with Python Polylogue's own recipe in Python
    // (hashlib and unicodedata), not with this code.
    assert_eq!(
        hex(&address("voyage-4", "hello")),
        "a4054bd18e1d989e3a07d9f552e4d68be5e409130cb6b66f540f1552ca7d4397"
    );
    // Python Polylogue addressed the NFC form; callers pass it normalized.
    let nfc: String = "Cafe\u{301}".nfc().collect();
    assert_eq!(
        hex(&address("voyage-4", &nfc)),
        "a6ad6a55c3e713108184d39d7d7a5161f8134ced709ac9f96e9bb60b29181687"
    );
}

#[test]
fn newer_paid_vectors_win_in_both_orders_and_repeat_changes_nothing() {
    for newest_first in [false, true] {
        let home = Home::new();
        indexed_texts(&home, "short");
        let (mut engine, _lock) = crate::import::tests::engine(&home, 64);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        let old = home.dir.path().join("old-paid.db");
        let new = home.dir.path().join("new-paid.db");
        let text = "the first text that was embedded";
        fixture::vector_store(&old, MODEL, &[(addressed(text), vec![1., 0., 0., 0.])], 10);
        fixture::vector_store(&new, MODEL, &[(addressed(text), vec![0., 1., 0., 0.])], 20);
        let source_bytes = [std::fs::read(&old).unwrap(), std::fs::read(&new).unwrap()];
        let paths = if newest_first {
            [&new, &old]
        } else {
            [&old, &new]
        };
        for path in paths {
            let report = engine
                .import_with_options(std::slice::from_ref(path), &options(VECTORS))
                .unwrap()
                .python_import
                .unwrap();
            assert!(report.refused.is_empty(), "{report:?}");
            assert_eq!(report.vectors_matched, 1);
        }
        assert_eq!(
            embedded(&engine),
            vec![vec![
                blake3::hash(text.as_bytes()).to_hex().to_string(),
                KEY.into(),
                "0".into(),
                "4".into(),
                bytes(&[0., 1., 0., 0.]),
                "20".into(),
            ]]
        );
        let target = read_only(&engine.archive.embeddings_db());
        let before = strings(
            &target,
            "SELECT id,text_hash,model,chunk,dims,vector,embedded_at FROM embedded ORDER BY id",
        );
        let revision = strings(
            &target,
            "SELECT model,revision FROM embedding_revision ORDER BY model",
        );
        let sequence = strings(
            &target,
            "SELECT name,seq FROM sqlite_sequence ORDER BY name",
        );
        for path in [&new, &old] {
            let report = engine
                .import_with_options(std::slice::from_ref(path), &options(VECTORS))
                .unwrap()
                .python_import
                .unwrap();
            let count = report.classes["text key"];
            assert_eq!((count.stored, count.present, count.refused), (0, 1, 0));
        }
        assert_eq!(
            strings(
                &target,
                "SELECT id,text_hash,model,chunk,dims,vector,embedded_at FROM embedded ORDER BY id"
            ),
            before
        );
        assert_eq!(
            strings(
                &target,
                "SELECT model,revision FROM embedding_revision ORDER BY model"
            ),
            revision
        );
        assert_eq!(
            strings(
                &target,
                "SELECT name,seq FROM sqlite_sequence ORDER BY name"
            ),
            sequence
        );
        assert_eq!(std::fs::read(&old).unwrap(), source_bytes[0]);
        assert_eq!(std::fs::read(&new).unwrap(), source_bytes[1]);
    }
}

#[test]
fn unknown_vector_purchase_time_is_explicitly_unmatched_and_never_given_an_order() {
    let home = Home::new();
    indexed_texts(&home, "short");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let path = home.dir.path().join("unknown-paid.db");
    fixture::vector_store(
        &path,
        MODEL,
        &[(
            addressed("the first text that was embedded"),
            vec![1., 0., 0., 0.],
        )],
        10,
    );
    let source = rusqlite::Connection::open(&path).unwrap();
    source
        .execute("UPDATE message_embeddings_meta SET embedded_at_ms=NULL", [])
        .unwrap();
    drop(source);
    let before = std::fs::read(&path).unwrap();
    let report = engine
        .import_with_options(std::slice::from_ref(&path), &options(VECTORS))
        .unwrap()
        .python_import
        .unwrap();
    assert_eq!(
        report.unmatched.get("its purchase time is not recorded"),
        Some(&1)
    );
    assert!(embedded(&engine).is_empty());
    assert_eq!(std::fs::read(&path).unwrap(), before);
}
