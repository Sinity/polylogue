use super::*;
use polylogue_model::{Document, DocumentAuthorship, DocumentId, DocumentPurpose, DocumentRow};
use serde_json::json;

fn document(id: &str, body: &str, time: i64) -> Document {
    let mut doc = Document::new(
        DocumentId::parse(&format!("claude-ai:{id}")).unwrap(),
        DocumentPurpose::Context,
    );
    doc.blocks = vec![Block::Text { text: body.into() }];
    doc.title = Some(String::new());
    doc.source_time = Timestamp::from_unix_ms(time);
    doc.metadata =
        json!({"source_kind":"project_document","supplied":false,"empty":"","opaque":null});
    doc
}

fn doc_batch(input: &str, doc: Document, prepared: bool) -> Batch {
    let mut batch = batch(input, 2, 0, Vec::new(), &[]);
    batch.input.format = "claude-ai".into();
    batch.parse[0].output.documents.push(DocumentRow {
        offset: 11,
        end: 83,
        document: doc,
    });
    batch.parse[0].counts = counts(1);
    if prepared {
        batch.parse[0].prepare();
    }
    batch
}

#[test]
fn prepared_and_unprepared_documents_keep_values_and_source_versions_without_chat_rows() {
    let w = test_archive();
    let original = document("project:a:doc:native", "Exact document λ🙂\nsecond line", 1);
    w.writer
        .send(doc_batch("a", original.clone(), false))
        .unwrap();
    w.writer
        .send(doc_batch("b", original.clone(), true))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader.document(&original.id).unwrap(),
        Some(original.clone())
    );
    let selected = reader.select_document(&original.id).unwrap().unwrap();
    let sources = selected.sources().unwrap();
    assert_eq!(sources.len(), 2);
    assert!(
        sources
            .iter()
            .all(|s| s.start == 11 && s.end == 83 && !s.locations.is_empty())
    );
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM session", &[])
            .unwrap()[0]["n"],
        0
    );
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM message", &[])
            .unwrap()[0]["n"],
        0
    );
    assert_eq!(original.authorship, DocumentAuthorship::Unknown);
    drop(selected);
    drop(reader);
    drop(pool);
    let changed = document("project:a:doc:native", "Changed supplied text", 2);
    w.writer
        .send(doc_batch("c", changed.clone(), true))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(reader.document(&original.id).unwrap(), Some(changed));
    let sources = reader
        .select_document(&original.id)
        .unwrap()
        .unwrap()
        .sources()
        .unwrap();
    assert_eq!(sources.iter().filter(|s| s.shown).count(), 1);
    assert_eq!(
        sources
            .iter()
            .map(|s| s.version.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        2
    );
}

#[test]
fn equal_text_with_distinct_document_ids_keeps_both_records_and_shared_text() {
    let w = test_archive();
    let a = document("project:a:doc:a", "Shared supplied text", 0);
    let b = document("project:a:doc:b", "Shared supplied text", 0);
    w.writer.send(doc_batch("one", a.clone(), true)).unwrap();
    w.writer.send(doc_batch("two", b.clone(), true)).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(reader.document(&a.id).unwrap(), Some(a));
    assert_eq!(reader.document(&b.id).unwrap(), Some(b));
    let rows = reader
        .query(
            "SELECT count(*) AS documents,count(DISTINCT text_id) AS texts FROM document_block",
            &[],
        )
        .unwrap();
    assert_eq!(rows[0]["documents"], 2);
    assert_eq!(rows[0]["texts"], 1);
}

#[test]
fn document_hash_uses_sorted_json_and_does_not_change_message_hash_domain() {
    let mut a = document("project:a:doc:a", "Source text", 0);
    let mut first = serde_json::Map::new();
    first.insert("z".into(), json!({"b":2,"a":1}));
    first.insert("a".into(), json!([{"z":false,"a":null}]));
    let mut second = serde_json::Map::new();
    second.insert("a".into(), json!([{"a":null,"z":false}]));
    second.insert("z".into(), json!({"a":1,"b":2}));
    a.metadata = Json::Object(first);
    let mut b = a.clone();
    b.metadata = Json::Object(second);
    let pa = PreparedDocument::new(&a).unwrap();
    let pb = PreparedDocument::new(&b).unwrap();
    assert_eq!(pa.hash, pb.hash);
    let message = text_message(
        "same",
        "session",
        Role::Assistant,
        "Source text",
        Some("end_turn"),
    );
    let prepared = PreparedMessage::new(&message).unwrap();
    let mut archive_conn = serde_json::to_value(&message).unwrap();
    archive_conn.sort_all_objects();
    assert_eq!(
        prepared.hash,
        *blake3::hash(&serde_json::to_vec(&archive_conn).unwrap()).as_bytes()
    );
}

#[test]
fn a_started_document_read_keeps_one_generation_after_rewrite_and_pool_close() {
    use std::ops::ControlFlow;
    let w = test_archive();
    let mut original = document("native", "old first block", 1);
    original.blocks.push(Block::Text {
        text: "old second block".into(),
    });
    w.writer
        .send(doc_batch("a", original.clone(), true))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let selected = reader.select_document(&original.id).unwrap().unwrap();
    pool.close();
    let mut blocks = Vec::new();
    selected
        .visit_blocks(|block| {
            blocks.push(block);
            if blocks.len() == 1 {
                let mut changed = original.clone();
                changed.source_time = Timestamp::from_unix_ms(2);
                changed.blocks = vec![Block::Text {
                    text: "new generation".into(),
                }];
                w.writer.send(doc_batch("b", changed, true)).unwrap();
                assert!(w.writer.flush().unwrap().errors.is_empty());
            }
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(blocks, original.blocks);
    assert_eq!(selected.document().unwrap(), original);
    drop(selected);
    drop(reader);
    let fresh = ReadPool::open(&w.archive).unwrap();
    assert_eq!(
        fresh
            .reader()
            .unwrap()
            .document(&original.id)
            .unwrap()
            .unwrap()
            .blocks,
        [Block::Text {
            text: "new generation".into()
        }]
    );
}

#[test]
fn shared_message_document_text_keeps_vectors_after_both_sources_disappear() {
    let w = test_archive();
    let body = "Shared message and document text λ🙂";
    let message = text_message("m", "s", Role::User, body, None);
    w.writer
        .send(batch(
            "message-input",
            2,
            0,
            vec![MessageRow {
                offset: 0,
                end: 90,
                message,
            }],
            &["s"],
        ))
        .unwrap();
    let doc = document("kept", body, 1);
    w.writer
        .send(doc_batch("document-input", doc.clone(), true))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let hash: Vec<u8> = index
        .query_row(
            "SELECT t.hash FROM document_block b JOIN text t ON t.id=b.text_id",
            [],
            |r| r.get(0),
        )
        .unwrap();
    drop(index);
    let vectors =
        crate::archive::open_write(&w.archive.embeddings_db(), crate::archive::cache::SETUP)
            .unwrap();
    vectors.execute("INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at) VALUES (?1,'synthetic/model/1/document/unit',0,1,?2,1)",rusqlite::params![hash,1.25f32.to_le_bytes().to_vec()]).unwrap();
    drop(vectors);
    for input in ["message-input", "document-input"] {
        let mut gone = batch(input, 2, 0, vec![], &[]);
        gone.change = Change::Disappeared;
        w.writer.send(gone).unwrap();
    }
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let vectors = crate::archive::open_read(&w.archive.embeddings_db()).unwrap();
    assert_eq!(
        vectors
            .query_row("SELECT count(*) FROM embedded", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    let pool = ReadPool::open(&w.archive).unwrap();
    assert!(pool.reader().unwrap().document(&doc.id).unwrap().is_some());
}

#[test]
fn a_same_chain_document_cannot_override_a_newer_source_time() {
    let w = test_archive();
    w.writer
        .send(doc_batch(
            "a",
            document("ordered", "newer source", 2000),
            true,
        ))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let mut older = doc_batch("a", document("ordered", "older source", 1000), true);
    older.change = Change::Reparse { chain: 1 };
    older.parse[0].reset = false;
    w.writer.send(older).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let reads = ReadPool::open(&w.archive).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .document(&DocumentId::parse("claude-ai:ordered").unwrap())
            .unwrap()
            .unwrap()
            .blocks,
        [Block::Text {
            text: "newer source".into()
        }]
    );
}

#[test]
fn a_later_record_of_the_same_chain_can_supply_a_newer_document_time() {
    let w = test_archive();
    w.writer
        .send(doc_batch("a", document("ordered", "older", 1000), true))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let mut newer = doc_batch("a", document("ordered", "newer", 2000), true);
    newer.change = Change::Reparse { chain: 1 };
    newer.parse[0].reset = false;
    newer.parse[0].output.documents[0].offset = 20;
    newer.parse[0].output.documents[0].end = 30;
    w.writer.send(newer).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let reads = ReadPool::open(&w.archive).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .document(&DocumentId::parse("claude-ai:ordered").unwrap())
            .unwrap()
            .unwrap()
            .blocks,
        [Block::Text {
            text: "newer".into()
        }]
    );
}
