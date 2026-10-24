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
    batch.parse[0].accounting = accounting(1);
    if prepared {
        batch.parse[0].prepare();
    }
    batch
}

#[test]
fn prepared_and_unprepared_documents_keep_values_and_source_versions_without_chat_rows() {
    let w = world();
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
    assert_eq!(reader.document_versions(&original.id).unwrap().len(), 1);
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
    let versions = reader.document_versions(&original.id).unwrap();
    assert_eq!(versions.len(), 2);
    assert_eq!(versions.iter().filter(|v| v.preferred).count(), 1);
    assert!(versions.iter().any(|v| v.document == original));
}

#[test]
fn equal_text_with_distinct_document_ids_keeps_both_records_and_shared_text() {
    let w = world();
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
fn document_hash_is_canonical_and_does_not_change_message_hash_domain() {
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
    let mut canonical = serde_json::to_value(&message).unwrap();
    canonical.sort_all_objects();
    assert_eq!(
        prepared.hash,
        *blake3::hash(&serde_json::to_vec(&canonical).unwrap()).as_bytes()
    );
}
