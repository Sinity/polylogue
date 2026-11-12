use super::*;
use polylogue_model::{BlobHash, Reference};
use rusqlite::{Connection, params, types::Value};

fn rows(conn: &Connection, sql: &str) -> Vec<Vec<Value>> {
    let mut statement = conn.prepare(sql).unwrap();
    let columns = statement.column_count();
    statement
        .query_map([], |row| (0..columns).map(|index| row.get(index)).collect())
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn a_deliberate_index_rebuild_keeps_authored_rows_and_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let archive_conn = open_archive_writer(&archive).unwrap();
    let stored = BlobHash::of(b"synthetic stored embedded text");
    let removed = BlobHash::of(b"synthetic removed text");
    use polylogue_model::assertion::{AssertionFact, AssertionKind, AssertionLink, LinkRole};
    crate::assertions::apply(
        &archive_conn,
        &[AssertionFact {
            id: "authored".into(),
            root: "authored".into(),
            kind: AssertionKind::Note,
            key: None,
            value: Some(serde_json::json!("exact authored value")),
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Import,
            machine: "synthetic".into(),
            created_at: Some(polylogue_model::Timestamp::from_unix_ms(1).unwrap()),
            valid_until: None,
            links: vec![AssertionLink {
                role: LinkRole::About,
                reference: Reference::parse("message:claude-code:stable").unwrap(),
                seen_hash: None,
                quote: None,
            }],
        }],
        1,
    )
    .unwrap();
    let vectors = open_write(&archive.embeddings_db(), cache::SETUP).unwrap();
    for (hash, value) in [(stored, 1.25f32), (removed, 2.5f32)] {
        vectors.execute("INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at) VALUES (?1,'synthetic/model/1/document/unit',0,1,?2,5)", params![&hash.0[..], value.to_le_bytes().as_slice()]).unwrap();
    }
    drop(vectors);
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    index.pragma_update(None, "user_version", 29).unwrap();
    index
        .execute("INSERT INTO meta VALUES ('old-index-marker','old')", [])
        .unwrap();
    drop(index);
    let queries = [
        "SELECT * FROM assertion ORDER BY id",
        "SELECT * FROM link ORDER BY record,role,ref",
        "SELECT * FROM revision_state ORDER BY id",
    ];
    let before: Vec<_> = queries.iter().map(|sql| rows(&archive_conn, sql)).collect();
    let vectors = open_read(&archive.embeddings_db()).unwrap();
    let paid = rows(&vectors, "SELECT * FROM embedded ORDER BY id");
    assert_eq!(paid.len(), 2);
    drop(vectors);
    drop(archive_conn);
    drop(lock);

    // The test explicitly requests a fresh derived index.
    for suffix in ["", "-wal", "-shm"] {
        let path = archive.root.join(format!("index.db{suffix}"));
        if path.exists() {
            std::fs::remove_file(path).unwrap();
        }
    }
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_read(&archive.index_db()).unwrap();
    assert_eq!(schema::version(&index).unwrap(), schema::INDEX_VERSION);
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM meta WHERE key='old-index-marker'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    let archive_conn = open_read(&archive.archive_db()).unwrap();
    let after: Vec<_> = queries.iter().map(|sql| rows(&archive_conn, sql)).collect();
    assert_eq!(after, before, "authored values stay verbatim");
    let vectors = open_read(&archive.embeddings_db()).unwrap();
    assert_eq!(
        rows(&vectors, "SELECT * FROM embedded ORDER BY id"),
        paid,
        "existing text hash, model, vector bytes and row identity stay unchanged"
    );
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let assertion = reader
        .assertion(&polylogue_model::RevisionReference::exact(
            "authored".into(),
            "authored".into(),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(assertion.fact.links.len(), 1);
    assert_eq!(assertion.fact.links[0].role, LinkRole::About);
    assert_eq!(
        assertion.fact.links[0].reference,
        Reference::Message(polylogue_model::MessageId::parse("claude-code:stable").unwrap())
    );
    assert_eq!(assertion.fact.text(), Some("exact authored value"));
}
