use super::*;
use polylogue_model::{BlobHash, Target};
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
fn canonical_message_hash_generation_preserves_authored_scope_exclusions_and_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let mut canonical = open_archive_writer(&archive).unwrap();
    let seen = BlobHash::of(b"synthetic previously seen text");
    let retained = BlobHash::of(b"synthetic retained embedded text");
    let removed = BlobHash::of(b"synthetic removed text");
    let removed_blob = BlobHash::of(b"synthetic removed attachment");
    canonical.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('authored','authored','note','exact authored value','operator','synthetic',1,1)", []).unwrap();
    canonical.execute("INSERT INTO note_target(note_id,ref,seen_hash,anchor_input,anchor_chain,anchor_start,anchor_end,anchor_parser) VALUES ('authored','message:claude-code:stable',?1,'original-input',7,11,29,3)", [&seen.0[..]]).unwrap();
    canonical.execute("INSERT INTO variant(id,target,seen_hash,kind,coverage,body,author,machine,created_at) VALUES ('variant','message:claude-code:stable',?1,'summary','whole','exact transformed text','operator','synthetic',2)", [&seen.0[..]]).unwrap();
    let refusal = serde_json::json!({
        "sessions":["claude-code:blocked"],
        "locations":[{"machine":"synthetic","path":"/synthetic/removed","device":12,"inode":34}],
        "revisions":[BlobHash::of(b"synthetic removed source").to_string()],
        "blobs":[removed_blob.to_string()]
    })
    .to_string();
    canonical.execute("INSERT INTO excision(id,scope,refused_json,message_ids_json,created_at,commit_seq) VALUES ('session-removal','session:claude-code:blocked',?1,'[\"claude-code:blocked-message\"]',3,2)", [&refusal]).unwrap();
    canonical
        .execute(
            "INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('text-removal',?1,4,3)",
            [format!("text:{removed}")],
        )
        .unwrap();
    canonical
        .execute(
            "INSERT INTO vector_deletion_policy VALUES ('text-removal',1,?1)",
            [serde_json::to_string(&vec![removed.to_string()]).unwrap()],
        )
        .unwrap();
    let vectors = open_write(&archive.embeddings_db(), cache::SETUP).unwrap();
    for (hash, value) in [(retained, 1.25f32), (removed, 2.5f32)] {
        vectors.execute("INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at) VALUES (?1,'synthetic/model/1/document/unit',0,1,?2,5)", params![&hash.0[..], value.to_le_bytes().as_slice()]).unwrap();
    }
    drop(vectors);
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    crate::excise::recovery::acknowledge(&canonical, &index).unwrap();
    crate::excise::vector_cleanup::recover(&archive, &mut canonical).unwrap();
    assert!(
        !crate::excise::recovery::needs_rebuild(&canonical, &index).unwrap(),
        "the generation change, rather than a changed removal rule, must retire this index"
    );
    index.pragma_update(None, "user_version", 29).unwrap();
    index
        .execute(
            "INSERT INTO meta VALUES ('legacy-hash-generation','old')",
            [],
        )
        .unwrap();
    drop(index);
    let queries = [
        "SELECT * FROM note ORDER BY id",
        "SELECT * FROM note_target ORDER BY note_id,ref",
        "SELECT * FROM variant ORDER BY id",
        "SELECT * FROM excision ORDER BY id",
        "SELECT * FROM vector_deletion_policy ORDER BY excision_id",
    ];
    let before: Vec<_> = queries.iter().map(|sql| rows(&canonical, sql)).collect();
    let vectors = open_read(&archive.embeddings_db()).unwrap();
    let paid = rows(&vectors, "SELECT * FROM embedded ORDER BY id");
    assert_eq!(
        paid.len(),
        1,
        "the existing removed-text policy was applied"
    );
    let receipts = rows(
        &vectors,
        "SELECT * FROM vector_policy_receipt ORDER BY excision_id",
    );
    drop(vectors);
    drop(canonical);
    drop(lock);

    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_read(&archive.index_db()).unwrap();
    assert_eq!(schema::version(&index).unwrap(), schema::INDEX_VERSION);
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM meta WHERE key='legacy-hash-generation'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    let canonical = open_read(&archive.archive_db()).unwrap();
    let after: Vec<_> = queries.iter().map(|sql| rows(&canonical, sql)).collect();
    assert_eq!(
        after, before,
        "authored values, nonnull seen_hash/anchors and every standing policy stay verbatim"
    );
    let refused = crate::excise::Refused::load(&canonical).unwrap();
    let matches = refused
        .matching(
            &["claude-code:blocked".into(), "claude-code:allowed".into()],
            &[
                "claude-code:blocked-message".into(),
                "claude-code:allowed-message".into(),
            ],
        )
        .unwrap();
    assert_eq!(
        matches.sessions.into_iter().collect::<Vec<_>>(),
        ["claude-code:blocked"]
    );
    assert_eq!(
        matches.messages.into_iter().collect::<Vec<_>>(),
        ["claude-code:blocked-message"]
    );
    let vectors = open_read(&archive.embeddings_db()).unwrap();
    assert_eq!(
        rows(&vectors, "SELECT * FROM embedded ORDER BY id"),
        paid,
        "existing text hash, model, vector bytes and row identity stay unchanged"
    );
    assert_eq!(
        rows(
            &vectors,
            "SELECT * FROM vector_policy_receipt ORDER BY excision_id"
        ),
        receipts
    );
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    let note = reader.note("authored").unwrap().unwrap();
    assert_eq!(
        note.targets,
        [Target::Message(
            polylogue_model::MessageId::parse("claude-code:stable").unwrap()
        )]
    );
    assert_eq!(note.value.as_deref(), Some("exact authored value"));
    assert_eq!(
        reader.refused_paths().unwrap(),
        [("synthetic".into(), "/synthetic/removed".into())]
    );
}
