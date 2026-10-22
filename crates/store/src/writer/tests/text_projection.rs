use super::*;
use rusqlite::StatementStatus;

#[test]
fn current_blocks_reuse_retained_text_identities() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open_write(&dir.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&db).unwrap();
    let tx = db.transaction().unwrap();
    let rows = (0..32)
        .map(|n| {
            row(
                n,
                text_message(
                    &format!("message-{n}"),
                    "session",
                    Role::User,
                    &format!("Synthetic unique body {n}"),
                    None,
                ),
            )
        })
        .collect();
    let b = batch("input", 0, 100, rows, &["session"]);
    assert_eq!(write_rows(&tx, &b, &b.parse[0], 1, 1, true).unwrap(), 32);
    let reads = tx
        .prepare_cached("SELECT id, indexed FROM text WHERE hash = ?1")
        .unwrap()
        .get_status(StatementStatus::Run);
    println!("text identity lookups for 32 distinct blocks: {reads}");
    assert_eq!(
        reads, 32,
        "current projection repeated historical text identity lookup"
    );
    assert_eq!(tx.query_row("SELECT count(*) FROM block b JOIN version_block v ON b.text_id=v.text_id AND b.body=v.body", [], |r|r.get::<_,i64>(0)).unwrap(),32);
}

fn snapshot(w: &World, start: u64, messages: Vec<Message>) -> WriteReport {
    let mut b = batch(
        "preferred",
        0,
        start,
        messages.into_iter().map(|m| row(start + 1, m)).collect(),
        &["session"],
    );
    b.parse[0].snapshot = Some(polylogue_model::Snapshot {
        start,
        end: start + 100,
        covered: vec![sid("session")],
        complete: true,
    });
    b.parse[0].prepare();
    w.writer.send(b).unwrap();
    w.writer.flush().unwrap()
}

fn search_count(w: &World, table: &str, term: &str) -> i64 {
    let db = crate::archive::open_read(&w.archive.index_db()).unwrap();
    db.query_row(
        &format!("SELECT count(*) FROM {table} WHERE {table} MATCH ?1"),
        [term],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn promotion_changes_search_membership_without_indexing_history_only_text() {
    let w = world();
    let mut preferred = text_message(
        "message",
        "session",
        Role::Assistant,
        "walrus",
        Some("end_turn"),
    );
    preferred.blocks.push(Block::ToolResult {
        call_id: None,
        outcome: ToolOutcome::Ok,
        output: Some("sharedterm".into()),
        data: None,
        attachments: vec![],
    });
    let mut alternative = preferred.clone();
    alternative.blocks[0] = Block::Text {
        text: "narwhal".into(),
    };
    if let Block::ToolResult { output, .. } = &mut alternative.blocks[1] {
        *output = Some("walrus".into());
    }
    assert!(snapshot(&w, 0, vec![preferred.clone()]).errors.is_empty());
    let mut losing = batch(
        "alternative",
        1,
        0,
        vec![row(1, alternative.clone())],
        &["session"],
    );
    losing.parse[0].prepare();
    w.writer.send(losing).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    assert_eq!(search_count(&w, "text_fts", "walrus"), 1);
    assert_eq!(search_count(&w, "output_fts", "walrus"), 0);
    assert_eq!(search_count(&w, "text_fts", "narwhal"), 0);
    assert!(snapshot(&w, 100, vec![]).errors.is_empty());
    assert_eq!(search_count(&w, "text_fts", "walrus"), 0);
    assert_eq!(search_count(&w, "output_fts", "walrus"), 1);
    assert_eq!(search_count(&w, "text_fts", "narwhal"), 1);
    assert_eq!(search_count(&w, "output_fts", "sharedterm"), 0);
    assert!(snapshot(&w, 200, vec![preferred]).errors.is_empty());
    assert_eq!(search_count(&w, "text_fts", "walrus"), 1);
    assert_eq!(search_count(&w, "output_fts", "walrus"), 0);
    assert_eq!(search_count(&w, "text_fts", "narwhal"), 0);
    assert_eq!(search_count(&w, "output_fts", "sharedterm"), 1);
}

#[test]
fn promotion_refuses_a_gap_in_retained_block_indices() {
    let w = world();
    let preferred = text_message(
        "message",
        "session",
        Role::Assistant,
        "walrus",
        Some("end_turn"),
    );
    let alternative = text_message(
        "message",
        "session",
        Role::Assistant,
        "narwhal",
        Some("end_turn"),
    );
    assert!(snapshot(&w, 0, vec![preferred]).errors.is_empty());
    w.writer
        .send(batch(
            "alternative",
            1,
            0,
            vec![row(1, alternative.clone())],
            &["session"],
        ))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let hash = blake3::hash(&serde_json::to_vec(&alternative).unwrap());
    let db = open_write(&w.archive.index_db(), 1).unwrap();
    assert_eq!(
        db.execute(
            "UPDATE version_block SET idx=7 WHERE version_hash=?1",
            [&hash.as_bytes()[..]]
        )
        .unwrap(),
        1
    );
    drop(db);
    let report = snapshot(&w, 100, vec![]);
    assert!(
        !report.errors.is_empty(),
        "misaligned history was projected successfully"
    );
    assert_eq!(texts(&w, "session"), ["walrus"]);
}
