use super::*;
use crate::encoding::PreparedMessage;

fn message(n: usize) -> Message {
    serde_json::from_value(serde_json::json!({
        "id":format!("synthetic:message:{n}"),"session":"synthetic:session",
        "role":"assistant","model":"synthetic","prompt_kind":"model",
        "blocks":[{"type":"text","text":format!("Synthetic retained text {n}")}],
        "usage":{},"stop_reason":"end_turn"
    }))
    .unwrap()
}

#[test]
fn version_row_ids_survive_text_interning_and_repeated_retention() {
    let mut conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.execute(
        "INSERT INTO text(id,hash,body) VALUES (1000,?1,?2)",
        params![
            blake3::hash(b"seed").as_bytes().as_slice(),
            crate::encoding::pack_text("seed").unwrap()
        ],
    )
    .unwrap();
    let tx = conn.transaction().unwrap();
    let m = message(1);
    let prepared = PreparedMessage::new(&m).unwrap();
    let pk = retain(&tx, &prepared, &m, &HashMap::new(), false).unwrap();
    assert_eq!(pk, 1);
    assert_eq!(tx.last_insert_rowid(), 1001);
    assert_eq!(
        retain(&tx, &prepared, &m, &HashMap::new(), false).unwrap(),
        pk
    );
    assert_eq!(load(&tx, &prepared.hash).unwrap(), (pk, m));
    assert_eq!(
        tx.query_row("SELECT version_pk FROM version_block", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        pk
    );
    assert!(
        tx.prepare("PRAGMA foreign_key_check")
            .unwrap()
            .query([])
            .unwrap()
            .next()
            .unwrap()
            .is_none()
    );
    prune(&tx).unwrap();
    assert_eq!(
        tx.query_row("SELECT count(*) FROM version_block", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn reverse_hashes_keep_contiguous_version_blocks_and_exact_historical_values() {
    let mut conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    crate::schema::create_index(&conn).unwrap();
    let mut prepared = (0..128)
        .map(|n| {
            let m = message(n);
            (PreparedMessage::new(&m).unwrap(), m)
        })
        .collect::<Vec<_>>();
    prepared.sort_by_key(|a| std::cmp::Reverse(a.0.hash));
    let tx = conn.transaction().unwrap();
    for (n, (p, m)) in prepared.iter().enumerate() {
        assert_eq!(
            retain(&tx, p, m, &HashMap::new(), true).unwrap(),
            n as i64 + 1
        );
        assert_eq!(load(&tx, &p.hash).unwrap(), (n as i64 + 1, m.clone()));
    }
    let plan: String = tx
        .query_row(
            "EXPLAIN QUERY PLAN SELECT body FROM version_block WHERE version_pk=?1 ORDER BY idx",
            [1],
            |r| r.get(3),
        )
        .unwrap();
    assert!(
        plan.contains("SEARCH version_block USING PRIMARY KEY"),
        "{plan}"
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM message_version", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        128
    );
    assert_eq!(
        tx.query_row(
            "SELECT count(DISTINCT version_pk) FROM version_block",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        128
    );
}
