use super::*;

#[test]
fn same_transaction_observations_replay_in_insertion_order_despite_nonmonotonic_times() {
    let w = world();
    let mut db =
        crate::archive::open_write(&w.archive.archive_db(), crate::archive::cache::SETUP).unwrap();
    db.execute("INSERT INTO input VALUES ('input','synthetic-text')", [])
        .unwrap();
    let tx = db.transaction().unwrap();
    let seq = bump_counter(&tx).unwrap();
    for at in [900, 1, 500, 2] {
        let mut b = batch("input", 0, 0, vec![], &[]);
        let observation = b.observation.as_mut().unwrap();
        observation.seen_at_ms = at;
        observation.path = format!("location-{at}");
        insert_observation(&tx, &b, "deleted", None, seq).unwrap();
    }
    tx.commit().unwrap();
    let actual: Vec<String> = db
        .prepare("SELECT id FROM observation ORDER BY rowid")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let published = crate::sinex_rows::Published::open(&w.archive.root).unwrap();
    let inventory = published.inventory().unwrap();
    assert_eq!(
        (1..=inventory.len())
            .map(|p| inventory.unit(p).unwrap().key)
            .collect::<Vec<_>>(),
        actual
    );
    assert_eq!(
        db.query_row(
            "SELECT path FROM observation ORDER BY rowid DESC LIMIT 1",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "location-2"
    );
}
