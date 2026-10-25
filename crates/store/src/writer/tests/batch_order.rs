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
    let batch_reader = crate::row_batches::BatchReader::open(&w.archive.root).unwrap();
    let batches: Vec<String> = batch_reader
        .batches(None, 100)
        .unwrap()
        .into_iter()
        .map(|batch| batch.key)
        .collect();
    assert_eq!(batches, actual);
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

#[test]
fn independent_archives_cannot_assign_the_same_observation_identity() {
    let mut ids = Vec::new();
    let mut coordinates = Vec::new();
    for seen_at in [900, 1] {
        let w = world();
        let mut db =
            crate::archive::open_write(&w.archive.archive_db(), crate::archive::cache::SETUP)
                .unwrap();
        db.execute("INSERT INTO input VALUES ('input','synthetic-text')", [])
            .unwrap();
        let tx = db.transaction().unwrap();
        let seq = bump_counter(&tx).unwrap();
        let mut b = batch("input", 0, 0, vec![], &[]);
        b.observation.as_mut().unwrap().seen_at_ms = seen_at;
        insert_observation(&tx, &b, "deleted", None, seq).unwrap();
        tx.commit().unwrap();
        ids.push(
            db.query_row("SELECT id FROM observation", [], |r| r.get::<_, String>(0))
                .unwrap(),
        );
        coordinates.push(
            db.query_row(
                "SELECT commit_seq,rowid,input,path FROM observation",
                [],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                    ))
                },
            )
            .unwrap(),
        );
    }
    assert_eq!(coordinates[0], coordinates[1]);
    assert_ne!(ids[0], ids[1]);
}
