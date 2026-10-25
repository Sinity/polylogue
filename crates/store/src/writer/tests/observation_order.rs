use super::*;

/// Records `b`'s observation as a disappearance.
fn observe(tx: &Transaction<'_>, b: &Batch, seq: i64) {
    let rows = crate::input_rows::InputRows {
        input: b.input.id.clone(),
        format: b.input.format.clone(),
        revision: None,
        extent: None,
        observation: bytes::observation(tx, b, "deleted", None, seq).unwrap(),
    };
    rows.insert(tx, seq).unwrap();
}

#[test]
fn the_latest_observation_is_the_last_inserted_despite_earlier_times() {
    let w = test_archive();
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
        observe(&tx, &b, seq);
    }
    tx.commit().unwrap();
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
        let w = test_archive();
        let mut db =
            crate::archive::open_write(&w.archive.archive_db(), crate::archive::cache::SETUP)
                .unwrap();
        db.execute("INSERT INTO input VALUES ('input','synthetic-text')", [])
            .unwrap();
        let tx = db.transaction().unwrap();
        let seq = bump_counter(&tx).unwrap();
        let mut b = batch("input", 0, 0, vec![], &[]);
        b.observation.as_mut().unwrap().seen_at_ms = seen_at;
        observe(&tx, &b, seq);
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
