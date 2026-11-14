use super::*;
use crate::older_archive::StoredVector;
use crate::writer::Restored;

const KEY: &str = "synthetic/model/2/document/unit";

fn paid(hash: u8, at: i64, values: &[f32]) -> StoredVector {
    StoredVector {
        text_hash: vec![hash],
        model: KEY.into(),
        chunk: 0,
        dims: values.len() as i64,
        vector: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        embedded_at: at,
    }
}

fn state(v: &Vectors) -> (Vec<(i64, Vec<u8>, i64)>, i64, i64) {
    let rows = v
        .conn
        .prepare("SELECT id,vector,embedded_at FROM embedded ORDER BY id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let sequence = v
        .conn
        .query_row(
            "SELECT coalesce(max(seq),0) FROM sqlite_sequence WHERE name='embedded'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    (rows, sequence, v.revision(KEY).unwrap())
}

#[test]
fn every_restore_order_keeps_the_newest_purchase_and_strict_repeat_writes_nothing() {
    for order in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let mut v = Vectors::writer(&archive, 256).unwrap();
        let versions = [
            paid(1, 10, &[1., 0.]),
            paid(1, 20, &[0., 1.]),
            paid(1, 30, &[-1., 0.]),
        ];
        let batch = order.map(|i| versions[i].clone());
        assert!(
            v.restore(&batch)
                .unwrap()
                .iter()
                .all(|r| !matches!(r, Restored::Refused(_)))
        );
        assert_eq!(state(&v).0, vec![(1, versions[2].vector.clone(), 30)]);
        let before = state(&v);
        let changes = v.conn.total_changes();
        assert_eq!(
            v.restore(&[versions[2].clone(), versions[0].clone()])
                .unwrap(),
            vec![Restored::Present; 2]
        );
        assert_eq!(state(&v), before, "order={order:?}");
        assert_eq!(
            v.conn.total_changes(),
            changes,
            "no trigger or row write on repeat/older input"
        );
    }
}

#[test]
fn paid_time_only_changes_are_written_and_ambiguous_or_dimension_conflicts_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let mut v = Vectors::writer(&archive, 256).unwrap();
    let original = paid(1, 10, &[1., 0.]);
    assert_eq!(
        v.restore(std::slice::from_ref(&original)).unwrap(),
        vec![Restored::Stored]
    );
    let before = state(&v);
    let newer = paid(1, 20, &[1., 0.]);
    assert_eq!(
        v.restore(std::slice::from_ref(&newer)).unwrap(),
        vec![Restored::Stored]
    );
    let after = state(&v);
    assert_eq!(after.0, vec![(before.0[0].0, original.vector, 20)]);
    assert_eq!(after.1, before.1);
    assert_eq!(after.2, before.2 + 1, "actual row trigger advances once");
    for invalid in [paid(1, 20, &[0., 1.]), paid(1, 30, &[1., 0., 0.])] {
        assert!(matches!(
            v.restore(&[invalid]).unwrap().as_slice(),
            [Restored::Refused(_)]
        ));
        assert_eq!(state(&v), after);
    }
    let mut overflow = paid(1, 30, &[1., 0.]);
    overflow.dims = i64::MAX;
    assert!(matches!(
        v.restore(&[overflow]).unwrap().as_slice(),
        [Restored::Refused(_)]
    ));
    let mut chunk = paid(1, 30, &[0., 1.]);
    chunk.chunk = 1;
    let mut model = paid(1, 30, &[1., 0., 0.]);
    model.model = "synthetic/model/3/document/unit".into();
    assert_eq!(
        v.restore(&[chunk, model]).unwrap(),
        vec![Restored::Stored; 2]
    );
    assert_eq!(
        state(&v).0.len(),
        3,
        "full model and chunk stay in the address"
    );
}

#[test]
fn mixed_restore_uses_row_revisions_and_refreshes_replacements_in_the_same_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let mut v = Vectors::writer(&archive, 256).unwrap();
    v.restore(&[paid(1, 10, &[1., 0.]), paid(2, 10, &[1., 0.])])
        .unwrap();
    let old = Vectors::snapshot(&archive, 256).unwrap();
    let base = old.load(KEY).unwrap();
    let before = state(&v);
    assert_eq!(
        v.restore(&[
            paid(1, 20, &[0., 1.]),
            paid(3, 20, &[-1., 0.]),
            paid(2, 30, &[1., 0.]),
        ])
        .unwrap(),
        vec![Restored::Stored; 3]
    );
    let after = state(&v);
    assert_eq!(
        after.2,
        before.2 + 3,
        "UPDATE + INSERT + paid-time UPDATE each advance the actual model trigger"
    );
    assert_eq!(after.1, before.1 + 1, "updates keep their IDs");
    let fresh = Vectors::snapshot(&archive, 256).unwrap();
    let refreshed = Arc::new(fresh.append(&base).unwrap());
    assert_eq!(refreshed.vectors(), 3);
    assert_eq!(
        refreshed.rows_read, 4,
        "one appended-row probe plus one three-row reload"
    );
    assert_eq!(refreshed.revision, after.2);
    let ranked = refreshed.similarity(&[1., 0.]).unwrap();
    assert_eq!(ranked.of(&[1]), Some(0.));
    assert_eq!(ranked.of(&[2]), Some(1.));
    assert_eq!(ranked.of(&[3]), Some(-1.));
    let older = Arc::new(old.append(&base).unwrap());
    assert_eq!(older.revision, base.revision);
    assert_eq!(older.vectors(), 2);
    assert_eq!(older.similarity(&[1., 0.]).unwrap().of(&[1]), Some(1.));
    // A metadata-only update is also visible without advancing max row ID.
    v.restore(&[paid(1, 40, &[0., 1.])]).unwrap();
    let newest = Vectors::snapshot(&archive, 256).unwrap();
    let metadata = newest.append(&refreshed).unwrap();
    assert_eq!(metadata.revision, refreshed.revision + 1);
    assert_eq!(metadata.rows_read, 3);
    assert_eq!(state(&v).1, after.1);
    let held = newest.append(&metadata).unwrap();
    assert_eq!(
        held.rows_read, 0,
        "unchanged revisions retain incremental no-op behavior"
    );
}

#[test]
fn delete_reinsert_does_not_keep_the_old_run_and_plain_append_stays_incremental() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let mut v = Vectors::writer(&archive, 256).unwrap();
    v.restore(&[paid(1, 10, &[1., 0.]), paid(2, 10, &[0., 1.])])
        .unwrap();
    let base = v.load(KEY).unwrap();
    // The released schema has row-level delete triggers too.
    v.conn
        .execute(
            "DELETE FROM embedded WHERE text_hash=x'01' AND model=?1",
            [KEY],
        )
        .unwrap();
    v.restore(&[paid(1, 20, &[-1., 0.])]).unwrap();
    let replaced = Arc::new(v.append(&base).unwrap());
    assert_eq!(replaced.revision, base.revision + 2);
    assert_eq!(replaced.vectors(), 2);
    assert_eq!(replaced.similarity(&[1., 0.]).unwrap().of(&[1]), Some(-1.));
    v.restore(&[paid(3, 30, &[1., 0.])]).unwrap();
    let appended = v.append(&replaced).unwrap();
    assert_eq!(appended.rows_read, 1);
    assert_eq!(appended.vectors(), 3);
}

#[test]
fn seeded_legacy_revision_and_failure_only_initial_loads_do_not_loop() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(include_str!(
        "../../schema/tests/embeddings_before_revision.sql"
    ))
    .unwrap();
    for hash in 1..=3u8 {
        conn.execute("INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at) VALUES(?1,?2,0,2,x'0000803f00000000',10)",params![vec![hash],KEY]).unwrap();
    }
    conn.pragma_update(None, "user_version", 1).unwrap();
    crate::schema::ensure_embeddings(&conn).unwrap();
    let mut v = Vectors {
        conn,
        writable: true,
    };
    assert_eq!(v.revision(KEY).unwrap(), 1);
    let base = v.load(KEY).unwrap();
    assert_eq!(base.vectors(), 3);
    v.restore(&[paid(4, 20, &[0., 1.])]).unwrap();
    let appended = v.append(&base).unwrap();
    assert_eq!((appended.vectors(), appended.rows_read), (4, 1));
    let other = "synthetic/failure/2/document/unit";
    v.fail(other, &[9], "refused").unwrap();
    let empty = v.load(other).unwrap();
    assert_eq!(empty.vectors(), 0);
    assert!(empty.revision > 0);
    let mut incoming = paid(8, 20, &[1., 0.]);
    incoming.model = other.into();
    v.restore(&[incoming]).unwrap();
    let loaded = v.append(&empty).unwrap();
    assert_eq!((loaded.vectors(), loaded.rows_read), (1, 1));
}
