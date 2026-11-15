use super::*;
use rusqlite::trace::{TraceEvent, TraceEventCodes};
use std::cell::Cell;

thread_local! {
    static RECEIPT_STEPS: Cell<u64> = const { Cell::new(0) };
}

fn trace(event: TraceEvent<'_>) {
    if let TraceEvent::Profile(statement, _) = event
        && statement
            .sql()
            .starts_with("UPDATE wire_receipt SET receipt")
    {
        RECEIPT_STEPS.with(|steps| {
            steps.set(
                steps.get()
                    + u64::try_from(statement.get_status(rusqlite::StatementStatus::VmStep))
                        .unwrap(),
            );
        });
    }
}

fn copy_receipts(count: i64, refuse_first: bool) -> u64 {
    let w = test_archive();
    chain(&w, 1, b'a', false);
    chain(&w, 2, b'b', true);
    chain(&w, 3, b'a', true);
    let mut conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.execute(
        "ATTACH DATABASE ?1 AS a",
        [w.archive.archive_db().to_str().unwrap()],
    )
    .unwrap();
    crate::source::views(&conn).unwrap();
    let (version, original): (String, String) = conn.query_row(
        "SELECT reading_version,receipt FROM wire_receipt WHERE chain=3 ORDER BY number LIMIT 1", [], |row| Ok((row.get(0)?,row.get(1)?))
    ).unwrap();
    conn.execute("INSERT INTO wire_reading SELECT input_pk,chain,'older=1',format,0,state FROM wire_reading WHERE chain=3", []).unwrap();
    conn.execute("INSERT INTO wire_receipt SELECT input_pk,chain,'older=1',number,receipt FROM wire_receipt WHERE chain=3 AND reading_version=?1", [&version]).unwrap();
    for n in 2..=count {
        for reading in [version.as_str(), "older=1"] {
            conn.execute("INSERT INTO wire_receipt SELECT input_pk,chain,reading_version,?1,receipt FROM wire_receipt WHERE chain=3 AND reading_version=?2 AND number=1", params![n,reading]).unwrap();
        }
    }
    let copy = crate::ReadingCopy {
        input: "doc".into(),
        from: 3,
        to: 1,
        reading_version: version.clone(),
        upto: 100,
        reaches_end: true,
    };
    if refuse_first {
        let mut invalid: serde_json::Value = serde_json::from_str(&original).unwrap();
        invalid["footprint"] = serde_json::json!({"kind":"unavailable"});
        conn.execute(
            "UPDATE wire_receipt SET receipt=?1 WHERE chain=3 AND reading_version=?2 AND number=?3",
            params![serde_json::to_string(&invalid).unwrap(), version, count],
        )
        .unwrap();
        let before = index_rows(&w, true);
        let tx = conn.transaction().unwrap();
        assert!(!crate::same_bytes::copy_reading(&tx, &copy, 99).unwrap());
        tx.rollback().unwrap();
        assert_eq!(
            index_rows(&w, true),
            before,
            "a refused retaking leaves no copied table or checkpoint"
        );
        conn.execute(
            "UPDATE wire_receipt SET receipt=?1 WHERE chain=3 AND reading_version=?2 AND number=?3",
            params![original, version, count],
        )
        .unwrap();
    }
    conn.set_prepared_statement_cache_capacity(0);
    RECEIPT_STEPS.with(|steps| steps.set(0));
    conn.trace_v2(TraceEventCodes::SQLITE_TRACE_PROFILE, Some(trace));
    let tx = conn.transaction().unwrap();
    assert!(crate::same_bytes::copy_reading(&tx, &copy, 100).unwrap());
    tx.commit().unwrap();
    conn.trace_v2(TraceEventCodes::empty(), None);
    let steps = RECEIPT_STEPS.with(Cell::get);
    let expected = crate::source::Source::reading_receipt(&conn, "doc", 1, 100).unwrap();
    let rows=conn.prepare("SELECT reading_version,number,receipt FROM wire_receipt WHERE chain=1 ORDER BY reading_version,number").unwrap().query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?,r.get::<_,String>(2)?))).unwrap().collect::<rusqlite::Result<Vec<_>>>().unwrap();
    assert_eq!(rows.len(), usize::try_from(count * 2).unwrap());
    for (i, (reading, number, text)) in rows.iter().enumerate() {
        assert_eq!(
            reading,
            if i < usize::try_from(count).unwrap() {
                "older=1"
            } else {
                version.as_str()
            }
        );
        assert_eq!(*number, i64::try_from(i).unwrap() % count + 1);
        let receipt: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(receipt["input"], "doc");
        assert_eq!(receipt["chain"], 1);
        assert_eq!(receipt["sequence"], expected.sequence);
        assert_eq!(receipt["id"], serde_json::json!(expected.id));
        assert_eq!(
            receipt["observation"],
            serde_json::json!(expected.observation)
        );
        assert_eq!(
            receipt["observed_at_ms"],
            serde_json::json!(expected.observed_at_ms)
        );
    }
    assert_eq!(
        conn.query_row("SELECT count(*) FROM wire_receipt WHERE chain=3", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        count * 2
    );
    assert!(
        conn.prepare("PRAGMA foreign_key_check")
            .unwrap()
            .query([])
            .unwrap()
            .next()
            .unwrap()
            .is_none()
    );
    let before = index_rows(&w, true);
    let tx = conn.transaction().unwrap();
    assert!(!crate::same_bytes::copy_reading(&tx, &copy, 101).unwrap());
    tx.rollback().unwrap();
    assert_eq!(
        index_rows(&w, true),
        before,
        "an exact repeat changes no rows or receipts"
    );
    steps
}

#[test]
fn copied_receipt_keys_retake_exactly_with_linear_work_and_atomic_refusal() {
    let small = copy_receipts(32, false);
    let large = copy_receipts(512, true);
    assert!(small > 0);
    assert!(
        large <= small * 20,
        "receipt count grew 16x, update VM work grew from {small} to {large}"
    );
}
