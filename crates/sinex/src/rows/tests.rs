use super::*;
use polylogue_store::Archive;
use polylogue_store::row_batches::Restore;
use polylogue_testkit::{Morrow, Set};
use std::io::Read;

/// Every table's rows, sorted. Blob rows are reduced to what a batch
/// carries: Sinex holds every blob as a plain file.
fn dump(root: &std::path::Path) -> Vec<String> {
    let db = rusqlite::Connection::open(root.join("archive.db")).unwrap();
    let mut out = Vec::new();
    for table in [
        "input",
        "revision",
        "extent",
        "machine",
        "observation",
        "note",
        "note_target",
        "note_relation",
        "setting",
        "delivery",
        "delivery_event",
        "excision",
    ] {
        let mut statement = db.prepare(&format!("SELECT * FROM {table}")).unwrap();
        let columns = statement.column_count();
        let mut rows = statement.query([]).unwrap();
        while let Some(row) = rows.next().unwrap() {
            let values: Vec<_> = (0..columns)
                .map(|i| format!("{:?}", row.get_ref(i).unwrap()))
                .collect();
            out.push(format!("{table} {values:?}"));
        }
    }
    // Blobs only records carry inline are derived again by parsing.
    let mut blobs = db
        .prepare(
            "SELECT hash, size FROM blob
             WHERE hash IN (SELECT blob FROM extent)",
        )
        .unwrap();
    for row in blobs
        .query_map([], |r| Ok((r.get::<_, Vec<u8>>(0)?, r.get::<_, i64>(1)?)))
        .unwrap()
    {
        out.push(format!("blob {:?}", row.unwrap()));
    }
    out.sort();
    out
}

#[test]
fn every_batch_of_a_real_archive_round_trips_through_its_encoding() {
    let world = Morrow::build(Set::Main);
    let batch_reader = BatchReader::open(&world.root).unwrap();
    let batches = batch_reader.batches(None, usize::MAX).unwrap();
    assert!(batches.iter().any(|b| b.kind == "revision"));
    assert!(batches.iter().any(|b| b.kind == "observation"));
    let target = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(target.path()).unwrap();
    let mut restore = Restore::begin(&archive).unwrap();
    let mut cited = 0;
    for batch in &batches {
        let mut encoded = encode(&batch_reader, batch).unwrap();
        let mut bytes = Vec::new();
        encoded.file.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes.len() as u64, encoded.length);
        assert_eq!(hashed_blobs::Hash::of(&bytes), encoded.hash);
        assert_eq!(
            encode(&batch_reader, batch).unwrap().hash,
            encoded.hash,
            "the same rows give the same bytes"
        );
        cited += encoded.blobs.len();
        let rows = decode(bytes.as_slice(), |row| Ok(restore.insert(row)?)).unwrap();
        assert_eq!(rows, encoded.rows);
    }
    assert!(cited > 0);
    restore.finish(&batch_reader.archive_id().unwrap()).unwrap();
    assert_eq!(dump(target.path()), dump(&world.root));
}

#[test]
fn decoding_refuses_values_it_cannot_store_exactly() {
    let refuse = |line: &str| {
        let result = decode(line.as_bytes(), |_| Ok(()));
        assert!(
            matches!(result, Err(Error::Invalid(_))),
            "{line}: {result:?}"
        );
    };
    refuse(r#"{"row":{"n":18446744073709551615},"table":"setting"}"#);
    refuse(r#"{"row":{"hash":{"hex":"abc"}},"table":"blob"}"#);
    refuse(r#"{"row":{"hash":{"hex":"zz"}},"table":"blob"}"#);
    refuse(r#"{"row":{"list":[1]},"table":"setting"}"#);
    refuse(r#"{"row":{"id":"x"}}"#);
    let mut seen = Vec::new();
    decode(
        r#"{"row":{"big":-9223372036854775808,"real":0.5,"bytes":{"hex":"00ff"},"none":null},"table":"t"}"#
            .as_bytes(),
        |row| {
            seen.extend(row.columns.iter().cloned().zip(row.values.iter().cloned()));
            Ok(())
        },
    )
    .unwrap();
    seen.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        seen,
        [
            ("big".to_owned(), Value::Integer(i64::MIN)),
            ("bytes".to_owned(), Value::Blob(vec![0, 255])),
            ("none".to_owned(), Value::Null),
            ("real".to_owned(), Value::Real(0.5)),
        ]
    );
}
