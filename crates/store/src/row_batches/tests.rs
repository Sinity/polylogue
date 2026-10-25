use super::*;
use crate::user_writes::{UserRequest, UserWrite};
use crate::{Delivery, DeliveryOutcome, Writer};

fn note(writer: &Writer, id: &str, text: &str) {
    writer
        .user(UserRequest::now(
            id.into(),
            UserWrite::AddNote {
                note: serde_json::from_value(serde_json::json!({
                    "kind": "note", "value": text, "author": "operator",
                    "targets": ["repo:synthetic"]
                }))
                .unwrap(),
            },
        ))
        .unwrap();
}

fn deliver(writer: &Writer, id: &str) {
    writer
        .user(UserRequest::now(
            id.into(),
            UserWrite::Deliver(Delivery {
                recipient_kind: "operator".into(),
                recipient_id: None,
                sender: "cli".into(),
                event: "advisory".into(),
                cwd: None,
                text: "Synthetic briefing".into(),
                refs: vec![],
                left_out: vec![],
            }),
        ))
        .unwrap();
}

/// Every batch with its rows, in write order.
fn contents(root: &Path) -> Vec<(RowBatch, Vec<String>)> {
    let batch_reader = BatchReader::open(root).unwrap();
    batch_reader
        .batches(None, usize::MAX)
        .unwrap()
        .into_iter()
        .map(|batch| {
            let mut rows = Vec::new();
            batch_reader
                .rows(&batch, |row| {
                    rows.push(format!("{} {:?} {:?}", row.table, row.columns, row.values));
                    Ok(())
                })
                .unwrap();
            (batch, rows)
        })
        .collect()
}

#[test]
fn batches_page_in_write_order() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    for n in 0..5 {
        note(
            &writer,
            &format!("note-{n}"),
            &format!("synthetic note {n}"),
        );
    }
    let batch_reader = BatchReader::open(dir.path()).unwrap();
    let all = batch_reader.batches(None, usize::MAX).unwrap();
    assert_eq!(all.len(), 5);
    assert!(all.windows(2).all(|pair| pair[0] < pair[1]));
    let mut paged = Vec::new();
    loop {
        let page = batch_reader.batches(paged.last(), 2).unwrap();
        if page.is_empty() {
            break;
        }
        paged.extend(page);
    }
    assert_eq!(paged, all);
}

#[test]
fn a_late_delivery_event_is_its_own_batch_and_leaves_the_delivery_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    deliver(&writer, "prepared");
    let before = contents(dir.path());
    assert_eq!(before.len(), 1);
    writer
        .user(UserRequest::now(
            "delivered".into(),
            UserWrite::DeliveryEvent {
                delivery: "prepared".into(),
                what: DeliveryOutcome::Delivered,
                detail: Some("Synthetic recipient accepted".into()),
            },
        ))
        .unwrap();
    let after = contents(dir.path());
    assert_eq!(after.len(), 2);
    assert_eq!(after[0], before[0]);
    assert_eq!(after[1].0.kind, "delivery_event");
}

#[test]
fn rows_restore_into_an_empty_archive_and_a_repeat_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");
    let target = dir.path().join("target");
    {
        let (archive, _lock) = Archive::open_or_create(&source).unwrap();
        let writer = Writer::start(&archive).unwrap();
        note(&writer, "note", "synthetic restored note");
        deliver(&writer, "delivery");
    }
    let expected = contents(&source);
    let batch_reader = BatchReader::open(&source).unwrap();
    let (archive, _lock) = Archive::open_or_create(&target).unwrap();
    let mut restore = Restore::begin(&archive).unwrap();
    // Batches may arrive in any order, and twice.
    for batch in batch_reader
        .batches(None, usize::MAX)
        .unwrap()
        .iter()
        .rev()
        .cycle()
        .take(4)
    {
        batch_reader.rows(batch, |row| restore.insert(row)).unwrap();
    }
    let inserted = restore.finish("synthetic-archive").unwrap();
    assert_eq!(contents(&target), expected);
    assert_eq!(
        inserted,
        expected
            .iter()
            .map(|(_, rows)| rows.len() as u64)
            .sum::<u64>()
    );
    let restored = BatchReader::open(&target).unwrap();
    assert_eq!(restored.archive_id().unwrap(), "synthetic-archive");
    let counter: String = restored
        .conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'write_counter'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(counter, expected.last().unwrap().0.commit_seq.to_string());
    assert!(on_sinex(&restored.conn).unwrap());
    assert!(matches!(
        Restore::begin(&archive),
        Err(StoreError::Invalid(message)) if message.contains("empty archive")
    ));
}

#[test]
fn a_restore_refuses_conflicting_unknown_or_dangling_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let columns: Vec<String> = [
        "id",
        "machine",
        "key",
        "value",
        "author",
        "set_at",
        "commit_seq",
    ]
    .map(String::from)
    .to_vec();
    let setting = |value: &str| {
        vec![
            Value::Text("setting".into()),
            Value::Text("*".into()),
            Value::Text("day.zone".into()),
            Value::Text(value.into()),
            Value::Text("operator".into()),
            Value::Integer(1),
            Value::Integer(1),
        ]
    };
    let mut restore = Restore::begin(&archive).unwrap();
    for _ in 0..2 {
        restore
            .insert(Row {
                table: "setting",
                columns: &columns,
                values: &setting("UTC"),
            })
            .unwrap();
    }
    let conflict = restore.insert(Row {
        table: "setting",
        columns: &columns,
        values: &setting("Europe/Warsaw"),
    });
    assert!(matches!(conflict, Err(StoreError::Invalid(m)) if m.contains("conflicts")));
    let unknown = restore.insert(Row {
        table: "client_request",
        columns: &columns,
        values: &setting("UTC"),
    });
    assert!(matches!(unknown, Err(StoreError::Invalid(m)) if m.contains("no known table")));
    let mut bad = columns.clone();
    bad[0] = "id) VALUES (1); --".into();
    let injected = restore.insert(Row {
        table: "setting",
        columns: &bad,
        values: &setting("UTC"),
    });
    assert!(matches!(injected, Err(StoreError::Invalid(m)) if m.contains("bad columns")));
    let target = ["note_id", "ref"].map(String::from);
    restore
        .insert(Row {
            table: "note_target",
            columns: &target,
            values: &[Value::Text("absent".into()), Value::Text("repo:x".into())],
        })
        .unwrap();
    let dangling = restore.finish("synthetic-archive");
    assert!(matches!(dangling, Err(StoreError::Invalid(m)) if m.contains("does not hold")));
}

#[test]
fn excision_refuses_once_sinex_holds_the_archive() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    note(&writer, "note", "synthetic text to keep");
    mark_on_sinex(&archive).unwrap();
    let store = std::sync::Arc::new(
        hashed_blobs::Store::open(dir.path().to_path_buf(), vec![], 0o600).unwrap(),
    );
    let request = |dry_run| crate::excise::ExciseRequest {
        id: "removal".into(),
        scope: crate::excise::Scope::Input("synthetic-input".into()),
        reason: None,
        dry_run,
    };
    assert!(matches!(
        writer.excise(request(true), store.clone()),
        Err(StoreError::NotFound(_))
    ));
    assert!(matches!(
        writer.excise(request(false), store),
        Err(StoreError::ExcisionOnSinex)
    ));
}
