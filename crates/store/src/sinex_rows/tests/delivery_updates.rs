use super::*;
use crate::archive::{Archive, cache, open_write};
use crate::user_writes::{UserRequest, UserWrite};
use crate::{Delivery, DeliveryOutcome, Writer};
use std::io::Cursor;

fn accepted_late_event(archive: &Archive) -> (String, RestoreUnit, Vec<u8>) {
    let writer = Writer::start(archive).unwrap();
    writer
        .user(UserRequest::now(
            "prepared".into(),
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
    let published = Published::open(&archive.root).unwrap();
    let id = published.archive_id().unwrap();
    let inventory = published.inventory().unwrap();
    assert_eq!(inventory.len(), 1);
    let unit = inventory.unit(1).unwrap();
    let batch = published.read(&unit).unwrap();
    let bytes = batch.contents();
    let original = RestoreUnit {
        unit,
        batch: Some(format!("blake3:{}", blake3::hash(&bytes))),
        bytes: Some(bytes.len() as u64),
        rows: batch.rows,
        blobs: vec![],
        excision: None,
    };
    drop(inventory);
    drop(published);
    let ack = writer
        .user(UserRequest::now(
            "delivered".into(),
            UserWrite::DeliveryEvent {
                delivery: "prepared".into(),
                what: DeliveryOutcome::Delivered,
                detail: Some("Synthetic recipient accepted".into()),
            },
        ))
        .unwrap();
    assert!(!ack.repeat);
    let repeat = writer
        .user(UserRequest::now(
            "delivered".into(),
            UserWrite::DeliveryEvent {
                delivery: "prepared".into(),
                what: DeliveryOutcome::Delivered,
                detail: Some("Synthetic recipient accepted".into()),
            },
        ))
        .unwrap();
    assert!(repeat.repeat);
    drop(writer);
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM delivery_event", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    (id, original, bytes)
}

#[test]
fn a_late_delivery_event_remains_pending_after_the_delivery_is_held() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let (_, original, bytes) = accepted_late_event(&archive);
    let published = Published::open(&archive.root).unwrap();
    let mut inventory = published.inventory().unwrap();
    inventory.mark_held(&[1]).unwrap();
    let current = published.read(&original.unit).unwrap();
    println!(
        "units={}, held={}, old_rows={}, current_rows={}, old_hash={}, current_hash={}",
        inventory.len(),
        inventory.held(),
        original.rows,
        current.rows,
        blake3::hash(&bytes),
        blake3::hash(&current.contents())
    );
    assert_eq!(
        current.contents(),
        bytes,
        "later facts must not change a parent batch"
    );
    assert_eq!(inventory.len(), 2);
    assert_eq!(inventory.unit(2).unwrap().kind, "delivery_event");
    assert!(
        inventory.last_missing().unwrap().is_some(),
        "accepted delivery event has no pending publication identity"
    );
}

#[test]
fn restoring_an_older_confirmed_view_preserves_a_later_local_delivery_event() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let (id, original, bytes) = accepted_late_event(&archive);
    let selected = select(&[original]).unwrap();
    let mut restoring = Restoring::open(&archive).unwrap();
    let result = restoring.install(&id, &selected, |_| Ok(Some(Cursor::new(bytes.clone()))));
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let events: i64 = conn
        .query_row("SELECT count(*) FROM delivery_event", [], |r| r.get(0))
        .unwrap();
    println!("restore_result={result:?}, retained_events={events}");
    assert!(
        result.is_err() || events == 1,
        "older confirmed view erased an accepted local delivery event"
    );
}
