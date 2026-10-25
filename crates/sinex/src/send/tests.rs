use super::*;
use crate::tests::{Failure, StandIn};
use polylogue_testkit::{Morrow, Set};
use std::collections::BTreeSet;

fn recorded_names(stand_in: &StandIn) -> BTreeSet<String> {
    stand_in
        .record
        .lock()
        .unwrap()
        .frames
        .keys()
        .cloned()
        .collect()
}

/// Every batch position from 1 to the last is recorded, each after the blobs
/// and rows material it cites, and no frame was refused.
fn assert_no_gap(stand_in: &StandIn, archive: &str, batches: u64) {
    let record = stand_in.record.lock().unwrap();
    for (name, (_, acknowledgment)) in &record.frames {
        assert_eq!(acknowledgment.refusal, None, "{name}");
    }
    for position in 1..=batches {
        let name = batch_name(archive, position);
        let at = record.position(&name).unwrap_or_else(|| panic!("{name}"));
        let Ingress::Events { events, .. } = &record.frames[&name].0 else {
            panic!("{name} is not an events frame")
        };
        let rows: PolylogueRows = serde_json::from_value(events[0].payload.clone()).unwrap();
        assert_eq!(rows.position, position);
        for blob in &rows.blobs {
            let blob_frame = format!("blob:{}", blob.strip_prefix("blake3:").unwrap());
            assert!(record.position(&blob_frame).unwrap() < at, "{blob_frame}");
        }
    }
    assert!(record.position(&batch_name(archive, batches + 1)).is_none());
}

#[tokio::test]
async fn a_pass_sends_what_sinex_lacks_and_a_second_pass_sends_nothing() {
    let world = Morrow::build(Set::Main);
    let batch_reader = BatchReader::open(&world.root).unwrap();
    let archive = batch_reader.archive_id().unwrap();
    let batches = batch_reader.batches(None, usize::MAX).unwrap().len() as u64;
    let stand_in = StandIn::start().await;
    let mut sinex = stand_in.connect().await;
    let sent = sinex.send_missing(&world.root).await.unwrap();
    assert_eq!(sent.batches, batches);
    assert!(sent.blobs > 0 && sent.rows > 0, "{sent:?}");
    assert_no_gap(&stand_in, &archive, batches);
    assert!(stand_in.spool_is_empty());
    assert_eq!(
        sinex.send_missing(&world.root).await.unwrap(),
        Sent::default()
    );
    let received = stand_in.record.lock().unwrap().received;
    assert_eq!(
        stand_in
            .connect()
            .await
            .send_missing(&world.root)
            .await
            .unwrap(),
        Sent::default(),
        "a new connection asks Sinex instead of sending again"
    );
    assert_eq!(stand_in.record.lock().unwrap().received, received);
}

#[tokio::test]
async fn an_outage_or_a_lost_acknowledgment_mid_pass_leaves_no_gap() {
    let world = Morrow::build(Set::Main);
    let batch_reader = BatchReader::open(&world.root).unwrap();
    let archive = batch_reader.archive_id().unwrap();
    let batches = batch_reader.batches(None, usize::MAX).unwrap().len() as u64;

    let clean = StandIn::start().await;
    clean
        .connect()
        .await
        .send_missing(&world.root)
        .await
        .unwrap();

    let stand_in = StandIn::start().await;
    // A blob, a batch's rows material and its event lost on the way, and
    // others recorded without their acknowledgment reaching Polylogue.
    for (at, failure) in [
        (1, Failure::Outage),
        (3, Failure::LostAcknowledgment),
        (5, Failure::Outage),
        (6, Failure::LostAcknowledgment),
        (9, Failure::LostAcknowledgment),
        (12, Failure::Outage),
    ] {
        stand_in.fail(at, failure);
    }
    let mut failed = 0;
    loop {
        // Each attempt is a new connection, as after a restart.
        match stand_in.connect().await.send_missing(&world.root).await {
            Ok(_) => break,
            Err(Error::Unreachable(_)) => failed += 1,
            Err(other) => panic!("{other}"),
        }
        assert!(failed <= 6, "the failures did not end");
    }
    assert_eq!(failed, 6);
    assert_no_gap(&stand_in, &archive, batches);
    assert_eq!(recorded_names(&stand_in), recorded_names(&clean));
    assert!(stand_in.spool_is_empty());
}

#[tokio::test]
async fn a_frame_too_large_for_one_message_travels_indirectly_under_its_own_name() {
    let stand_in = StandIn::start().await;
    let mut sinex = stand_in.connect().await;
    sinex.max_message = 8192;
    let bytes = b"synthetic material bytes";
    let hash = hashed_blobs::Hash::of(bytes);
    let large = "x".repeat(16000);
    let material = sinex
        .material(
            "large-material".into(),
            &hash,
            bytes.len() as u64,
            &bytes[..],
            serde_json::json!({ "note": large }),
        )
        .await
        .unwrap();
    let events = Ingress::Events {
        frame: "large-events".into(),
        failures: vec![],
        events: vec![ProducerEvent {
            event_type: "polylogue.rows".into(),
            payload: serde_json::json!({ "note": large }),
            location: Some(RecordLocation {
                material,
                anchor: 0,
            }),
            ordinal: 0,
            parents: vec![],
            reading_order: 1,
            class: Class::Open,
        }],
    };
    let acknowledgment = sinex.send(events.clone()).await.unwrap();
    assert_eq!(acknowledgment.frame, "large-events");
    let record = stand_in.record.lock().unwrap();
    assert_eq!(record.indirect, ["large-material", "large-events"]);
    assert_eq!(record.frames["large-events"].0, events);
    let Ingress::Material { material: read, .. } = &record.frames["large-material"].0 else {
        panic!("the material frame is read back")
    };
    assert_eq!(read.id, material);
    assert_eq!(read.manifest["note"], large.as_str());
    drop(record);
    assert!(stand_in.spool_is_empty());
}
