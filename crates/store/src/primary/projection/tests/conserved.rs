use super::*;
use crate::primary::{
    ConfirmedOrigin, NativeFactContext, NativeKind, NativeSource, ProjectionIdentity,
};
use polylogue_model::{
    Timestamp,
    historical::{NativeProjectionProvenance, SourceSchema},
};

fn observed(position: i64, epoch: i64, text: &str) -> Confirmed {
    let request = serde_json::json!({"id":"native-history","machine":"sender","at":"1970-01-01T00:00:00.001Z","authorship":{"author":"operator","via":"cli"},"write":{"write":"add_note","note":{"kind":"note","value":text,"targets":["repo:synthetic"]}}});
    let outcome = serde_json::json!({"id":"native-history","repeat":false,"pending":false});
    let effect = serde_json::json!({"notes":[{"note":{"id":"native-history","first_note":"native-history","kind":"note","value":text,"author":"operator","created_at":"1970-01-01T00:00:00.001Z","targets":["repo:synthetic"]},"machine":"sender","revises":[]}],"withdrawals":[],"settings":[],"variants":{"variants":[],"variant_parts":[]},"kept_results":[],"deliveries":[],"chats":[],"backfills":[]});
    let raw=serde_json::to_vec(&serde_json::json!({"record":{"product":"request","request":request,"outcome":outcome,"effect":effect}})).unwrap();
    let source = SourceSchema::NativeProjection {
        source: "polylogue".into(),
        source_version: 3,
        source_fingerprint: "frozen-source3".into(),
        kind: "polylogue.note".into(),
        kind_version: 4,
        kind_definition_hash: "0".repeat(64),
    };
    let accepted = crate::assertions::conversion::conserve_request(
        source,
        &serde_json::to_vec(&request).unwrap(),
        Some(&serde_json::to_vec(&outcome).unwrap()),
        Some(&serde_json::to_vec(&effect).unwrap()),
        None,
        Some(NativeProjectionProvenance {
            event: "native-event".into(),
            run: "native-run".into(),
            commit_seq: 1,
            output_index: 0,
            record_hash: "2".repeat(64),
            projection_epoch: epoch,
            projection_payload_hash: "1".repeat(64),
            location: None,
            parents: vec![],
            producer_family: Some("polylogue".into()),
            producer_identity: Some("synthetic-producer".into()),
            ts_orig: None,
            ts_quality: "unknown".into(),
            ts_coided: None,
            ts_persisted: Timestamp::from_unix_ms(1).unwrap(),
            record_state: "present".into(),
            class: "work".into(),
            validation: "full".into(),
            equivalence_key: None,
            reading_order: 0,
            projected_payload: raw,
        }),
    )
    .unwrap();
    Confirmed {
        position,
        event: "native-event".into(),
        output_index: 0,
        native: Some(NativeFactContext {
            kind: NativeKind {
                name: "polylogue.note".into(),
                version: 4,
            },
            kind_definition_hash: "0".repeat(64),
            projection: ProjectionIdentity {
                policy_epoch: epoch,
                payload_hash: "1".repeat(64),
            },
            origin: ConfirmedOrigin {
                source: Some(NativeSource {
                    name: "polylogue".into(),
                    version: 3,
                }),
                source_fingerprint: Some("frozen-source3".into()),
                producer_family: Some("polylogue".into()),
                producer_identity: Some("synthetic-producer".into()),
            },
        }),
        change: ConfirmedChange::Live(Box::new(Product::ConservedRequest {
            accepted,
            claims: vec![],
        })),
    }
}

#[test]
fn conserved_same_event_observations_survive_repeat_and_the_actual_authored_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let mut cached = observed(1, 2, "kept");
    let cached_context = polylogue_model::historical::CachedNativeContext {
        event: cached.event.clone(),
        admitted: Some(1),
        position: 1,
        source_order: 0,
        state: polylogue_model::historical::CachedRecordState::Live,
        native: cached.native.clone(),
    };
    let ConfirmedChange::Live(product) = &mut cached.change else {
        unreachable!()
    };
    let Product::ConservedRequest {
        accepted: polylogue_model::historical::ConservedAccepted::Historical(record),
        ..
    } = product.as_mut()
    else {
        unreachable!()
    };
    record
        .original
        .as_mut()
        .unwrap()
        .cached
        .push(cached_context.clone());
    apply(&mut index, &resolver, &[cached, barrier(1)]).unwrap();
    for (position, epoch) in [(2, 3), (3, 3)] {
        apply(
            &mut index,
            &resolver,
            &[observed(position, epoch, "kept"), barrier(position)],
        )
        .unwrap();
    }
    let before = crate::primary::authored::historical::get(&index, "native-history")
        .unwrap()
        .unwrap();
    assert_eq!(
        before
            .native
            .observed
            .iter()
            .map(|entry| entry.provenance.projection_epoch)
            .collect::<Vec<_>>(),
        [2, 3]
    );
    assert!(before.imports.observed.is_empty());
    assert_eq!(before.original.as_ref().unwrap().cached, [cached_context]);
    let actual = super::super::replay::native_context(&index, "native-event")
        .unwrap()
        .unwrap();
    assert_eq!(actual.projection.policy_epoch, 3);
    let plan = Plan {
        readings: vec![],
        inputs: Default::default(),
        materials: Default::default(),
        available: Default::default(),
        requests: true,
        queries: false,
        binding_messages: Default::default(),
        coverage: false,
    };
    let tx = index.transaction().unwrap();
    requests(&tx, &plan).unwrap();
    tx.commit().unwrap();
    let rebuilt = crate::primary::authored::historical::get(&index, "native-history")
        .unwrap()
        .unwrap();
    assert_eq!(rebuilt, before);
    let cached: Vec<u8> = index
        .query_row(
            "SELECT product FROM primary_fact WHERE event='native-event'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        apply(
            &mut index,
            &resolver,
            &[observed(4, 3, "changed"), barrier(4)]
        )
        .is_err()
    );
    assert_eq!(
        index
            .query_row(
                "SELECT product FROM primary_fact WHERE event='native-event'",
                [],
                |row| row.get::<_, Vec<u8>>(0)
            )
            .unwrap(),
        cached
    );
    assert_eq!(
        crate::primary::authored::historical::get(&index, "native-history")
            .unwrap()
            .unwrap(),
        before
    );
}

#[test]
fn pending_conserved_redelivery_merges_only_the_receive_only_wrapper() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let first = observed(1, 2, "kept");
    stage(&mut index, std::slice::from_ref(&first)).unwrap();
    stage(&mut index, &[observed(1, 3, "kept")]).unwrap();
    stage(&mut index, &[observed(1, 3, "kept")]).unwrap();
    let bytes: Vec<u8> = index
        .query_row(
            "SELECT product FROM primary_pending WHERE event='native-event'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let ConfirmedChange::Live(product) = unpacked(&bytes).unwrap() else {
        panic!("pending live wrapper missing")
    };
    let Product::ConservedRequest {
        accepted: polylogue_model::historical::ConservedAccepted::Historical(record),
        ..
    } = product.as_ref()
    else {
        panic!("pending historical request missing")
    };
    assert_eq!(record.native.observed.len(), 2);
    assert!(stage(&mut index, &[observed(1, 3, "changed")]).is_err());
    assert_eq!(
        index
            .query_row(
                "SELECT product FROM primary_pending WHERE event='native-event'",
                [],
                |row| row.get::<_, Vec<u8>>(0)
            )
            .unwrap(),
        bytes
    );
}
