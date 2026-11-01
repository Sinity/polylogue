use super::*;
use polylogue_model::{historical::SourceSchema, query_objects::HistoricalRunReplay};

fn schema() -> SourceSchema {
    SourceSchema::PolylogueArchive { version: 4 }
}
fn original() -> QueryRun {
    let object = question("sessions");
    run(
        &object,
        "historical",
        1,
        ChangePosition {
            generation: "synthetic".into(),
            index: 2,
            archive: 1,
        },
    )
}
fn component() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::QUERY_OBJECTS).unwrap();
    conn
}
#[test]
fn historical_run_keeps_deep_available_bytes_and_requires_actual_projection_order() {
    let mut run = original();
    let mut nested = serde_json::json!("kept");
    for _ in 0..200 {
        nested = serde_json::json!([nested]);
    }
    run.binding.original_mut().unwrap().accepted_request = Some(nested);
    let encoded = serde_json::to_vec(&run).unwrap();
    let record = conserve_historical_run(
        schema(),
        &encoded,
        Some(run.binding_hash().unwrap()),
        Some(run.record_hash().unwrap()),
        Default::default(),
    )
    .unwrap();
    assert!(matches!(
        record.replay,
        HistoricalRunReplay::Compatible { .. }
    ));
    assert_eq!(record.encoded_run, encoded);
    record.validate().unwrap();
    let conn = component();
    assert!(apply_historical_run(&conn, &record, 1, None).is_err());
    assert_eq!(
        conn.query_row("SELECT count(*) FROM query_run", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    apply_historical_run(&conn, &record, 1, Some(4)).unwrap();
    apply_historical_run(&conn, &record, 1, Some(4)).unwrap();
    assert_eq!(
        lookup_run(&conn, Source::Ledger, &run.id, None)
            .unwrap()
            .unwrap()
            .value,
        run
    );
    let bytes: Vec<u8> = conn
        .query_row("SELECT historical FROM query_run", [], |row| row.get(0))
        .unwrap();
    let retained: polylogue_model::query_objects::HistoricalQueryRun =
        crate::encoding::unpack_json(&bytes).unwrap();
    assert_eq!(retained, record);
}
#[test]
fn retired_run_units_stay_unreplayable_and_unknown_summary_hashes_remain_unknown() {
    let original = original();
    let mut body = serde_json::to_value(&original).unwrap();
    body["binding"]["execution"]["replay"]["bound_ast"]["source"]["unit"] = "notes".into();
    let encoded = serde_json::to_vec(&body).unwrap();
    let record =
        conserve_historical_run(schema(), &encoded, None, None, Default::default()).unwrap();
    assert!(matches!(
        record.replay,
        HistoricalRunReplay::Unavailable { .. }
    ));
    let conn = component();
    apply_historical_run(&conn, &record, 1, None).unwrap();
    assert!(matches!(
        lookup_run(&conn, Source::Ledger, &original.id, None),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    let hashes: (Option<String>, Option<String>) = conn
        .query_row("SELECT binding_hash,run_hash FROM query_run", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(hashes, (None, None));
    let completion = outcome(&original, 2);
    apply_run_outcome(&conn, &completion).unwrap();
    assert_eq!(
        conn.query_row("SELECT binding_hash FROM query_run", [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        completion.binding_hash.to_string()
    );
    let before: Vec<u8> = conn
        .query_row("SELECT historical FROM query_run", [], |row| row.get(0))
        .unwrap();
    let mut conflicting = record.clone();
    let mut changed = body;
    changed["machine"] = "different".into();
    conflicting.encoded_run = serde_json::to_vec(&changed).unwrap();
    assert!(matches!(
        apply_historical_run(&conn, &conflicting, 2, Some(4)),
        Err(StoreError::Conflict(_))
    ));
    assert_eq!(
        conn.query_row("SELECT historical FROM query_run", [], |row| row
            .get::<_, Vec<u8>>(0))
            .unwrap(),
        before
    );
}
#[test]
fn predecessor_run_cannot_gain_request_object_reference_semantics() {
    let mut run = original();
    run.binding
        .original_mut()
        .unwrap()
        .query_options
        .direct_reference = Some(polylogue_model::Reference::Request("opaque".into()));
    let record = conserve_historical_run(
        schema(),
        &serde_json::to_vec(&run).unwrap(),
        None,
        None,
        Default::default(),
    )
    .unwrap();
    assert!(matches!(
        record.replay,
        HistoricalRunReplay::Unavailable { .. }
    ));
}

#[test]
fn cached_run_rebuild_keeps_original_row_integrity_and_cached_provenance() {
    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE source_input(id TEXT PRIMARY KEY) STRICT;")
        .unwrap();
    conn.execute_batch(
        &crate::source::from_archive(crate::schema::QUERY_OBJECTS)
            .unwrap()
            .ddl,
    )
    .unwrap();
    conn.execute_batch(crate::primary::projection::SCHEMA)
        .unwrap();
    let run = original();
    apply_run(&conn, &run, 8, 4).unwrap();
    let mut old = conserve_historical_run(
        schema(),
        &serde_json::to_vec(&run).unwrap(),
        Some(run.binding_hash().unwrap()),
        Some(run.record_hash().unwrap()),
        Default::default(),
    )
    .unwrap();
    old.records
        .push(b"retained original local encoding".to_vec());
    conn.execute(
        "UPDATE source_query_run SET historical=?1",
        [crate::encoding::pack_json(&old).unwrap()],
    )
    .unwrap();
    let raw = serde_json::to_vec(&serde_json::json!({"product":"query_run","run":run})).unwrap();
    conn.execute("INSERT INTO primary_fact(event,admitted,changed,source_order,state,product,kind) VALUES('actual-run-event',8,9,4,'live',?1,'query')",[crate::encoding::pack_bytes(&raw).unwrap()]).unwrap();
    let tx = conn.transaction().unwrap();
    crate::assertions::conversion::migrate_cached_products(&tx).unwrap();
    let packed: Vec<u8> = tx
        .query_row("SELECT product FROM primary_fact", [], |row| row.get(0))
        .unwrap();
    let Product::HistoricalQueryRun { run: retained } =
        crate::encoding::unpack_json(&packed).unwrap()
    else {
        panic!("run was not conserved")
    };
    assert_eq!(retained.declared_binding_hash, old.declared_binding_hash);
    assert_eq!(retained.declared_run_hash, old.declared_run_hash);
    assert_eq!(retained.encoded_run, old.encoded_run);
    assert!(retained.records.contains(&old.records[0]));
    assert!(retained.records.contains(&raw));
    assert_eq!(retained.cached.len(), 1);
    assert_eq!(retained.cached[0].event, "actual-run-event");
    assert_eq!(retained.cached[0].admitted, Some(8));
    assert!(retained.native.observed.is_empty());
    crate::query_objects::clear(&tx).unwrap();
    apply_historical_run(&tx, &retained, 8, Some(4)).unwrap();
    let rebuilt: Vec<u8> = tx
        .query_row("SELECT historical FROM source_query_run", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(
        crate::encoding::unpack_json::<polylogue_model::query_objects::HistoricalQueryRun>(
            &rebuilt
        )
        .unwrap(),
        *retained
    );
    tx.commit().unwrap();
}

#[test]
fn old_notes_question_retains_its_identity_but_is_not_current_native_or_replay_input() {
    let archive = WriterArchive::new();
    let object = crate::fixture::query_objects::synthetic_source3_notes().unwrap();
    validate_integrity(&object).unwrap();
    let mut conn = crate::archive::open_archive_writer(&archive.archive).unwrap();
    let tx = conn.transaction().unwrap();
    apply_object(&tx, &object).unwrap();
    tx.commit().unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let retained = reader
        .query_object(&object.id.to_string(), None)
        .unwrap()
        .unwrap();
    assert_eq!(retained.value, object);
    assert_eq!(retained.custody, QueryCustody::Canonical);
    let error = archive.writer.query_object(object.clone()).unwrap_err();
    assert!(matches!(error, StoreError::Invalid(_)), "{error}");
    let options = polylogue_query::Options {
        clock: Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            DaySetting::new(jiff::tz::TimeZone::UTC, jiff::civil::time(0, 0, 0, 0)).unwrap(),
        ),
        origins: vec![],
        command_words: vec![],
        page_size: None,
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        chat_pages: vec![],
    };
    let replay = reader.prepare_query_object_selection(
        &retained.value,
        &options,
        polylogue_query::plan::catalog_fingerprint(),
    );
    let error = replay
        .err()
        .expect("retired Notes must not bind to Assertions");
    assert!(
        matches!(error, crate::selection::SelectionError::Query(_)),
        "{error}"
    );
    assert_eq!(
        reader
            .query_object(&object.id.to_string(), None)
            .unwrap()
            .unwrap()
            .value,
        object
    );
}

#[test]
fn portable_run_history_demotes_carried_observations_and_restores_idempotently() {
    use polylogue_model::{
        Author,
        authored::ImportAttribution,
        historical::{NativeProjectionObservation, NativeProjectionProvenance},
    };
    let conn = component();
    let mut carried = crate::fixture::query_objects::synthetic_source3_run().unwrap();
    let source_claim = NativeProjectionObservation {
        source: schema(),
        provenance: NativeProjectionProvenance {
            event: "synthetic-carried-event".into(),
            run: "synthetic-native-run".into(),
            commit_seq: 3,
            output_index: 2,
            record_hash: "carried-record-hash".into(),
            projection_epoch: 4,
            projection_payload_hash: "carried-projection-hash".into(),
            location: None,
            parents: vec![],
            producer_family: None,
            producer_identity: None,
            ts_orig: None,
            ts_quality: "unknown".into(),
            ts_coided: None,
            ts_persisted: Timestamp::from_unix_ms(1).unwrap(),
            record_state: "live".into(),
            class: "record".into(),
            validation: "valid".into(),
            equivalence_key: None,
            reading_order: 2,
            projected_payload: b"synthetic carried payload".to_vec(),
        },
    };
    carried.native.observed.push(source_claim.clone());
    let ingress = ImportAttribution {
        machine: "actual-synthetic-receiver".into(),
        input: "bundle-source".into(),
        chain: 1,
        actor: Author::Operator,
        at: Timestamp::from_unix_ms(2).unwrap(),
    };
    let sender = ImportAttribution {
        machine: "sender-claim".into(),
        input: "prior-source".into(),
        chain: 7,
        actor: Author::Operator,
        at: Timestamp::from_unix_ms(1).unwrap(),
    };
    carried.imports.observed.push(sender.clone());
    restore_historical_run(&conn, &carried, &[], &ingress, 1).unwrap();
    let (known, _) = historical_run_on(&conn, &carried.id).unwrap().unwrap();
    assert!(known.native.observed.is_empty());
    assert_eq!(known.native.declared, vec![source_claim]);
    assert_eq!(known.imports.observed, vec![ingress.clone()]);
    assert_eq!(known.imports.declared, vec![sender]);
    assert_eq!(known.encoded_run, carried.encoded_run);
    assert_eq!(known.declared_binding_hash, None);
    assert_eq!(known.declared_run_hash, None);
    let mut later = ingress.clone();
    later.at = Timestamp::from_unix_ms(99).unwrap();
    restore_historical_run(&conn, &carried, &[], &later, 2).unwrap();
    assert_eq!(
        historical_run_on(&conn, &carried.id).unwrap().unwrap().0,
        known
    );
    let mut conflicting = carried.clone();
    let mut changed: serde_json::Value =
        crate::encoding::read_json(&conflicting.encoded_run).unwrap();
    changed["machine"] = "different-machine".into();
    conflicting.encoded_run = serde_json::to_vec(&changed).unwrap();
    assert!(matches!(
        restore_historical_run(&conn, &conflicting, &[], &ingress, 3),
        Err(StoreError::Conflict(_))
    ));
    assert_eq!(
        historical_run_on(&conn, &carried.id).unwrap().unwrap().0,
        known
    );
    clear(&conn).unwrap();
    restore_historical_run(&conn, &carried, &[], &ingress, 1).unwrap();
    assert_eq!(
        historical_run_on(&conn, &carried.id).unwrap().unwrap().0,
        known
    );
}
