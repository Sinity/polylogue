use super::*;
use crate::fixture::{Fixture, SessionFacts};
use polylogue_model::{
    Author, BlobHash,
    assertion::{AssertionFact, AssertionKind, values},
};

fn definition(id: &str, name: &str, text: &str, hash: QuestionId) -> AssertionFact {
    AssertionFact {
        id: id.into(),
        root: id.into(),
        kind: AssertionKind::SavedQuery,
        key: Some(name.into()),
        value: Some(
            serde_json::to_value(values::SavedQuery {
                text: text.into(),
                hash: BlobHash(hash.0),
                denominator: Some(format!("query:{hash}")),
            })
            .unwrap(),
        ),
        author: Author::Unattributed,
        via: None,
        machine: None,
        created_at: None,
        valid_until: None,
        links: vec![],
    }
}

#[test]
fn historical_saved_name_without_identity_refuses_only_when_used() {
    use polylogue_model::historical::{
        HistoricalAssertionEvidence, NoteKind, RecordedKind, SourceSchema,
    };
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();

    crate::assertions::apply_historical(
        &conn,
        &HistoricalAssertionEvidence {
            predecessor: None,
            id: "old-definition".into(),
            root: "old-definition".into(),
            source: SourceSchema::PolylogueArchive { version: 4 },
            kind: RecordedKind::Known(NoteKind::SavedQuery),
            key: Some("historical".into()),
            value: Some("sessions where repo:old".into()),
            recorded_author: None,
            machine: None,
            created_at: None,
            valid_until: None,
            targets: vec![],
            relations: vec![],
            links: vec![],
        },
        1,
    )
    .unwrap();
    conn.execute_batch("BEGIN").unwrap();
    let native = Bindings::default();
    assert!(
        saved_queries_on(&conn, &conn, None, None)
            .unwrap()
            .is_empty()
    );
    prepare_query_plan_on(&conn, &conn, "sessions", &native, None, None, None).unwrap();
    prepare_query_plan_on(
        &conn,
        &conn,
        "sessions where repo:historical",
        &native,
        None,
        None,
        None,
    )
    .unwrap();
    let refused =
        prepare_query_plan_on(&conn, &conn, "query:historical", &native, None, None, None)
            .unwrap_err();
    assert_eq!(refused.kind(), "not_yet_available");
    assert!(
        refused
            .to_string()
            .contains("assertion:old-definition@old-definition")
    );
}

#[test]
fn accepting_preparation_uses_original_native_rows_and_scratch_definitions() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    fixture
        .session(SessionFacts {
            id: "test:latest",
            last: Some(1000),
            ..Default::default()
        })
        .unwrap();
    let object =
        polylogue_query::plan::prepare("sessions where repo:scratch", &Bindings::default(), |_| {
            Ok(())
        })
        .unwrap()
        .query_object()
        .unwrap();
    let archive = crate::archive::open_write(&fixture.archive.archive_db(), 1).unwrap();

    crate::query_objects::apply_object(&archive, &object).unwrap();
    let unnamed =
        polylogue_query::plan::prepare("sessions where repo:unnamed", &Bindings::default(), |_| {
            Ok(())
        })
        .unwrap()
        .query_object()
        .unwrap();
    crate::query_objects::apply_object(&archive, &unnamed).unwrap();
    drop(archive);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let original = reader.conn().unwrap();

    let mut scratch = Connection::open_in_memory().unwrap();
    let schema = crate::source::from_archive(crate::schema::ARCHIVE).unwrap();
    scratch.execute_batch(&schema.ddl).unwrap();
    crate::assertions::apply(
        &scratch,
        &[definition(
            "pending",
            "focus",
            "sessions where repo:scratch",
            object.id,
        )],
        1,
    )
    .unwrap();
    let tx = scratch.transaction().unwrap();
    let by_hash = prepare_query_plan_on(
        original,
        &tx,
        &format!("query:{}", unnamed.id),
        &Bindings::default(),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(by_hash.definitions()[0].question_id, unnamed.id);
    let restored =
        polylogue_query::plan::replay(&unnamed, &polylogue_query::plan::catalog_fingerprint())
            .unwrap();
    assert_eq!(
        by_hash.expanded_query().query(),
        restored.expanded_query().query()
    );
    let missing = format!("query:{}", "00".repeat(32));
    assert_eq!(
        prepare_query_plan_on(
            original,
            &tx,
            &missing,
            &Bindings::default(),
            None,
            None,
            None
        )
        .unwrap_err()
        .kind(),
        "not_yet_available"
    );
    let refused = prepare_query_plan_on(
        original,
        &tx,
        &missing,
        &Bindings::default(),
        None,
        None,
        None,
    )
    .unwrap_err();
    let detail = refused.detail();
    let roundtrip = SelectionError::from(refused.into_store_error());
    assert_eq!(roundtrip.kind(), "not_yet_available");
    assert_eq!(roundtrip.detail(), detail);
    let mut native = Bindings {
        origins: vec!["test".into()],
        ..Default::default()
    };
    // Untrusted supplied dynamic definitions cannot replace either actual view.
    let forged =
        polylogue_query::plan::prepare("sessions where repo:forged", &Bindings::default(), |_| {
            Ok(())
        })
        .unwrap();
    native
        .add_saved("focus", &forged, "forged", "forged")
        .unwrap();
    let actual =
        prepare_query_plan_on(original, &tx, "query:focus", &native, None, None, None).unwrap();
    let loaded = saved_queries_on(original, &tx, None, None).unwrap();
    assert_eq!(loaded["focus"][0].hash, object.id);
    assert_eq!(loaded["focus"][0].denominator, Some(object.id));
    assert_eq!(actual.definitions()[0].question_id, object.id);
    let latest = prepare_query_plan_on(original, &tx, "latest", &native, None, None, None).unwrap();
    let explicit =
        polylogue_query::plan::prepare("session:test:latest", &native, |_| Ok(())).unwrap();
    assert_eq!(latest.question_id(), explicit.question_id());
    let ordinary =
        prepare_query_plan_on(original, original, "sessions", &native, None, None, None).unwrap();
    let mut options = super::super::tests::options();
    options.origins = native.origins;
    assert_eq!(
        ordinary.question_id(),
        reader
            .prepare_query_plan("sessions", &options)
            .unwrap()
            .question_id()
    );
    assert!(
        prepare_query_plan_on(
            original,
            &tx,
            "self",
            &Bindings::default(),
            None,
            None,
            None
        )
        .is_err()
    );
    let new_plan =
        polylogue_query::plan::prepare("sessions where repo:new", &Bindings::default(), |_| Ok(()))
            .unwrap();
    crate::assertions::apply(
        &tx,
        &[definition(
            "new-pending",
            "new",
            "sessions where repo:new",
            new_plan.question_id(),
        )],
        2,
    )
    .unwrap();
    let mut provisional = ProvisionalQueries::default();
    provisional.record(&new_plan).unwrap();
    assert_eq!(
        provisional.object(new_plan.question_id()).unwrap().id,
        new_plan.question_id()
    );
    assert!(provisional.object(QuestionId([0; 32])).is_none());
    let by_hash = format!("query:{}", new_plan.question_id());
    let from_prepared = prepare_query_plan_on(
        original,
        &tx,
        &by_hash,
        &Bindings::default(),
        None,
        None,
        Some(&provisional),
    )
    .unwrap();
    assert_eq!(
        from_prepared.definitions()[0].question_id,
        new_plan.question_id()
    );
    assert_eq!(
        prepare_query_plan_on(
            original,
            &tx,
            &by_hash,
            &Bindings::default(),
            None,
            None,
            None
        )
        .unwrap_err()
        .kind(),
        "not_yet_available"
    );
    let prepared = prepare_query_plan_on(
        original,
        &tx,
        "query:new",
        &Bindings::default(),
        None,
        None,
        Some(&provisional),
    )
    .unwrap();
    assert_eq!(
        prepared.definitions()[0].question_id,
        new_plan.question_id()
    );
    assert_eq!(provisional.objects().count(), 1);
    assert_eq!(
        prepare_query_plan_on(
            original,
            &tx,
            "query:new",
            &Bindings::default(),
            None,
            None,
            None
        )
        .unwrap_err()
        .kind(),
        "not_yet_available"
    );
    prepare_query_plan_on(
        original,
        &tx,
        "sessions",
        &Bindings::default(),
        None,
        None,
        None,
    )
    .unwrap();
}
