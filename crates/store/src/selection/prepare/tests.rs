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
        via: polylogue_model::authorship::Channel::Import,
        machine: "synthetic-host".into(),
        created_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        valid_until: None,
        links: vec![],
    }
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

    crate::query_objects::apply_object(&archive, &object, 1).unwrap();
    let unnamed =
        polylogue_query::plan::prepare("sessions where repo:unnamed", &Bindings::default(), |_| {
            Ok(())
        })
        .unwrap()
        .query_object()
        .unwrap();
    crate::query_objects::apply_object(&archive, &unnamed, 1).unwrap();
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
        polylogue_query::plan::replay(&unnamed, polylogue_query::plan::catalog_fingerprint())
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
        .code(),
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
    let message = refused.to_string();
    let roundtrip = SelectionError::from(refused.into_store_error());
    assert_eq!(roundtrip.code(), "not_yet_available");
    assert_eq!(roundtrip.to_string(), message);
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
    let loaded = named_definitions(original, &tx, "query:focus", &[], None, None)
        .unwrap()
        .saved;
    assert_eq!(loaded["focus"][0].hash, object.id);
    assert_eq!(loaded["focus"][0].denominator, Some(object.id));
    assert_eq!(actual.definitions()[0].question_id, object.id);
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
        .code(),
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
        .code(),
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

#[test]
fn hash_prefix_resolves() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    let object = polylogue_query::plan::prepare(
        "sessions where repo:prefixed",
        &Bindings::default(),
        |_| Ok(()),
    )
    .unwrap()
    .query_object()
    .unwrap();
    let archive = crate::archive::open_write(&fixture.archive.archive_db(), 1).unwrap();
    crate::query_objects::apply_object(&archive, &object, 1).unwrap();
    drop(archive);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let original = reader.conn().unwrap();
    let full = object.id.to_string();
    let prepare = |text: &str| {
        prepare_query_plan_on(
            original,
            original,
            text,
            &Bindings::default(),
            None,
            None,
            None,
        )
    };
    let by_hash = prepare(&format!("query:{full}")).unwrap();
    for length in [12, 20, 63] {
        let by_prefix = prepare(&format!("query:{}", &full[..length])).unwrap();
        assert_eq!(
            by_prefix.definitions()[0].question_id,
            object.id,
            "{length}"
        );
        assert_eq!(by_prefix.question_id(), by_hash.question_id(), "{length}");
    }
    // Shorter than 12 characters is not a hash, and no saved query has that name.
    assert!(prepare(&format!("query:{}", &full[..11])).is_err());
    // A prefix no retained question starts with is unavailable.
    let other = if full.starts_with('0') { "1" } else { "0" };
    assert_eq!(
        prepare(&format!("query:{other}{}", &full[1..12]))
            .unwrap_err()
            .code(),
        "not_yet_available"
    );
}
