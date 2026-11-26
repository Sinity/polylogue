use super::*;
use crate::fixture::{Fixture, SessionFacts};
use polylogue_model::{
    Author, BlobHash,
    annotation::{AnnotationFact, AnnotationKind, values},
};

fn definition(id: &str, name: &str, text: &str, hash: QuestionId) -> AnnotationFact {
    AnnotationFact {
        id: id.into(),
        root: id.into(),
        kind: AnnotationKind::SavedQuery,
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
        created_at: Some(polylogue_model::Timestamp::from_unix_ms(1).unwrap()),
        valid_until: None,
        links: vec![],
    }
}

#[test]

fn accepting_preparation_uses_original_rows() {
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
    let schema = crate::source::from_archive(&crate::schema::ARCHIVE).unwrap();
    scratch.execute_batch(&schema.ddl).unwrap();
    crate::annotations::apply(
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
        prepare_query_plan_on(original, &tx, &missing, &Bindings::default(), None, None)
            .unwrap_err()
            .code(),
        polylogue_model::Code::NotYetAvailable
    );
    let refused = prepare_query_plan_on(original, &tx, &missing, &Bindings::default(), None, None)
        .unwrap_err();
    let message = refused.to_string();
    let roundtrip = SelectionError::from(refused.into_store_error());
    assert_eq!(roundtrip.code(), polylogue_model::Code::NotYetAvailable);
    assert_eq!(roundtrip.to_string(), message);
    let mut supplied = Bindings {
        origins: vec!["test".into()],
        ..Default::default()
    };
    // Untrusted supplied dynamic definitions cannot replace either stored view.
    let forged =
        polylogue_query::plan::prepare("sessions where repo:forged", &Bindings::default(), |_| {
            Ok(())
        })
        .unwrap();
    supplied
        .add_saved("focus", &forged, "forged", "forged")
        .unwrap();
    let prepared =
        prepare_query_plan_on(original, &tx, "query:focus", &supplied, None, None).unwrap();
    let loaded = named_definitions(original, &tx, "query:focus", &[], None)
        .unwrap()
        .saved;
    assert_eq!(loaded["focus"][0].hash, object.id);
    assert_eq!(loaded["focus"][0].denominator, Some(object.id));
    assert_eq!(prepared.definitions()[0].question_id, object.id);
    let plain =
        prepare_query_plan_on(original, original, "sessions", &supplied, None, None).unwrap();
    let mut options = super::super::tests::options();
    options.origins = supplied.origins;
    assert_eq!(
        plain.question_id(),
        reader
            .prepare_query_plan("sessions", &options)
            .unwrap()
            .question_id()
    );
    assert_eq!(
        prepare_query_plan_on(original, &tx, "self", &Bindings::default(), None, None)
            .unwrap_err()
            .code(),
        polylogue_model::Code::NotFound
    );
    let new_plan =
        polylogue_query::plan::prepare("sessions where repo:new", &Bindings::default(), |_| Ok(()))
            .unwrap();
    crate::annotations::apply(
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
        Some(&provisional),
    )
    .unwrap();
    assert_eq!(
        from_prepared.definitions()[0].question_id,
        new_plan.question_id()
    );
    assert_eq!(
        prepare_query_plan_on(original, &tx, &by_hash, &Bindings::default(), None, None)
            .unwrap_err()
            .code(),
        polylogue_model::Code::NotYetAvailable
    );
    let prepared = prepare_query_plan_on(
        original,
        &tx,
        "query:new",
        &Bindings::default(),
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
        prepare_query_plan_on(original, &tx, "query:new", &Bindings::default(), None, None)
            .unwrap_err()
            .code(),
        polylogue_model::Code::NotYetAvailable
    );
    prepare_query_plan_on(original, &tx, "sessions", &Bindings::default(), None, None).unwrap();
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
        prepare_query_plan_on(original, original, text, &Bindings::default(), None, None)
    };
    let by_hash = prepare(&format!("query:{full}")).unwrap();
    for length in [8, 12, 20, 63] {
        let by_prefix = prepare(&format!("query:{}", &full[..length])).unwrap();
        assert_eq!(
            by_prefix.definitions()[0].question_id,
            object.id,
            "{length}"
        );
        assert_eq!(by_prefix.question_id(), by_hash.question_id(), "{length}");
    }
    // Shorter than 8 characters is not a hash, and no saved query has that name
    // (docs/saved-queries.md: a unique prefix of at least eight hex characters).
    assert_eq!(
        prepare(&format!("query:{}", &full[..7]))
            .unwrap_err()
            .code(),
        polylogue_model::Code::InvalidQuery
    );
    // A prefix no saved question starts with is unavailable.
    let other = if full.starts_with('0') { "1" } else { "0" };
    assert_eq!(
        prepare(&format!("query:{other}{}", &full[1..12]))
            .unwrap_err()
            .code(),
        polylogue_model::Code::NotYetAvailable
    );
}

#[test]
fn numeric_saved_revision_binds_the_original_recorded_question_and_body() {
    use polylogue_model::annotation::{AnnotationLink, LinkRole};
    let directory = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(directory.path()).unwrap();
    let conn = crate::archive::open_archive_writer(&fixture.archive).unwrap();
    let first =
        polylogue_query::plan::prepare("sessions where repo:first", &Bindings::default(), |_| {
            Ok(())
        })
        .unwrap()
        .query_object()
        .unwrap();
    let later =
        polylogue_query::plan::prepare("sessions where repo:later", &Bindings::default(), |_| {
            Ok(())
        })
        .unwrap()
        .query_object()
        .unwrap();
    crate::query_objects::apply_object(&conn, &first, 1).unwrap();
    crate::query_objects::apply_object(&conn, &later, 2).unwrap();
    let first_fact = definition("first", "focus", "sessions where repo:first", first.id);
    let mut later_fact = definition("later", "focus", "sessions where repo:later", later.id);
    later_fact.root = "first".into();
    later_fact.created_at = None;
    later_fact.links.push(AnnotationLink {
        role: LinkRole::Revises,
        reference: Reference::Annotation(RevisionReference::exact("first".into(), "first".into())),
        seen_hash: None,
        quote: None,
    });
    // Arrival order is not revision order.
    crate::annotations::apply(&conn, &[later_fact], 3).unwrap();
    crate::annotations::apply(&conn, &[first_fact], 10).unwrap();
    drop(conn);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let conn = reader.conn().unwrap();
    for (spelling, id, hash) in [
        ("focus@1", "first", first.id),
        ("focus@2", "later", later.id),
        ("focus", "later", later.id),
    ] {
        let text = format!("query:{spelling}");
        let loaded = named_definitions(conn, conn, &text, &[], None)
            .unwrap()
            .saved;
        let definition = &loaded[spelling][0];
        assert_eq!(definition.revision, id);
        assert_eq!(definition.root, "first");
        assert_eq!(definition.hash, hash);
        let SavedQueryBody::Available {
            object: Some(body), ..
        } = &definition.body
        else {
            panic!("original stored body")
        };
        assert_eq!(body.id, hash);
        assert_eq!(
            body.normal_form,
            if id == "first" {
                first.normal_form.clone()
            } else {
                later.normal_form.clone()
            }
        );
        let plan = reader
            .prepare_query_plan(&text, &super::super::tests::options())
            .unwrap();
        let used = plan
            .definitions()
            .iter()
            .find(|used| used.name == spelling)
            .unwrap();
        assert_eq!(used.revision, id);
        assert_eq!(used.question_id, hash);
    }
    assert_eq!(
        reader
            .prepare_query_plan("query:focus@9", &super::super::tests::options())
            .unwrap_err()
            .code(),
        polylogue_model::Code::NotFound
    );
}

/// Two prepared questions sharing a prefix are ambiguous, as stored ones are
/// (docs/references.md, hash prefixes).
#[test]
fn provisional_prefix_shared_by_two_questions_is_ambiguous() {
    let plan =
        polylogue_query::plan::prepare("sessions where repo:new", &Bindings::default(), |_| Ok(()))
            .unwrap();
    let mut provisional = ProvisionalQueries::default();
    provisional.record(&plan).unwrap();
    let prefix = plan.question_id().to_string()[..12].to_owned();
    assert_eq!(
        provisional.by_prefix(&prefix).unwrap(),
        Some(plan.question_id())
    );
    assert_eq!(provisional.by_prefix("ffffffffffff").unwrap(), None);
    let twin = provisional.object(plan.question_id()).unwrap().clone();
    provisional.objects.insert("twin".into(), twin);
    assert_eq!(
        provisional.by_prefix(&prefix).unwrap_err().kind(),
        polylogue_model::Code::Ambiguous
    );
}

#[test]
fn kept_result_ids_are_read_quoted_or_bare() {
    assert_eq!(
        spelled_kept_results(
            r#"join (result:"a:b.c") on id = id | join (result:plain-1) on id = id | result:"q\"x\\y""#
        ),
        ["a:b.c", "plain-1", r#"q"x\y"#]
    );
    assert!(spelled_kept_results(r#"result:"never closed"#).is_empty());
}
