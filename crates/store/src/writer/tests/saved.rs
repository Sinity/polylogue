use super::*;
use polylogue_model::{
    BlobHash,
    assertion::{AssertionKind, NewAssertion, values::SavedQuery},
    user::{UserRequest, UserWrite},
};

fn context() -> crate::user_writes::PreparationContext {
    crate::user_writes::PreparationContext::new(vec!["codex".into()])
}

fn bindings() -> polylogue_query::Bindings {
    polylogue_query::Bindings {
        origins: vec!["codex".into()],
        ..Default::default()
    }
}

fn plan(text: &str, bindings: &polylogue_query::Bindings) -> polylogue_query::plan::Plan {
    polylogue_query::plan::prepare(text, bindings, |_| Ok(())).unwrap()
}

fn request(id: &str, write: UserWrite) -> UserRequest {
    crate::fixture::authored::operator_request(id.into(), write)
}

fn save(id: &str, name: &str, text: &str) -> UserRequest {
    request(
        id,
        UserWrite::SaveQuery {
            name: name.into(),
            text: text.into(),
            denominator: None,
        },
    )
}

fn definition(name: &str, text: &str, plan: &polylogue_query::plan::Plan) -> NewAssertion {
    NewAssertion {
        kind: AssertionKind::SavedQuery,
        key: Some(name.into()),
        value: Some(
            serde_json::to_value(SavedQuery {
                text: text.into(),
                hash: BlobHash(plan.question_id().0),
                denominator: None,
            })
            .unwrap(),
        ),
        valid_until: None,
        links: Vec::new(),
    }
}

#[test]
fn saved_acceptance_retains_objects_and_frozen_dependencies_before_publishing_assertions() {
    let archive = test_archive();
    let mut native = bindings();
    let alpha = plan("sessions", &native);
    native
        .add_saved("alpha", &alpha, "batch.1", "batch.1")
        .unwrap();
    let beta = plan("query:alpha", &native);
    let batch = request(
        "batch",
        UserWrite::Assert {
            assertions: vec![
                definition("alpha", "sessions", &alpha),
                definition("beta", "query:alpha", &beta),
            ],
        },
    );
    archive
        .writer
        .user_preparing(batch.clone(), context(), None)
        .unwrap();
    let reads = ReadPool::open(&archive.archive).unwrap();
    for expected in [&alpha, &beta] {
        let retained = reads
            .reader()
            .unwrap()
            .query_object(&expected.question_id().to_string(), None)
            .unwrap()
            .unwrap();
        assert_eq!(retained.value, expected.query_object().unwrap());
    }
    let before = {
        let reader = reads.reader().unwrap();
        crate::selection::prepare_query_plan_on(
            reader.conn().unwrap(),
            reader.conn().unwrap(),
            "query:beta",
            &bindings(),
            None,
            None,
            None,
        )
        .unwrap()
        .query_object()
        .unwrap()
    };
    archive
        .writer
        .user_preparing(save("changed-alpha", "alpha", "messages"), context(), None)
        .unwrap();
    let reader = reads.reader().unwrap();
    let frozen = crate::selection::prepare_query_plan_on(
        reader.conn().unwrap(),
        reader.conn().unwrap(),
        "query:beta",
        &bindings(),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(frozen.query_object().unwrap(), before);
    drop(reader);
    // A retry reuses the admitted batch, even after its dependencies change
    // and without a fresh preparation catalog.
    assert!(archive.writer.user(batch).unwrap().repeat);
}

#[test]
fn invalid_ordered_saved_batch_does_not_ack_objects_or_publish_partial_assertions() {
    let archive = test_archive();
    let alpha = plan("sessions", &bindings());
    let mut wrong = definition("beta", "messages", &alpha);
    wrong.value.as_mut().unwrap()["hash"] = serde_json::to_value(BlobHash([0; 32])).unwrap();
    assert!(
        archive
            .writer
            .user_preparing(
                request(
                    "invalid-batch",
                    UserWrite::Assert {
                        assertions: vec![definition("alpha", "sessions", &alpha), wrong],
                    },
                ),
                context(),
                None,
            )
            .is_err()
    );
    let reads = ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    assert!(reader.user_request("invalid-batch").unwrap().is_none());
    assert!(
        reader
            .query_object(&alpha.question_id().to_string(), None)
            .unwrap()
            .is_none()
    );
    assert!(
        crate::assertions::saved_query_snapshot(reader.conn().unwrap(), ["alpha", "beta"])
            .unwrap()
            .definitions
            .is_empty()
    );
}

#[test]
fn missing_saved_preparation_context_refuses_new_writes_but_preserves_frozen_retries() {
    let archive = test_archive();
    let original = save("saved", "everything", "sessions");
    assert!(matches!(
        archive.writer.user(original.clone()),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    archive
        .writer
        .user_preparing(original.clone(), context(), None)
        .unwrap();
    assert!(archive.writer.user(original).unwrap().repeat);
    archive
        .writer
        .user(request(
            "ordinary-setting",
            UserWrite::SetSetting {
                key: "browser.actions".into(),
                value: "on".into(),
                machine: None,
            },
        ))
        .unwrap();
}

#[test]
fn saved_figure_prepares_and_retains_its_denominator_as_a_separate_question() {
    let archive = test_archive();
    let figure = plan("sessions | count", &bindings());
    let population = plan("sessions", &bindings());
    let missing = save("figure", "total", "sessions | count");
    assert!(matches!(
        archive.writer.user_preparing(missing, context(), None),
        Err(StoreError::Query(
            polylogue_query::QueryError::Invalid { .. }
        ))
    ));
    archive
        .writer
        .user_preparing(
            request(
                "figure",
                UserWrite::SaveQuery {
                    name: "total".into(),
                    text: "sessions | count".into(),
                    denominator: Some("sessions".into()),
                },
            ),
            context(),
            None,
        )
        .unwrap();
    let reads = ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    let fact = reader
        .assertion(&polylogue_model::RevisionReference {
            root: "figure".into(),
            revision: Some("figure".into()),
        })
        .unwrap()
        .unwrap();
    let value: SavedQuery = serde_json::from_value(fact.fact.value.unwrap()).unwrap();
    assert_eq!(value.hash, BlobHash(figure.question_id().0));
    assert_eq!(
        value.denominator,
        Some(format!("query:{}", population.question_id()))
    );
    for expected in [figure, population] {
        assert!(
            reader
                .query_object(&expected.question_id().to_string(), None)
                .unwrap()
                .is_some()
        );
    }
}
