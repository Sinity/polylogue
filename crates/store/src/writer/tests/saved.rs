use super::*;
use polylogue_model::{
    BlobHash,
    annotation::{AnnotationKind, NewAnnotation, values::SavedQuery},
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
        UserWrite::Save(Box::new(polylogue_model::user::Save {
            name: Some(name.into()),
            query: text.into(),
            denominator: None,
            rows_json: r#"{"unit":"sessions","columns":[],"rows":[]}"#.into(),
            read_at: "{}".into(),
            watch: None,
            outcomes: None,
        })),
    )
}

fn definition(name: &str, text: &str, plan: &polylogue_query::plan::Plan) -> NewAnnotation {
    NewAnnotation {
        kind: AnnotationKind::SavedQuery,
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
fn saved_acceptance_keeps_objects_and_dependencies() {
    let archive = test_archive();
    let mut saved_bindings = bindings();
    let alpha = plan("sessions", &saved_bindings);
    saved_bindings
        .add_saved("alpha", &alpha, "batch.1", "batch.1")
        .unwrap();
    let beta = plan("query:alpha", &saved_bindings);
    let batch = request(
        "batch",
        UserWrite::Assert {
            annotations: vec![
                definition("alpha", "sessions", &alpha),
                definition("beta", "query:alpha", &beta),
            ],
        },
    );
    archive
        .writer
        .user(crate::writer::UserSubmission {
            request: batch.clone(),
            scope: None,
            preparation: Some(context()),
            parent: None,
        })
        .unwrap();
    let reads = ReadPool::open(&archive.archive).unwrap();
    for expected in [&alpha, &beta] {
        let kept = reads
            .reader()
            .unwrap()
            .query_object(&expected.question_id().to_string())
            .unwrap()
            .unwrap();
        assert_eq!(kept, expected.query_object().unwrap());
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
        )
        .unwrap()
        .query_object()
        .unwrap()
    };
    archive
        .writer
        .user(crate::writer::UserSubmission {
            request: save("changed-alpha", "alpha", "messages"),
            scope: None,
            preparation: Some(context()),
            parent: None,
        })
        .unwrap();
    let reader = reads.reader().unwrap();
    let saved_plan = crate::selection::prepare_query_plan_on(
        reader.conn().unwrap(),
        reader.conn().unwrap(),
        "query:beta",
        &bindings(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(saved_plan.query_object().unwrap(), before);
    drop(reader);
    // A retry reuses the committed batch, even after its dependencies change
    // and without a fresh preparation catalog.
    assert!(archive.writer.user(batch).unwrap().repeat);
}

#[test]
fn invalid_saved_batch_acks_no_objects_and_sends_no_assertions() {
    let archive = test_archive();
    let alpha = plan("sessions", &bindings());
    let mut wrong = definition("beta", "messages", &alpha);
    wrong.value.as_mut().unwrap()["hash"] = serde_json::to_value(BlobHash([0; 32])).unwrap();
    assert!(
        archive
            .writer
            .user(crate::writer::UserSubmission {
                request: request(
                    "invalid-batch",
                    UserWrite::Assert {
                        annotations: vec![definition("alpha", "sessions", &alpha), wrong],
                    },
                ),
                scope: None,
                preparation: Some(context()),
                parent: None
            })
            .is_err()
    );
    let reads = ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    assert!(reader.user_request("invalid-batch").unwrap().is_none());
    assert!(
        reader
            .query_object(&alpha.question_id().to_string())
            .unwrap()
            .is_none()
    );
    assert!(
        crate::annotations::saved_query_snapshot(reader.conn().unwrap(), ["alpha", "beta"])
            .unwrap()
            .definitions
            .is_empty()
    );
}

#[test]
fn missing_preparation_context_refuses_new_writes_only() {
    let archive = test_archive();
    let original = save("saved", "everything", "sessions");
    assert!(matches!(
        archive.writer.user(original.clone()),
        Err(StoreError::ReferenceUnavailable { .. })
    ));
    archive
        .writer
        .user(crate::writer::UserSubmission {
            request: original.clone(),
            scope: None,
            preparation: Some(context()),
            parent: None,
        })
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
fn saved_figure_keeps_its_denominator_as_a_separate_question() {
    let archive = test_archive();
    let figure = plan("sessions | count", &bindings());
    let population = plan("sessions", &bindings());
    let missing = save("figure", "total", "sessions | count");
    assert!(matches!(
        archive.writer.user(crate::writer::UserSubmission {
            request: missing,
            scope: None,
            preparation: Some(context()),
            parent: None
        }),
        Err(StoreError::Query(
            polylogue_query::QueryError::Invalid { .. }
        ))
    ));
    archive
        .writer
        .user(crate::writer::UserSubmission {
            request: request(
                "figure",
                UserWrite::Save(Box::new(polylogue_model::user::Save {
                    name: Some("total".into()),
                    query: "sessions | count".into(),
                    denominator: Some("sessions".into()),
                    rows_json: r#"{"unit":"sessions","columns":[],"rows":[]}"#.into(),
                    read_at: "{}".into(),
                    watch: None,
                    outcomes: None,
                })),
            ),
            scope: None,
            preparation: Some(context()),
            parent: None,
        })
        .unwrap();
    let reads = ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    let annotation = reader
        .annotation(&polylogue_model::RevisionReference {
            root: "figure".into(),
            revision: Some("figure".into()),
        })
        .unwrap()
        .unwrap();
    let value: SavedQuery = serde_json::from_value(annotation.fact.value.unwrap()).unwrap();
    assert_eq!(value.hash, BlobHash(figure.question_id().0));
    assert_eq!(
        value.denominator,
        Some(format!("query:{}", population.question_id()))
    );
    for expected in [figure, population] {
        assert!(
            reader
                .query_object(&expected.question_id().to_string())
                .unwrap()
                .is_some()
        );
    }
}

#[test]
fn kept_names_follow_run_time_and_exact_opaque_ids_stay_addressable() {
    let archive = test_archive();
    for (id, at, name) in [
        ("z-older", 100, "example"),
        ("a-newer", 200, "example"),
        ("example", 50, "other"),
    ] {
        let mut request = save(id, name, "sessions");
        request.at = polylogue_model::Timestamp::from_unix_ms(at).unwrap();
        archive
            .writer
            .user(crate::writer::UserSubmission {
                request,
                scope: None,
                preparation: Some(context()),
                parent: None,
            })
            .unwrap();
    }
    let reads = ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    let prepare = |text| {
        crate::selection::prepare_query_plan_on(
            reader.conn().unwrap(),
            reader.conn().unwrap(),
            text,
            &bindings(),
            None,
            None,
        )
        .unwrap()
    };
    let newest = prepare("result:other");
    assert_eq!(
        newest
            .bindings()
            .kept_names
            .get("other")
            .map(String::as_str),
        Some("example")
    );
    // An exact ID wins over a colliding name, even when that name has newer runs.
    let exact = prepare("result:example");
    assert!(!exact.bindings().kept_names.contains_key("example"));
    drop(reader);
    // Use a separate name to exercise newest ordering without that collision.
    for (id, at) in [("z-first", 100), ("a-last", 200)] {
        let mut request = save(id, "chronological", "sessions");
        request.at = polylogue_model::Timestamp::from_unix_ms(at).unwrap();
        archive
            .writer
            .user(crate::writer::UserSubmission {
                request,
                scope: None,
                preparation: Some(context()),
                parent: None,
            })
            .unwrap();
    }
    let reader = reads.reader().unwrap();
    let newest = crate::selection::prepare_query_plan_on(
        reader.conn().unwrap(),
        reader.conn().unwrap(),
        "result:chronological",
        &bindings(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        newest
            .bindings()
            .kept_names
            .get("chronological")
            .map(String::as_str),
        Some("a-last")
    );
}
