use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
use polylogue_model::authorship::{CallerEvidence, Channel, compute_authorship};

fn options() -> Options {
    Options {
        clock: Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("Etc/UTC").unwrap(),
                jiff::civil::Time::midnight(),
            )
            .unwrap(),
        ),
        origins: vec!["test".into()],
        command_words: vec![],
        page_size: Some(1),
        random_seed: Some(7),
        embeddings: false,
        self_session: Some("test:caller".into()),
        saved: Default::default(),
        fields: vec!["id".into()],
        kept: Default::default(),
        chat_pages: vec![],
    }
}
#[test]
fn execution_binding_replays_captured_options_and_refuses_a_changed_basis() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:session",
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            session,
            MessageFacts {
                id: "test:message",
                role: "user",
                written_by: "human",
                prompt_kind: Some("prose"),
                text: Some("synthetic"),
                ..Default::default()
            },
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let options = options();
    let prepared = reader
        .prepare_selection("messages | sort by id", &options)
        .unwrap();
    let object = prepared.query_object().unwrap();
    let snapshot = crate::query::execute::options_snapshot(
        &options,
        true,
        Some("provider:vector-basis".into()),
    )
    .unwrap();
    let binding = prepared
        .execution_binding(
            snapshot,
            ExecutionPresentation::Query,
            Some(serde_json::json!({"id":"accepted-operation"})),
        )
        .unwrap();
    let run = QueryRun {
        id: "00000000-0000-7000-8000-000000000001".parse().unwrap(),
        question: prepared.question_id(),
        at: options.clock.now.into(),
        read_at: reader.change_position().unwrap(),
        machine: "synthetic-machine".into(),
        authorship: compute_authorship(
            Channel::Cli,
            &CallerEvidence {
                controlling_terminal: true,
                ..Default::default()
            },
            None,
        ),
        binding,
    };
    let replay = reader
        .prepare_query_run_selection(
            &object,
            &run,
            &options.clock,
            Some("provider:vector-basis"),
            Some(99),
        )
        .unwrap();
    let mut rows = vec![];
    reader
        .visit_selection::<SelectionError>(&replay, None, None, |row| {
            rows.push(row.values);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(rows, vec![serde_json::json!({"id":"test:message"})]);
    assert!(run.binding.query_options.why);
    assert_eq!(
        run.binding
            .query_options
            .self_session
            .as_ref()
            .unwrap()
            .to_string(),
        "test:caller"
    );
    assert_eq!(
        run.binding.accepted_request.as_ref().unwrap()["id"],
        "accepted-operation"
    );
    assert_eq!(
        reader
            .prepare_query_run_selection(
                &object,
                &run,
                &options.clock,
                Some("provider:different-basis"),
                None
            )
            .err()
            .unwrap()
            .kind(),
        "stale_cursor"
    );
    let mut changed = run.clone();
    changed
        .binding
        .replay
        .context
        .versions
        .insert("catalog".into(), "retired-registry".into());
    assert_eq!(
        reader
            .prepare_query_run_selection(
                &object,
                &changed,
                &options.clock,
                Some("provider:vector-basis"),
                None
            )
            .err()
            .unwrap()
            .kind(),
        "not_built"
    );
    let mut wrong = run.binding.query_options.clone();
    wrong.fields.push("role".into());
    assert_eq!(
        prepared
            .execution_binding(wrong, ExecutionPresentation::Query, None)
            .err()
            .unwrap()
            .kind(),
        "invalid_query"
    );
}

#[test]
fn acknowledged_direct_range_restores_presentation_without_changing_question_identity() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:session",
            ..Default::default()
        })
        .unwrap();
    for id in ["test:a", "test:b"] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "user",
                    written_by: "human",
                    prompt_kind: Some("prose"),
                    text: Some("synthetic"),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let options = options();
    let direct = reader
        .prepare_selection("message:test:a..test:b", &options)
        .unwrap();
    let semantic = reader
        .prepare_query_selection("message:test:a..test:b", &options)
        .unwrap();
    assert_eq!(direct.question_id(), semantic.question_id());
    let binding = direct
        .execution_binding(
            crate::query::execute::options_snapshot(&options, false, None).unwrap(),
            ExecutionPresentation::Read {
                options: Default::default(),
                delivery: Default::default(),
            },
            None,
        )
        .unwrap();
    assert_eq!(
        binding.query_options.direct_reference,
        direct.direct_reference().cloned()
    );
    let object = direct.query_object().unwrap();
    let run = QueryRun {
        id: "00000000-0000-7000-8000-000000000002".parse().unwrap(),
        question: object.id,
        at: options.clock.now.into(),
        read_at: reader.change_position().unwrap(),
        machine: "synthetic-machine".into(),
        authorship: compute_authorship(
            Channel::Cli,
            &CallerEvidence {
                controlling_terminal: true,
                ..Default::default()
            },
            None,
        ),
        binding,
    };
    let replay = reader
        .prepare_query_run_selection(&object, &run, &options.clock, None, None)
        .unwrap();
    assert_eq!(replay.direct_reference(), direct.direct_reference());
    let mut forged = run.clone();
    forged.binding.query_options.direct_reference =
        Some(Reference::Message("test:a".parse().unwrap()));
    assert_eq!(
        reader
            .prepare_query_run_selection(&object, &forged, &options.clock, None, None)
            .err()
            .unwrap()
            .kind(),
        "conflict"
    );
    let mut old_scope = run;
    old_scope.binding.replay.scope_fingerprint = Some("another-grant".into());
    assert_eq!(
        reader
            .prepare_query_run_selection(&object, &old_scope, &options.clock, None, None)
            .err()
            .unwrap()
            .kind(),
        "scope_changed"
    );
}
