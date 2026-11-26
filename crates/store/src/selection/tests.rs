mod kept;
mod pages;
mod plans;

use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
pub(super) fn options() -> Options {
    Options {
        clock: query_language::time::Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("Europe/Warsaw").unwrap(),
                jiff::civil::time(4, 0, 0, 0),
            )
            .unwrap(),
        ),
        origins: vec!["test".into()],
        page_size: Some(1),
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        here: None,
        seen_marks_of: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        provider_catalog: None,
    }
}
fn save_definition(f: &mut Fixture, name: &str, text: &str) -> String {
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let plan = reader.prepare_query_plan(text, &options()).unwrap();
    drop(reader);
    f.saved_query(name, &plan).unwrap()
}
fn fixture() -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (id, last) in [("test:one", 1), ("test:two", 2), ("test:three", 3)] {
        let session = f
            .session(SessionFacts {
                id,
                last: Some(last),
                ..Default::default()
            })
            .unwrap();
        f.message(
            session,
            MessageFacts {
                id: &format!("{id}:m"),
                role: "assistant",
                written_by: "model",
                text: Some("synthetic body"),
                ..Default::default()
            },
        )
        .unwrap();
    }
    (dir, f)
}
#[test]
fn prepared_selection_streams_total_answer_and_stops_at_callback() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let prepared = reader.prepare_selection("sessions", &options()).unwrap();
    let mut refs = vec![];
    let mut after = None;
    reader
        .visit_selection::<SelectionError>(&prepared, crate::read::VisitFrom::First, None, |row| {
            refs.push(row.target.unwrap().to_string());
            after = Some(row.position);
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
    assert_eq!(refs, ["session:test:three"]);
    reader
        .visit_selection::<SelectionError>(
            &prepared,
            after
                .as_ref()
                .map_or(crate::read::VisitFrom::First, crate::read::VisitFrom::After),
            None,
            |row| {
                refs.push(row.target.unwrap().to_string());
                Ok(ControlFlow::Continue(()))
            },
        )
        .unwrap();
    assert_eq!(
        refs,
        ["session:test:three", "session:test:two", "session:test:one"]
    );
}
fn targets(reader: &crate::Reader<'_>, text: &str, options: &Options) -> Vec<String> {
    let prepared = reader.prepare_selection(text, options).unwrap();
    let mut refs = vec![];
    reader
        .visit_selection::<SelectionError>(&prepared, crate::read::VisitFrom::First, None, |row| {
            refs.push(row.target.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    refs
}
#[test]
fn latest_is_the_newest_session_and_its_identity_ignores_execution() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        targets(&reader, "latest", &options()),
        ["session:test:three"]
    );
    let latest = reader.prepare_selection("latest", &options()).unwrap();
    let mut changed = options();
    changed.page_size = Some(300);
    changed.random_seed = Some(999);
    assert_eq!(
        latest.question_id(),
        reader
            .prepare_selection("latest", &changed)
            .unwrap()
            .question_id()
    );
}
#[test]
fn latest_excludes_the_calling_session() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut caller = options();
    caller.self_session = Some("test:three".into());
    assert_eq!(targets(&reader, "latest", &caller), ["session:test:two"]);
}
#[test]
fn self_without_a_calling_session_is_not_found() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let refused = reader.prepare_selection("self", &options()).err().unwrap();
    assert_eq!(refused.code(), polylogue_model::Code::NotFound);
}
#[test]
fn latest_respects_scope() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    // The newest session that also meets the other conditions, not the
    // newest session overall.
    assert_eq!(
        targets(&reader, "latest AND NOT session:test:three", &options()),
        ["session:test:two"]
    );
    let mut caller = options();
    caller.self_session = Some("test:two".into());
    assert_eq!(
        targets(&reader, "latest AND NOT session:test:three", &caller),
        ["session:test:one"]
    );
}
#[test]
fn prepared_selection_keeps_block_reference() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let target = "block:test:one:m#0";
    let prepared = reader.prepare_selection(target, &options()).unwrap();
    let mut refs = vec![];
    reader
        .visit_selection::<SelectionError>(&prepared, crate::read::VisitFrom::First, None, |row| {
            refs.push(row.target.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(refs, [target]);
}

#[test]
fn selected_reference_survives_and_membership_keeps_offset() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.fields = vec!["title".into()];
    let prepared = reader
        .prepare_selection("sessions | offset 1 | limit 1", &options)
        .unwrap();
    let mut rows = vec![];
    reader
        .visit_selection::<SelectionError>(&prepared, crate::read::VisitFrom::First, None, |row| {
            rows.push(row);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].target.as_ref().unwrap().to_string(),
        "session:test:two"
    );
    assert!(
        reader
            .selection_contains(
                &prepared,
                &Reference::Message(MessageId::parse("test:two:m").unwrap()),
                None,
            )
            .unwrap()
    );
    assert!(
        !reader
            .selection_contains(
                &prepared,
                &Reference::Message(MessageId::parse("test:three:m").unwrap()),
                None,
            )
            .unwrap()
    );
}
#[test]
fn group_identity_excludes_counts_and_unknown_observation_stays_unknown() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let prepared = reader
        .prepare_selection("sessions | group by origin", &options())
        .unwrap();
    let mut rows = vec![];
    reader
        .visit_selection::<SelectionError>(&prepared, crate::read::VisitFrom::First, None, |row| {
            rows.push(row);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].identity,
        Some(RowIdentity::Group {
            keys: BTreeMap::from([("origin".into(), Value::String("test".into()))])
        })
    );
    assert_eq!(
        rows[0].aggregates,
        BTreeMap::from([("count".into(), Some(3.))])
    );
    assert!(rows[0].target.is_none());
}
#[test]
fn pure_bindings_preserve_closed_group_values_without_a_clock() {
    let mut bindings = polylogue_query::Bindings::default();
    let plan =
        polylogue_query::plan::prepare("tool_calls | group by outcome", &bindings, |_| Ok(()))
            .unwrap();
    bindings
        .add_saved("by-outcome", &plan, "rev", "root")
        .unwrap();
    let schema = bindings.output_schema("query:by-outcome").unwrap();
    let outcome = schema
        .fields
        .iter()
        .find(|field| field.name == "outcome")
        .unwrap();
    assert_eq!(outcome.kind, query_language::catalog::FieldKind::Exact);
    assert!(outcome.values.iter().any(|value| value == "error"));
    let kept = polylogue_query::compile::columns_schema(
        "tool_calls",
        &[polylogue_query::Column {
            name: "outcome".into(),
            kind: polylogue_query::ColKind::Text,
            reference: None,
        }],
    );
    assert_eq!(
        kept.fields[0].kind,
        query_language::catalog::FieldKind::Text
    );
    assert!(kept.fields[0].values.is_empty());
}

#[test]
fn no_hook_or_activity_events_means_unknown_live_state() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    f.session(SessionFacts {
        id: "test:timeless",
        ..Default::default()
    })
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.fields = vec!["id".into(), "live".into()];
    let selected = reader.prepare_selection("sessions", &options).unwrap();
    let mut rows = vec![];
    reader
        .visit_selection::<SelectionError>(&selected, crate::read::VisitFrom::First, None, |row| {
            rows.push(row);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(rows[0].values["live"], "unknown");
}
#[test]
fn saved_aliases_bind_to_definition_hashes_and_keep_revision_metadata() {
    let (_dir, mut f) = fixture();
    let first = save_definition(&mut f, "alpha", "sessions where origin:test");
    let second = save_definition(&mut f, "beta", "sessions where origin:test");
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let a = reader.prepare_selection("query:alpha", &options()).unwrap();
    let b = reader.prepare_selection("query:beta", &options()).unwrap();
    assert_eq!(a.question_id(), b.question_id());
    assert_eq!(a.definitions()[0].revision, first);
    assert_eq!(b.definitions()[0].revision, second);
    assert!(
        matches!(&a.bound_query().query().source,polylogue_query::ast::Source::Saved{name,..} if name=="alpha")
    );
}

/// docs/saved-queries.md, Identity and Saved definitions: a question names
/// a saved query by the hash it resolved to, so a new revision of the name
/// never changes what a prepared question runs.
#[test]
fn a_saved_query_used_as_a_condition_runs_the_body_its_plan_identity_hashed() {
    let (_dir, mut f) = fixture();
    let first = save_definition(&mut f, "mine", "sessions where id:test:one");
    let ids = |reader: &crate::Reader<'_>, plan: &polylogue_query::plan::Plan| {
        let program = reader.lower_plan(plan, &options()).unwrap();
        let mut ids = reader
            .query_rows(&program)
            .unwrap()
            .into_iter()
            .map(|row| row["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        ids.sort();
        ids
    };
    let questions = [
        "sessions where query:mine",
        "sessions where query:mine OR id:test:three",
        "sessions where NOT query:mine",
    ];
    let plans = {
        let pool = f.reads().unwrap();
        let reader = pool.reader().unwrap();
        questions.map(|text| reader.prepare_query_plan(text, &options()).unwrap())
    };
    let revised = {
        let pool = f.reads().unwrap();
        let reader = pool.reader().unwrap();
        reader
            .prepare_query_plan("sessions where id:test:two", &options())
            .unwrap()
    };
    f.saved_query_revision("mine", &revised, Some((&first, &first)))
        .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let expected = [
        vec!["test:one"],
        vec!["test:one", "test:three"],
        vec!["test:three", "test:two"],
    ];
    for ((text, plan), expected) in questions.iter().zip(&plans).zip(&expected) {
        assert_eq!(&ids(&reader, plan), expected, "{text}");
        // The name now resolves to the revision, and a new question runs it.
        let current = reader.prepare_query_plan(text, &options()).unwrap();
        assert_ne!(current.question_id(), plan.question_id(), "{text}");
    }
    let current = reader.prepare_query_plan(questions[0], &options()).unwrap();
    assert_eq!(ids(&reader, &current), ["test:two"]);
}

#[test]
fn model_authored_user_task_does_not_answer_a_turn() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "test:turn",
            ..Default::default()
        })
        .unwrap();
    f.message(
        s,
        MessageFacts {
            id: "test:turn:human",
            role: "user",
            written_by: "human",
            prompt_kind: Some("prose"),
            at: Some(1000),
            ..Default::default()
        },
    )
    .unwrap();
    f.message(
        s,
        MessageFacts {
            id: "test:turn:task",
            role: "user",
            written_by: "model",
            prompt_kind: Some("model"),
            at: Some(2000),
            ..Default::default()
        },
    )
    .unwrap();
    let conn = crate::archive::open_write(&f.archive.index_db(), 1).unwrap();
    conn.execute(
        "UPDATE message SET parent_id='test:turn:human' WHERE id='test:turn:task'",
        [],
    )
    .unwrap();
    drop(conn);
    let options = Options {
        fields: vec![
            "id".into(),
            "answer".into(),
            "latency".into(),
            "attempts".into(),
        ],
        ..options()
    };
    let read = |f: &Fixture| {
        let pool = f.reads().unwrap();
        let reader = pool.reader().unwrap();
        let mut rows = vec![];
        let selection = reader.prepare_selection("turns", &options).unwrap();
        reader
            .visit_selection::<SelectionError>(
                &selection,
                crate::read::VisitFrom::First,
                None,
                |row| {
                    rows.push(row.values);
                    Ok(ControlFlow::Continue(()))
                },
            )
            .unwrap();
        rows
    };
    let before = read(&f);
    assert_eq!(before.len(), 1);
    assert!(before[0]["answer"].is_null());
    assert!(before[0]["latency"].is_null());
    assert_eq!(before[0]["attempts"], 0);
    f.message(
        s,
        MessageFacts {
            id: "test:turn:answer",
            role: "assistant",
            written_by: "model",
            at: Some(3000),
            ..Default::default()
        },
    )
    .unwrap();
    let conn = crate::archive::open_write(&f.archive.index_db(), 1).unwrap();
    conn.execute(
        "UPDATE message SET parent_id='test:turn:human' WHERE id='test:turn:answer'",
        [],
    )
    .unwrap();
    drop(conn);
    let after = read(&f);
    assert_eq!(after[0]["answer"], "test:turn:answer");
    assert_eq!(after[0]["latency"], 2000);
    assert_eq!(after[0]["attempts"], 1);
}

#[test]
fn annotation_reference_keeps_its_family_and_revision() {
    let (_dir, mut fixture) = fixture();
    let first = fixture
        .note("note", None, Some("first"), "session:test:one")
        .unwrap();
    let second = fixture
        .note("note", None, Some("second"), "session:test:one")
        .unwrap();
    // A fork with no revises link: both revisions of `first`'s family stay
    // current. The stored current-revision index is recomputed as a writer would.
    let conn = crate::archive::open_write(&fixture.archive.archive_db(), 1).unwrap();
    conn.execute(
        "UPDATE annotation SET root=?1 WHERE id=?2",
        rusqlite::params![first, second],
    )
    .unwrap();
    crate::annotations::current_revisions::recompute(&conn, [first.clone(), second.clone()].into())
        .unwrap();
    drop(conn);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.fields = vec!["value".into()];
    let selected = reader
        .prepare_selection("annotations where current:* | sort by id", &options)
        .unwrap();
    let mut rows = vec![];
    reader
        .visit_selection::<SelectionError>(&selected, crate::read::VisitFrom::First, None, |row| {
            rows.push((row.target.unwrap().to_string(), row.values));
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(
        rows,
        vec![
            (
                format!("annotation:{first}@{first}"),
                serde_json::json!({"value":"first"})
            ),
            (
                format!("annotation:{first}@{second}"),
                serde_json::json!({"value":"second"})
            )
        ]
    );
    let exact = reader
        .prepare_selection(&format!("annotation:{first}@{first}"), &options)
        .unwrap();
    let mut exact_values = vec![];
    reader
        .visit_selection::<SelectionError>(&exact, crate::read::VisitFrom::First, None, |row| {
            exact_values.push(row.values);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(exact_values, vec![serde_json::json!({"value":"first"})]);
    assert_eq!(
        reader
            .prepare_selection(&format!("annotation:{first}"), &options)
            .err()
            .unwrap()
            .code(),
        polylogue_model::Code::Ambiguous
    );
    let family = reader
        .prepare_selection(
            &format!("annotations where annotation:{first} | sort by id"),
            &options,
        )
        .unwrap();
    assert!(family.direct_target().is_none());
    let mut count = 0;
    reader
        .visit_selection::<SelectionError>(&family, crate::read::VisitFrom::First, None, |_| {
            count += 1;
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn direct_reads_keep_opaque_ids_and_ranges_whole() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:session",
            ..Default::default()
        })
        .unwrap();
    let opaque_id = "test:native with | sort by id\nquote\"slash\\tab\t";
    let body = "exact stored body\nquotes \"and\" slash\\ and Łódź";
    for id in ["test:a", "test:b", opaque_id] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "user",
                    written_by: "human",
                    prompt_kind: Some("prose"),
                    text: Some(body),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    fixture.event(session, "synthetic", None).unwrap();
    save_definition(
        &mut fixture,
        "literal-target",
        "messages where message:test:a",
    );
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let options = options();
    let reference = Reference::Message(MessageId::parse(opaque_id).unwrap());
    let opaque = reader
        .prepare_selection(&reference.to_string(), &options)
        .unwrap();
    assert_eq!(
        opaque.direct_target().map(ReadTarget::reference),
        Some(reference.clone())
    );
    let messages = reader
        .messages(&SessionId::parse("test:session").unwrap(), false)
        .unwrap();
    let message = messages
        .iter()
        .find(|message| message.id.to_string() == opaque_id)
        .unwrap();
    assert_eq!(message.text(), body);
    let quoted = reader
        .prepare_selection(
            &reference_atom(&ReadTarget::try_from(reference.clone()).unwrap()),
            &options,
        )
        .unwrap();
    assert_eq!(
        quoted.direct_target().map(ReadTarget::reference),
        Some(reference.clone())
    );
    assert_eq!(opaque.question_id(), quoted.question_id());
    let explicit = reader
        .prepare_selection(
            &format!(
                "messages where {} | sort by id",
                reference_atom(&ReadTarget::try_from(reference.clone()).unwrap())
            ),
            &options,
        )
        .unwrap();
    assert!(explicit.direct_target().is_none());
    for selection in [&opaque, &quoted, &explicit] {
        let mut found = Vec::new();
        reader
            .visit_selection::<SelectionError>(
                selection,
                crate::read::VisitFrom::First,
                None,
                |row| {
                    assert_eq!(row.values["id"], opaque_id);
                    found.push(row.target.unwrap().reference());
                    Ok(ControlFlow::Continue(()))
                },
            )
            .unwrap();
        assert_eq!(found.as_slice(), std::slice::from_ref(&reference));
    }
    let missing = Reference::Message(MessageId::parse(&format!("{opaque_id}missing")).unwrap());
    for target in [
        missing.to_string(),
        reference_atom(&ReadTarget::try_from(missing.clone()).unwrap()),
    ] {
        assert_eq!(
            reader
                .prepare_selection(&target, &options)
                .err()
                .unwrap()
                .code(),
            polylogue_model::Code::NotFound
        );
    }
    let saved = reader
        .prepare_selection("query:literal-target", &options)
        .unwrap();
    assert!(saved.direct_target().is_none());
    let mut saved_ids = Vec::new();
    reader
        .visit_selection::<SelectionError>(&saved, crate::read::VisitFrom::First, None, |row| {
            saved_ids.push(row.target.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(saved_ids, ["message:test:a"]);
    let object = opaque.query_object().unwrap();
    polylogue_query::plan::validate_object(&object).unwrap();
    let event = reader.prepare_selection("event:e4", &options).unwrap();
    assert_eq!(event.direct_target(), Some(&ReadTarget::Event("e4".into())));
    assert_eq!(
        reader
            .prepare_selection("block:test:a#999", &options)
            .err()
            .unwrap()
            .code(),
        polylogue_model::Code::NotFound
    );
    let sorted = reader
        .prepare_selection(
            "messages where message:test:a..test:b | sort by id",
            &options,
        )
        .unwrap();
    assert!(sorted.direct_target().is_none());
    let mut selected = vec![];
    reader
        .visit_selection::<SelectionError>(&sorted, crate::read::VisitFrom::First, None, |row| {
            selected.push(row.target.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(selected, ["message:test:a", "message:test:b"]);
    let query_rows = reader
        .prepare_query_selection("messages where message:test:a..test:b", &options)
        .unwrap();
    assert!(query_rows.direct_target().is_none());
    let mut query_count = 0;
    reader
        .visit_selection::<SelectionError>(&query_rows, crate::read::VisitFrom::First, None, |_| {
            query_count += 1;
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(query_count, 2);
    let range = reader
        .prepare_selection("message:test:a..test:b", &options)
        .unwrap();
    assert!(matches!(
        range.direct_target(),
        Some(ReadTarget::MessageRange { .. })
    ));
}

#[test]
fn direct_authored_reference_uses_family_revision_membership() {
    let (_dir, mut fixture) = fixture();
    let first = fixture
        .note("lesson", None, Some("first"), "session:test:one")
        .unwrap();
    let second = fixture
        .note("lesson", None, Some("second"), "session:test:two")
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let reference = format!("annotation:{first}@{first}");
    let prepared = reader.prepare_selection(&reference, &options()).unwrap();
    let mut selected = Vec::new();
    reader
        .visit_selection::<SelectionError>(&prepared, crate::read::VisitFrom::First, None, |row| {
            selected.push(row.target.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(selected, [reference]);
    let missing = format!("annotation:{first}@{second}");
    assert!(
        matches!(reader.prepare_selection(&missing, &options()), Err(SelectionError::Invalid { kind, .. }) if kind == polylogue_model::Code::NotFound)
    );
    let wrong_family = reader
        .prepare_query_selection(&missing, &options())
        .unwrap();
    let mut count = 0;
    reader
        .visit_selection::<SelectionError>(
            &wrong_family,
            crate::read::VisitFrom::First,
            None,
            |_| {
                count += 1;
                Ok(ControlFlow::Continue(()))
            },
        )
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn event_read_target_uses_event_grain() {
    use crate::fixture::writer::WriterArchive;
    use polylogue_model::{Event, Output, TimeQuality};
    let mut archive = WriterArchive::new();
    let output = Output {
        events: vec![Event {
            session: None,
            kind: "synthetic_event".into(),
            at: None,
            at_quality: TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            after: None,
            range: Some((0, 1)),
            data: serde_json::json!({"synthetic":true}),
        }],
        ..Default::default()
    };
    archive.write_output(
        "synthetic-event",
        "claude-code",
        "synthetic/event",
        output,
        1,
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let query_rows = reader
        .prepare_query_selection("events", &options())
        .unwrap();
    let mut references = vec![];
    reader
        .visit_selection::<SelectionError>(
            &query_rows,
            crate::read::VisitFrom::First,
            None,
            |row| {
                references.push(row.target.unwrap());
                Ok(ControlFlow::Continue(()))
            },
        )
        .unwrap();
    assert_eq!(references.len(), 1);
    assert!(matches!(references[0], ReadTarget::Event(_)));
    let target = references[0].to_string();
    let direct = reader.prepare_selection(&target, &options()).unwrap();
    assert_eq!(direct.program().unit, Unit::Events);
    assert_eq!(direct.direct_target(), Some(&references[0]));
    let mut selected = vec![];
    reader
        .visit_selection::<SelectionError>(&direct, crate::read::VisitFrom::First, None, |row| {
            selected.push(row.target.unwrap());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(selected, references);
    let plan = reader.prepare_query_plan(&target, &options()).unwrap();
    assert!(matches!(&plan.bound_query().query().source,
        polylogue_query::ast::Source::Unit { unit, implicit: true, .. } if unit == "sessions"));
}

#[test]
fn annotation_values_are_decoded_once_and_predicates_match_recorded_text() {
    let (_dir, mut fixture) = fixture();
    for text in ["", "quoted \"body\"\nnext"] {
        fixture
            .note("note", None, Some(text), "session:test:one")
            .unwrap();
    }
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut opts = options();
    opts.fields = vec!["value".into()];
    let selected = reader
        .prepare_query_selection("annotations | sort by id", &opts)
        .unwrap();
    let mut values = vec![];
    reader
        .visit_selection::<SelectionError>(&selected, crate::read::VisitFrom::First, None, |row| {
            values.push(row.values["value"].clone());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(
        values,
        vec![
            serde_json::json!(""),
            serde_json::json!("quoted \"body\"\nnext")
        ]
    );
    let selected = reader
        .prepare_query_selection("annotations where value:quoted", &opts)
        .unwrap();
    let mut values = vec![];
    reader
        .visit_selection::<SelectionError>(&selected, crate::read::VisitFrom::First, None, |row| {
            values.push(row.values["value"].clone());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(values, vec![serde_json::json!("quoted \"body\"\nnext")]);
}

/// The prepared selection that annotation writes use and the listing the query
/// command shows must select the same rows for one question.
#[test]
fn selection_and_listing_agree_when_every_row_is_committed() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.page_size = Some(10);
    let selected = |text: &str| {
        let prepared = reader.prepare_query_selection(text, &options).unwrap();
        let mut targets = vec![];
        reader
            .visit_selection::<SelectionError>(
                &prepared,
                crate::read::VisitFrom::First,
                None,
                |row| {
                    targets.push(row.target.unwrap().to_string());
                    Ok(ControlFlow::Continue(()))
                },
            )
            .unwrap();
        targets
    };
    let listed = |text: &str| {
        let plan = reader.prepare_query_plan(text, &options).unwrap();
        let program = crate::query::lower(&plan, &options).unwrap();
        reader
            .query_page(&program, None)
            .unwrap()
            .rows
            .iter()
            .map(|row| row_target(&program, row).unwrap().to_string())
            .collect::<Vec<_>>()
    };
    for (text, expected) in [
        (
            "messages where id:\"test:two:m\" active:* duplicate:* hidden:*",
            vec!["message:test:two:m"],
        ),
        (
            "messages where message:\"test:two:m\"",
            vec!["message:test:two:m"],
        ),
        // A typed reference is not a native ID; the message reference form
        // above is how a caller holding a reference names the message.
        (
            "messages where id:\"message:test:two:m\" active:* duplicate:* hidden:*",
            vec![],
        ),
    ] {
        assert_eq!(selected(text), expected, "{text}");
        assert_eq!(listed(text), expected, "{text}");
    }
}

#[test]
fn a_selection_failure_carries_the_store_code_every_surface_uses() {
    let cases = [
        (
            StoreError::Conflict("request-1".into()),
            polylogue_model::Code::Conflict,
        ),
        (
            StoreError::ReadIndexReplaced,
            polylogue_model::Code::Conflict,
        ),
        (
            StoreError::Stale {
                reference: "note:1".into(),
                current: vec!["note:2".into()],
            },
            polylogue_model::Code::Stale,
        ),
        (
            StoreError::NotFound("session".into()),
            polylogue_model::Code::NotFound,
        ),
        (StoreError::ReadCancelled, polylogue_model::Code::Cancelled),
    ];
    for (error, code) in cases {
        assert_eq!(error.code(), code, "{error}");
        assert_eq!(SelectionError::from(error).code(), code);
    }
}

/// `cwd:` finds sessions through a key range over the directories sessions
/// recorded, never by reading every session row, at a deep directory and at
/// `/`; and it matches whole path components.
#[test]
fn cwd_searches_recorded_directories_and_never_scans_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (id, cwd) in [
        ("test:here", "/work/morrow"),
        ("test:below", "/work/morrow/sub"),
        ("test:sibling", "/work/morrow-old"),
        ("test:elsewhere", "/elsewhere"),
    ] {
        f.session(SessionFacts {
            id,
            cwd: Some(cwd),
            last: Some(1),
            ..Default::default()
        })
        .unwrap();
    }
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    for text in [
        "sessions where cwd:/work/morrow",
        "sessions where cwd:/",
        "sessions where cwd:/work/morrow OR cwd:/elsewhere",
    ] {
        let program = crate::query::compile(text, &options()).unwrap();
        let plan: Vec<String> = reader
            .query(
                &format!(
                    "EXPLAIN QUERY PLAN {}",
                    program.statements.page(
                        &program.order,
                        program.params.len(),
                        None,
                        false,
                        crate::query::PageColumns::All,
                    )
                ),
                &[],
            )
            .unwrap()
            .into_iter()
            .map(|row| row["detail"].as_str().unwrap().to_owned())
            .collect();
        assert!(
            plan.iter()
                .any(|step| step.starts_with("SEARCH session_cwd_member")),
            "{text}: {plan:#?}"
        );
        assert!(
            !plan
                .iter()
                .any(|step| step == "SCAN s" || step.starts_with("SCAN s ")),
            "{text}: {plan:#?}"
        );
    }
    let mut options = options();
    options.page_size = None;
    options.fields = vec!["id".into()];
    let ids = |text: &str| {
        let selected = reader.prepare_selection(text, &options).unwrap();
        let mut ids = Vec::new();
        reader
            .visit_selection::<SelectionError>(
                &selected,
                crate::read::VisitFrom::First,
                None,
                |row| {
                    ids.push(row.values["id"].as_str().unwrap().to_owned());
                    Ok(ControlFlow::Continue(()))
                },
            )
            .unwrap();
        ids.sort();
        ids
    };
    assert_eq!(
        ids("sessions where cwd:/work/morrow"),
        ["test:below", "test:here"]
    );
    assert_eq!(ids("sessions where cwd:/").len(), 4);
}

#[test]
fn unseeded_random_windows_keep_rows_and_their_order() {
    let (_directory, fixture) = fixture();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.random_seed = None;
    for count in 1..=3 {
        let selection = reader
            .prepare_query_selection(
                &format!("sessions | sort by random() | limit {count}"),
                &options,
            )
            .unwrap();
        let mut rows = Vec::new();
        reader
            .visit_selection::<SelectionError>(
                &selection,
                crate::read::VisitFrom::First,
                None,
                |row| {
                    rows.push(row);
                    Ok(ControlFlow::Continue(()))
                },
            )
            .unwrap();
        assert_eq!(rows.len(), count);
        let targets = rows
            .iter()
            .map(|row| row.target.as_ref().unwrap().to_string())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(targets.len(), count);
        assert!(rows.windows(2).all(|pair| {
            pair[0].position.values[1].as_i64().unwrap()
                <= pair[1].position.values[1].as_i64().unwrap()
        }));
    }
}

#[test]
fn matches_evaluate_selected_parents_once_and_keep_their_order() {
    let (_directory, fixture) = fixture();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.random_seed = None;
    for order in ["id desc", "random()"] {
        for _ in 0..8 {
            let selection = reader.prepare_query_selection(
                &format!("sessions where exists messages(role:assistant) | sort by {order} | limit 2 | matches"),
                &options
            ).unwrap();
            let mut rows = Vec::new();
            reader
                .visit_selection::<SelectionError>(
                    &selection,
                    crate::read::VisitFrom::First,
                    None,
                    |row| {
                        rows.push(row);
                        Ok(ControlFlow::Continue(()))
                    },
                )
                .unwrap();
            assert_eq!(rows.len(), 2, "{order}");
            let parents = rows
                .iter()
                .map(|row| row.values["parent"].as_str().unwrap())
                .collect::<Vec<_>>();
            assert_ne!(parents[0], parents[1]);
            if order == "id desc" {
                assert_eq!(parents, ["session:test:two", "session:test:three"]);
            } else {
                assert!(
                    rows[0].position.values[1].as_i64().unwrap()
                        <= rows[1].position.values[1].as_i64().unwrap()
                );
            }
            for row in &rows {
                assert_eq!(
                    row.values["parent"],
                    format!("session:{}", row.values["parent.id"].as_str().unwrap())
                );
            }
        }
    }
}

/// Two sessions, each with two tool calls.
fn sessions_with_calls() -> (tempfile::TempDir, Fixture) {
    use crate::fixture::CallFacts;
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for id in ["test:one", "test:two"] {
        let s = f
            .session(SessionFacts {
                id,
                last: Some(1),
                ..Default::default()
            })
            .unwrap();
        let m = f
            .message(
                s,
                MessageFacts {
                    id: &format!("{id}-m"),
                    role: "assistant",
                    written_by: "model",
                    ..Default::default()
                },
            )
            .unwrap();
        for _ in 0..2 {
            f.tool_call(
                s,
                m,
                CallFacts {
                    tool: "Bash",
                    action: "shell",
                    outcome: "ok",
                    ..Default::default()
                },
            )
            .unwrap();
        }
    }
    (dir, f)
}

#[test]
fn session_field_uses_session_index() {
    let (_dir, f) = sessions_with_calls();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let program = crate::query::compile("tool_calls where session:test:one", &options()).unwrap();
    let plan: Vec<String> = reader
        .query(
            &format!(
                "EXPLAIN QUERY PLAN {}",
                program.statements.page(
                    &program.order,
                    program.params.len(),
                    None,
                    false,
                    crate::query::PageColumns::All,
                )
            ),
            &values(&program.params),
        )
        .unwrap()
        .into_iter()
        .map(|row| row["detail"].as_str().unwrap().to_owned())
        .collect();
    assert!(
        plan.iter()
            .any(|step| step.contains("USING INDEX tool_call_session")),
        "{plan:#?}"
    );
    assert!(
        !plan
            .iter()
            .any(|step| step == "SCAN t" || step.starts_with("SCAN t ")),
        "{plan:#?}"
    );
}

#[test]
fn session_field_and_reference_return_same_rows() {
    let (_dir, f) = sessions_with_calls();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.page_size = None;
    let rows = |text: &str| {
        let program = crate::query::compile(text, &options).unwrap();
        let mut keys: Vec<i64> = reader
            .query_rows(&program)
            .unwrap()
            .into_iter()
            .map(|row| row["_pk"].as_i64().unwrap())
            .collect();
        keys.sort_unstable();
        keys
    };
    let by_field = rows("tool_calls where session:test:one");
    assert_eq!(by_field.len(), 2);
    assert_eq!(by_field, rows("test:one | tool_calls"));
}

#[test]
fn fixed_populations_keep_zero_one_many_grains_and_cursor_order() {
    use crate::read::VisitFrom;
    use polylogue_model::query_objects::{ExecutionPresentation, QueryExecutionOptions};
    let (_dir, fixture) = fixture();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    for references in [
        vec![],
        vec!["session:test:two"],
        vec![
            "message:test:three:m",
            "session:test:one",
            "message:test:two:m",
        ],
    ] {
        let references: Vec<Reference> = references
            .into_iter()
            .map(|reference| Reference::parse(reference).unwrap())
            .collect();
        let selected = reader
            .prepare_fixed_selection("sessions", &references, &options())
            .unwrap();
        let mut rows = Vec::new();
        reader
            .visit_selection::<SelectionError>(&selected, VisitFrom::First, None, |row| {
                rows.push(row);
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| row.target.as_ref().unwrap().reference())
                .collect::<Vec<_>>(),
            references
        );
        if let Some(first) = rows.first() {
            let mut later = Vec::new();
            reader
                .visit_selection::<SelectionError>(
                    &selected,
                    VisitFrom::After(&first.position),
                    None,
                    |row| {
                        later.push(row.target.unwrap().reference());
                        Ok(ControlFlow::Continue(()))
                    },
                )
                .unwrap();
            assert_eq!(later, references[1..]);
        }
        if let Some(last) = rows.last() {
            let mut earlier = Vec::new();
            reader
                .visit_selection::<SelectionError>(
                    &selected,
                    VisitFrom::Before(&last.position),
                    None,
                    |row| {
                        earlier.push(row.target.unwrap().reference());
                        Ok(ControlFlow::Continue(()))
                    },
                )
                .unwrap();
            assert_eq!(
                earlier,
                references[..references.len() - 1]
                    .iter()
                    .rev()
                    .cloned()
                    .collect::<Vec<_>>()
            );
        }
        let binding = selected
            .execution_binding(
                QueryExecutionOptions {
                    references: None,
                    fields: vec![],
                    self_session: None,
                    here: None,
                    seen_marks_of: None,
                    embeddings: false,
                    why: false,
                    ranking_identity: None,
                },
                ExecutionPresentation::Read {
                    options: Default::default(),
                    delivery: Default::default(),
                    format: polylogue_model::read::Format::Json,
                    page_budget: Some(Default::default()),
                    direct_source: None,
                },
                None,
            )
            .unwrap();
        assert_eq!(binding.query_options.references, Some(references));
    }
}
