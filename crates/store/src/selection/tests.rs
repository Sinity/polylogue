mod kept;

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
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        chat_pages: vec![],
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
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
            refs.push(row.target.unwrap().to_string());
            after = Some(row.position);
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
    assert_eq!(refs, ["session:test:three"]);
    reader
        .visit_selection::<SelectionError>(&prepared, after.as_ref(), None, |row| {
            refs.push(row.target.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(
        refs,
        ["session:test:three", "session:test:two", "session:test:one"]
    );
}
#[test]
fn prepared_selection_identity_resolves_latest_without_hashing_execution() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let latest = reader.prepare_selection("latest", &options()).unwrap();
    let explicit = reader
        .prepare_selection("session:test:three", &options())
        .unwrap();
    assert_eq!(latest.question_id(), explicit.question_id());
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
fn prepared_selection_retains_exact_block_reference() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let target = "block:test:one:m#0";
    let prepared = reader.prepare_selection(target, &options()).unwrap();
    let mut refs = vec![];
    reader
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
            refs.push(row.target.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(refs, [target]);
}

#[test]
fn selected_reference_survives_projection_and_membership_keeps_offset() {
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
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
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
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
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
fn native_facets_share_field_values_and_stream_cwd_ancestors() {
    use crate::fixture::CallFacts;
    use crate::query::facets::{FacetError, FacetKey};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "test:facet",
            cwd: Some("/work/morrow/sub"),
            repo: Some("morrow"),
            ..Default::default()
        })
        .unwrap();
    for id in ["test:one", "test:two"] {
        f.message(
            s,
            MessageFacts {
                id,
                role: "assistant",
                written_by: "model",
                model: Some("model-a"),
                ..Default::default()
            },
        )
        .unwrap();
    }
    let m = f
        .message(
            s,
            MessageFacts {
                id: "test:tool",
                role: "assistant",
                written_by: "model",
                model: Some("model-b"),
                ..Default::default()
            },
        )
        .unwrap();
    f.tool_call(
        s,
        m,
        CallFacts {
            tool: "mcp__notes__find",
            action: "mcp",
            outcome: "ok",
            ..Default::default()
        },
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let keys = [
        FacetKey::new(Unit::Sessions, "repo"),
        FacetKey::new(Unit::Sessions, "cwd"),
        FacetKey::new(Unit::Sessions, "model"),
        FacetKey::new(Unit::ToolCalls, "mcp_server"),
    ];
    let mut values = BTreeMap::<_, BTreeMap<_, _>>::new();
    reader
        .visit_session_facets::<FacetError>(
            &SessionId::parse("test:facet").unwrap(),
            &keys,
            options().clock.now,
            |row| {
                values
                    .entry(row.key)
                    .or_default()
                    .insert(row.value, row.count);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(values[&keys[0]], BTreeMap::from([("morrow".into(), 1)]));
    assert_eq!(
        values[&keys[1]],
        BTreeMap::from([
            ("/".into(), 1),
            ("/work".into(), 1),
            ("/work/morrow".into(), 1),
            ("/work/morrow/sub".into(), 1)
        ])
    );
    assert_eq!(
        values[&keys[2]],
        BTreeMap::from([("model-a".into(), 2), ("model-b".into(), 1)])
    );
    assert_eq!(values[&keys[3]], BTreeMap::from([("notes".into(), 1)]));
    let selected = reader
        .prepare_selection("sessions where cwd:/work/morrow", &options())
        .unwrap();
    let mut rows = 0;
    reader
        .visit_selection::<SelectionError>(&selected, None, None, |_| {
            rows += 1;
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(rows, 1);
}
#[test]
fn absence_of_hook_and_activity_evidence_is_unknown_live_state() {
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
        .visit_selection::<SelectionError>(&selected, None, None, |row| {
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
            .visit_selection::<SelectionError>(&selection, None, None, |row| {
                rows.push(row.values);
                Ok(ControlFlow::Continue(()))
            })
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
fn canonical_object_selection_replays_frozen_definitions_in_the_same_reader() {
    let (_dir, mut f) = fixture();
    save_definition(&mut f, "picked", "sessions where session:test:one");
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let prepared = reader
        .prepare_selection("query:picked", &options())
        .unwrap();
    let object = prepared.query_object().unwrap();
    let mut unrelated = options();
    unrelated.saved.clear();
    let replay = reader
        .prepare_query_object_selection(
            &object,
            &unrelated,
            polylogue_query::plan::catalog_fingerprint(),
        )
        .unwrap();
    let mut references = vec![];
    reader
        .visit_selection::<SelectionError>(&replay, None, None, |row| {
            references.push(row.target.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(references, ["session:test:one"]);
    assert_eq!(
        prepared.cursor_context().versions,
        replay.cursor_context().versions
    );
}

#[test]
fn selected_assertion_reference_retains_its_actual_family_and_revision() {
    let (_dir, mut fixture) = fixture();
    let first = fixture
        .note("note", None, Some("first"), "session:test:one")
        .unwrap();
    let second = fixture
        .note("note", None, Some("second"), "session:test:one")
        .unwrap();
    // A fork with no revises link: both revisions of `first`'s family stay
    // current. The stored currency index is recomputed as a writer would.
    let conn = crate::archive::open_write(&fixture.archive.archive_db(), 1).unwrap();
    conn.execute(
        "UPDATE assertion SET root=?1 WHERE id=?2",
        rusqlite::params![first, second],
    )
    .unwrap();
    crate::assertions::currency::recompute(&conn, [first.clone(), second.clone()].into()).unwrap();
    drop(conn);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.fields = vec!["value".into()];
    let selected = reader
        .prepare_selection("assertions where current:* | sort by id", &options)
        .unwrap();
    let mut rows = vec![];
    reader
        .visit_selection::<SelectionError>(&selected, None, None, |row| {
            rows.push((row.target.unwrap().to_string(), row.values));
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(
        rows,
        vec![
            (
                format!("assertion:{first}@{first}"),
                serde_json::json!({"value":"first"})
            ),
            (
                format!("assertion:{first}@{second}"),
                serde_json::json!({"value":"second"})
            )
        ]
    );
    let exact = reader
        .prepare_selection(&format!("assertion:{first}@{first}"), &options)
        .unwrap();
    let mut exact_values = vec![];
    reader
        .visit_selection::<SelectionError>(&exact, None, None, |row| {
            exact_values.push(row.values);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(exact_values, vec![serde_json::json!({"value":"first"})]);
    assert_eq!(
        reader
            .prepare_selection(&format!("assertion:{first}"), &options)
            .err()
            .unwrap()
            .code(),
        "ambiguous"
    );
    let family = reader
        .prepare_selection(
            &format!("assertions where assertion:{first} | sort by id"),
            &options,
        )
        .unwrap();
    assert!(family.direct_target().is_none());
    let mut count = 0;
    reader
        .visit_selection::<SelectionError>(&family, None, None, |_| {
            count += 1;
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn direct_native_reads_preserve_opaque_ids_and_ranges_without_truncating_queries() {
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
            .visit_selection::<SelectionError>(selection, None, None, |row| {
                assert_eq!(row.values["id"], opaque_id);
                found.push(row.target.unwrap().reference());
                Ok(ControlFlow::Continue(()))
            })
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
            "not_found"
        );
    }
    let saved = reader
        .prepare_selection("query:literal-target", &options)
        .unwrap();
    assert!(saved.direct_target().is_none());
    let mut saved_ids = Vec::new();
    reader
        .visit_selection::<SelectionError>(&saved, None, None, |row| {
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
        "not_found"
    );
    let ordinary = reader
        .prepare_selection(
            "messages where message:test:a..test:b | sort by id",
            &options,
        )
        .unwrap();
    assert!(ordinary.direct_target().is_none());
    let mut selected = vec![];
    reader
        .visit_selection::<SelectionError>(&ordinary, None, None, |row| {
            selected.push(row.target.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(selected, ["message:test:a", "message:test:b"]);
    let semantic = reader
        .prepare_query_selection("messages where message:test:a..test:b", &options)
        .unwrap();
    assert!(semantic.direct_target().is_none());
    let mut semantic_count = 0;
    reader
        .visit_selection::<SelectionError>(&semantic, None, None, |_| {
            semantic_count += 1;
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(semantic_count, 2);
    let range = reader
        .prepare_selection("message:test:a..test:b", &options)
        .unwrap();
    assert!(matches!(
        range.direct_target(),
        Some(ReadTarget::MessageRange { .. })
    ));
}

#[test]
fn direct_authored_reference_uses_exact_family_revision_membership() {
    let (_dir, mut fixture) = fixture();
    let first = fixture
        .note("lesson", None, Some("first"), "session:test:one")
        .unwrap();
    let second = fixture
        .note("lesson", None, Some("second"), "session:test:two")
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let reference = format!("assertion:{first}@{first}");
    let prepared = reader.prepare_selection(&reference, &options()).unwrap();
    let mut selected = Vec::new();
    reader
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
            selected.push(row.target.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(selected, [reference]);
    let missing = format!("assertion:{first}@{second}");
    assert!(
        matches!(reader.prepare_selection(&missing, &options()), Err(SelectionError::Invalid { kind, .. }) if kind == "not_found")
    );
    let wrong_family = reader
        .prepare_query_selection(&missing, &options())
        .unwrap();
    let mut count = 0;
    reader
        .visit_selection::<SelectionError>(&wrong_family, None, None, |_| {
            count += 1;
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn native_event_read_target_uses_event_grain_without_rebinding_ordinary_queries() {
    use crate::fixture::writer::WriterArchive;
    use polylogue_model::{Event, Output, TimeQuality};
    let mut archive = WriterArchive::new();
    let output = Output {
        events: vec![Event {
            session: None,
            kind: "native_event".into(),
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
    archive.write_output("native-event", "claude-code", "synthetic/event", output, 1);
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let semantic = reader
        .prepare_query_selection("events", &options())
        .unwrap();
    let mut references = vec![];
    reader
        .visit_selection::<SelectionError>(&semantic, None, None, |row| {
            references.push(row.target.unwrap());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(references.len(), 1);
    assert!(matches!(references[0], ReadTarget::Event(_)));
    let target = references[0].to_string();
    let direct = reader.prepare_selection(&target, &options()).unwrap();
    assert_eq!(direct.program().unit, Unit::Events);
    assert_eq!(direct.direct_target(), Some(&references[0]));
    let mut selected = vec![];
    reader
        .visit_selection::<SelectionError>(&direct, None, None, |row| {
            selected.push(row.target.unwrap());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(selected, references);
    let ordinary = reader.prepare_query_plan(&target, &options()).unwrap();
    assert!(matches!(&ordinary.bound_query().query().source,
        polylogue_query::ast::Source::Unit { unit, implicit: true, .. } if unit == "sessions"));
}

#[test]
fn assertion_values_are_decoded_once_and_predicates_match_recorded_text() {
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
        .prepare_query_selection("assertions | sort by id", &opts)
        .unwrap();
    let mut values = vec![];
    reader
        .visit_selection::<SelectionError>(&selected, None, None, |row| {
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
        .prepare_query_selection("assertions where value:quoted", &opts)
        .unwrap();
    let mut values = vec![];
    reader
        .visit_selection::<SelectionError>(&selected, None, None, |row| {
            values.push(row.values["value"].clone());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(values, vec![serde_json::json!("quoted \"body\"\nnext")]);
}

/// The prepared selection that assertion writes use and the listing the query
/// command shows must select the same rows for one question.
#[test]
fn prepared_selection_and_listing_select_the_same_rows_for_an_id_with_every_row_admitted() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.page_size = Some(10);
    let selected = |text: &str| {
        let prepared = reader.prepare_query_selection(text, &options).unwrap();
        let mut targets = vec![];
        reader
            .visit_selection::<SelectionError>(&prepared, None, None, |row| {
                targets.push(row.target.unwrap().to_string());
                Ok(ControlFlow::Continue(()))
            })
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
        (StoreError::Conflict("request-1".into()), "conflict"),
        (StoreError::ReadIndexReplaced, "conflict"),
        (
            StoreError::Stale {
                reference: "note:1".into(),
                current: vec!["note:2".into()],
            },
            "stale",
        ),
        (StoreError::NotFound("session".into()), "not_found"),
        (StoreError::ReadCancelled, "cancelled"),
    ];
    for (error, code) in cases {
        assert_eq!(error.code(), code, "{error}");
        assert_eq!(SelectionError::from(error).code(), code);
    }
}

/// law: cwd-reads-directory-key-range
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
            .query(&format!("EXPLAIN QUERY PLAN {}", program.sql), &[])
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
            .visit_selection::<SelectionError>(&selected, None, None, |row| {
                ids.push(row.values["id"].as_str().unwrap().to_owned());
                Ok(ControlFlow::Continue(()))
            })
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
fn unseeded_random_windows_keep_selected_rows_and_order_witnesses() {
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
            .visit_selection::<SelectionError>(&selection, None, None, |row| {
                rows.push(row);
                Ok(ControlFlow::Continue(()))
            })
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
                &format!("sessions where exists message(role:assistant) | sort by {order} | limit 2 | matches"),
                &options
            ).unwrap();
            let mut rows = Vec::new();
            reader
                .visit_selection::<SelectionError>(&selection, None, None, |row| {
                    rows.push(row);
                    Ok(ControlFlow::Continue(()))
                })
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
