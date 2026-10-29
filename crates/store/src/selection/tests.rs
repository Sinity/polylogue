use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
fn options() -> Options {
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
        command_words: vec![],
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
            refs.push(row.reference.unwrap().to_string());
            after = Some(row.position);
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
    assert_eq!(refs, ["session:test:three"]);
    reader
        .visit_selection::<SelectionError>(&prepared, after.as_ref(), None, |row| {
            refs.push(row.reference.unwrap().to_string());
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
            refs.push(row.reference.unwrap().to_string());
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
        .prepare_selection("sessions | limit 1 | offset 1", &options)
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
        rows[0].reference.as_ref().unwrap().to_string(),
        "session:test:two"
    );
    assert!(
        reader
            .selection_contains(
                &prepared,
                &Reference::Message(MessageId::parse("test:two:m").unwrap())
            )
            .unwrap()
    );
    assert!(
        !reader
            .selection_contains(
                &prepared,
                &Reference::Message(MessageId::parse("test:three:m").unwrap())
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
    assert!(rows[0].reference.is_none());
    assert!(rows[0].first_seen.is_none());
}
#[test]
fn pure_bindings_preserve_closed_group_values_without_a_clock() {
    let bindings = polylogue_query::Bindings {
        saved: BTreeMap::from([(
            "by-outcome".into(),
            vec![polylogue_query::SavedQuery {
                text: "tool_calls | group by outcome".into(),
                revision: "rev".into(),
                first_note: "root".into(),
            }],
        )]),
        ..Default::default()
    };
    let schema = bindings.output_schema("query:by-outcome").unwrap();
    let outcome = schema
        .fields
        .iter()
        .find(|field| field.name == "outcome")
        .unwrap();
    assert_eq!(outcome.kind, query_language::catalog::FieldKind::Match);
    assert!(outcome.values.iter().any(|value| value == "error"));
    let kept = polylogue_query::compile::columns_schema(
        &bindings.catalog(),
        "tool_calls",
        &[polylogue_query::Column {
            name: "outcome".into(),
            kind: polylogue_query::ColKind::Text,
            reference: None,
        }],
    );
    assert_eq!(kept.fields[0].values, outcome.values);
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
    assert!(rows[0].first_seen.is_none());
}
#[test]
fn saved_aliases_bind_to_definition_hashes_and_keep_revision_metadata() {
    let (_dir, mut f) = fixture();
    let first = f
        .note(
            "saved_query",
            Some("alpha"),
            Some("sessions where origin:test"),
            "query:alpha",
        )
        .unwrap();
    let second = f
        .note(
            "saved_query",
            Some("beta"),
            Some("sessions where origin:test"),
            "query:beta",
        )
        .unwrap();
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
