use super::*;
use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};

mod lineage;
mod saved_fields;
mod search;

fn options() -> crate::query::Options {
    crate::query::Options {
        clock: query_language::time::Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("Etc/UTC").unwrap(),
                jiff::civil::time(0, 0, 0, 0),
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
        fields: vec!["id".into(), "at".into()],
        kept: Default::default(),
        provider_catalog: None,
    }
}

fn group_rows(reader: &crate::Reader<'_>, query: &str) -> Vec<Value> {
    let mut options = options();
    options.fields.clear();
    options.page_size = Some(20);
    options.origins.push("codex".into());
    let program = crate::query::compile(query, &options).unwrap();
    let mut page = reader.query_page(&program, None).unwrap();
    assert!(!page.more);
    let columns: std::collections::BTreeSet<_> = program
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect();
    for row in &mut page.rows {
        present(&program, row).unwrap();
        let keys: std::collections::BTreeSet<_> = row
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, columns, "{query}");
    }
    page.rows
}

#[test]
fn group_rows_carry_only_declared_columns() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    for id in ["test:first", "test:second", "codex:third"] {
        fixture
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
    }
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        group_rows(&reader, "sessions | group by origin"),
        [
            serde_json::json!({"origin": "test", "count": 2}),
            serde_json::json!({"origin": "codex", "count": 1}),
        ]
    );
}

#[test]
fn sort_by_count_after_group() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:paths",
            ..Default::default()
        })
        .unwrap();
    let message = fixture
        .message(
            session,
            MessageFacts {
                id: "test:reads",
                role: "assistant",
                written_by: "model",
                ..Default::default()
            },
        )
        .unwrap();
    for path in [
        "src/a.rs", "src/b.rs", "src/c.rs", "src/b.rs", "src/c.rs", "src/b.rs",
    ] {
        fixture
            .tool_call(
                session,
                message,
                CallFacts {
                    tool: "Read",
                    action: "read",
                    command: None,
                    path: Some(path),
                    outcome: "ok",
                    exit_code: None,
                    at: None,
                    output: Some("synthetic contents"),
                },
            )
            .unwrap();
    }
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        group_rows(&reader, "tool_calls | group by path | sort by count desc"),
        [
            serde_json::json!({"path": "src/b.rs", "count": 3}),
            serde_json::json!({"path": "src/c.rs", "count": 2}),
            serde_json::json!({"path": "src/a.rs", "count": 1}),
        ]
    );
}
#[test]
fn stored_pages_keep_total_count_and_format_only_declared_values() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:session",
            ..Default::default()
        })
        .unwrap();
    for (id, at) in [("test:c", 3000), ("test:a", 1000), ("test:b", 2000)] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "user",
                    written_by: "human",
                    prompt_kind: Some("prose"),
                    at: Some(at),
                    text: Some("synthetic"),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let program = crate::query::compile("messages | sort by id", &options()).unwrap();
    let mut after = None;
    let mut ids = vec![];
    loop {
        let mut page = reader.query_page(&program, after.as_deref()).unwrap();
        assert_eq!(reader.query_count(&program).unwrap(), Some((3, 0)));
        assert_eq!(page.rows.len(), 1);
        let raw = page.rows.last().unwrap();
        after = Some(
            program
                .order
                .iter()
                .map(|key| raw[&key.column].clone())
                .collect::<Vec<_>>(),
        );
        let row = &mut page.rows[0];
        present(&program, row).unwrap();
        assert_eq!(row.as_object().unwrap().len(), 2);
        ids.push(row["id"].as_str().unwrap().to_owned());
        let expected = match row["id"].as_str().unwrap() {
            "test:a" => "1970-01-01T00:00:01Z",
            "test:b" => "1970-01-01T00:00:02Z",
            "test:c" => "1970-01-01T00:00:03Z",
            other => panic!("unexpected id {other}"),
        };
        assert_eq!(row["at"], expected);
        if !page.more {
            break;
        }
    }
    assert_eq!(ids, ["test:a", "test:b", "test:c"]);
    assert_eq!(
        reader
            .query_keys(&program, QueryPopulation::Answer)
            .unwrap()
            .len(),
        3
    );
}
/// A stored instant the model cannot represent is a stored-data error, not an
/// unknown time.
#[test]
fn a_stored_instant_out_of_range_is_an_error() {
    let zone = jiff::tz::TimeZone::get("Etc/UTC").unwrap();
    let mut valid = serde_json::json!(1_000);
    shape_value(ColKind::Time, &zone, &mut valid).unwrap();
    assert_eq!(valid, "1970-01-01T00:00:01Z");
    let mut broken = serde_json::json!(i64::MAX);
    let error = shape_value(ColKind::Time, &zone, &mut broken).unwrap_err();
    assert_eq!(error.code(), polylogue_model::Code::Store);
}

#[test]
fn zero_transfer_size_is_a_typed_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.page_size = Some(0);
    let program = crate::query::compile("messages", &options).unwrap();
    let error = reader.query_page(&program, None).err().unwrap();
    assert_eq!(error.code(), polylogue_model::Code::BadRequest);
}

#[test]
fn run_namespace_never_falls_back_to_an_origin() {
    let mut options = options();
    options.origins.push("run".into());
    assert_eq!(
        crate::query::compile("run:not-a-run", &options)
            .unwrap_err()
            .kind(),
        polylogue_model::Code::InvalidQuery
    );
    assert_eq!(
        crate::query::compile("run:019a2f20-1234-7000-8000-0123456789ab", &options)
            .unwrap_err()
            .kind(),
        polylogue_model::Code::NotBuilt
    );
}

#[test]
fn source_time_tests_accept_a_fixture_event_instant() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:session",
            ..Default::default()
        })
        .unwrap();
    fixture.event(session, "synthetic", Some(1000)).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let program =
        crate::query::compile("events where at = 1970-01-01T00:00:01Z", &options()).unwrap();
    let mut page = reader.query_page(&program, None).unwrap();
    assert_eq!(reader.query_count(&program).unwrap(), Some((1, 0)));
    assert_eq!(page.rows.len(), 1);
    present(&program, &mut page.rows[0]).unwrap();
    assert_eq!(page.rows[0]["at"], "1970-01-01T00:00:01Z");
}

/// A time bound on a session or message keeps a period that reaches past
/// the bound though it starts before it, keeps an instant after it and
/// drops an instant before it, sub-millisecond digits included.
#[test]
fn a_time_bound_tests_periods_by_their_end_and_instants_by_their_time() {
    use polylogue_model::{Precision, TimeAccuracy, TimeQuality, Timestamp};
    let ms = |text: &str| Timestamp::parse_rfc3339(text).unwrap().unix_ms();
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let message = |fixture: &mut Fixture, name: &str, at: &str, until: Option<&str>| {
        let session = fixture
            .session(SessionFacts {
                id: &format!("test:{name}"),
                last: Some(ms(at)),
                ..Default::default()
            })
            .unwrap();
        let pk = fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("test:{name}-message"),
                    role: "assistant",
                    written_by: "model",
                    at: Some(ms(at)),
                    ..Default::default()
                },
            )
            .unwrap();
        fixture
            .message_time(
                pk,
                Some(Timestamp::parse_rfc3339(at).unwrap()),
                TimeAccuracy {
                    quality: TimeQuality::Intrinsic,
                    precision: Some(if until.is_some() {
                        Precision::Bounded
                    } else {
                        Precision::Instant
                    }),
                    until: until.map(|until| Timestamp::parse_rfc3339(until).unwrap()),
                },
            )
            .unwrap();
    };
    message(&mut fixture, "before", "2026-10-01T00:00:00Z", None);
    message(&mut fixture, "after", "2026-10-20T00:00:00Z", None);
    message(
        &mut fixture,
        "period",
        "2026-10-01T00:00:00Z",
        Some("2026-10-25T00:00:00Z"),
    );
    // Half a millisecond short of the bound, in the bound's own millisecond.
    message(
        &mut fixture,
        "just-before",
        "2026-10-09T23:59:59.9995Z",
        None,
    );
    message(
        &mut fixture,
        "just-after",
        "2026-10-10T00:00:00.0005Z",
        None,
    );
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.fields = vec!["id".into()];
    options.page_size = Some(20);
    let ids = |query: &str| {
        let program = crate::query::compile(query, &options).unwrap();
        let page = reader.query_page(&program, None).unwrap();
        let mut ids: Vec<String> = page
            .rows
            .iter()
            .map(|row| row["id"].as_str().unwrap().to_owned())
            .collect();
        ids.sort();
        assert_eq!(
            reader.query_count(&program).unwrap().map(|counts| counts.0),
            Some(ids.len() as u64),
            "{query}"
        );
        ids
    };
    assert_eq!(
        ids("sessions where since:2026-10-10"),
        ["test:after", "test:just-after", "test:period"]
    );
    assert_eq!(
        ids("messages where since:2026-10-10"),
        [
            "test:after-message",
            "test:just-after-message",
            "test:period-message"
        ]
    );
    assert_eq!(
        ids("sessions where last_active < 2026-10-10"),
        ["test:before", "test:just-before"]
    );
    // The bound names a whole second, which `just-after` falls inside.
    assert_eq!(
        ids("sessions where last_active > 2026-10-10T00:00:00Z"),
        ["test:after"]
    );
}

#[test]
fn source_time_without_metadata_stays_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    fixture
        .session(SessionFacts {
            id: "test:parent",
            ..Default::default()
        })
        .unwrap();
    fixture
        .link("test:parent", "test:missing-child", "spawn", None)
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.fields.clear();
    for (text, count) in [
        ("session_links where child.started >= 2026-10-01", 0),
        ("session_links where NOT child.started >= 2026-10-01", 0),
        ("session_links where child.started:unknown", 1),
    ] {
        let program = crate::query::compile(text, &options).unwrap();
        assert_eq!(
            reader.query_count(&program).unwrap().map(|counts| counts.0),
            Some(count),
            "{text}"
        );
    }
}

#[test]
fn delivery_filters_read_string_refs_and_structured_omissions_without_reparsing_strings() {
    use polylogue_model::user::{Delivery, LeftOut, UserWrite};
    let archive = crate::fixture::writer::WriterArchive::new();
    for (id, refs, left_out) in [
        (
            "advice-with-refs",
            vec![
                "annotation:subject@revision".into(),
                "file:src/parser.rs@commit".into(),
                r#"a literal \"quote\" and slash\"#.into(),
            ],
            vec![LeftOut {
                what: "session:codex:other".into(),
                reason: "budget".into(),
            }],
        ),
        ("advice-without-refs", vec![], vec![]),
    ] {
        archive
            .writer
            .archive()
            .user(crate::fixture::authored::operator_request(
                id.into(),
                UserWrite::Deliver(Delivery {
                    recipient_kind: "session".into(),
                    recipient_id: Some("codex:not-yet-captured".into()),
                    sender: polylogue_model::DeliverySender::Hook,
                    event: polylogue_model::DeliveryEvent::Advice,
                    cwd: None,
                    text: "synthetic pointer".into(),
                    refs,
                    left_out,
                }),
            ))
            .unwrap();
    }
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.fields = vec!["id".into()];
    for (query, expected) in [
        (r#"deliveries where refs:"annotation:subject@revision""#, 1),
        (r#"deliveries where refs:"src/parser.rs""#, 1),
        (r#"deliveries where refs:"quote""#, 1),
        (r#"deliveries where refs:"missing""#, 0),
        (r#"deliveries where left_out:"codex:other""#, 1),
        (r#"deliveries where left_out:"budget""#, 0),
        (r#"deliveries where refs:"codex:other""#, 0),
    ] {
        let program = crate::query::compile(query, &options).unwrap();
        let page = reader
            .query_page(&program, None)
            .unwrap_or_else(|error| panic!("{query}: {error:?}"));
        assert_eq!(
            reader.query_count(&program).unwrap().map(|counts| counts.0),
            Some(expected),
            "{query}"
        );
        assert_eq!(page.rows.len() as u64, expected, "{query}");
        assert!(!page.more, "{query}");
        if expected == 1 {
            assert_eq!(page.rows[0]["id"], "advice-with-refs", "{query}");
        }
    }
}

/// The count cache keeps its newest 64 answers at one read position and
/// every count still comes from its own question.
#[test]
fn the_count_cache_holds_at_most_64_answers() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    fixture
        .session(SessionFacts {
            id: "test:a",
            ..Default::default()
        })
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.fields = vec!["id".into()];
    for number in 0..70 {
        let program = crate::query::compile(
            &format!("sessions where id:\"test:none{number}\""),
            &options,
        )
        .unwrap();
        assert_eq!(
            reader.query_count(&program).unwrap().map(|counts| counts.0),
            Some(0),
            "{number}"
        );
    }
    assert_eq!(reader.count_cache().lock().unwrap().entries.len(), 64);
}

#[cfg(feature = "faults")]
#[test]
fn a_cached_count_is_reused_by_the_next_reader_and_page() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    for id in ["test:a", "test:b"] {
        fixture
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
    }
    let mut options = options();
    options.fields = vec!["id".into()];
    let program = crate::query::compile("sessions | sort by id", &options).unwrap();
    let pool = fixture.reads().unwrap();
    let first_reader = pool.reader().unwrap();
    let first = first_reader.query_page(&program, None).unwrap();
    assert_eq!(first_reader.query_count(&program).unwrap(), Some((2, 0)));
    let after = program
        .order
        .iter()
        .map(|key| first.rows[0][&key.column].clone())
        .collect::<Vec<_>>();
    drop(first_reader);
    let count_sql = program.statements.count(false);
    crate::fault::arm("read.query", &count_sql, crate::fault::Action::Panic, 0, 1);
    let next_reader = pool.reader().unwrap();
    let next = next_reader.query_page(&program, Some(&after)).unwrap();
    assert_eq!(next_reader.query_count(&program).unwrap(), Some((2, 0)));
    crate::fault::clear();
    assert_eq!(next.rows.len(), 1);
    assert_eq!(next.rows[0]["id"], "test:b");
    drop(next_reader);
    let new_pool = fixture.reads().unwrap();
    assert_eq!(
        new_pool.reader().unwrap().query_count(&program).unwrap(),
        Some((2, 0))
    );
}

#[test]
fn cached_counts_follow_the_resolved_calendar() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.fields = vec!["id".into()];
    let mut program = crate::query::compile("sessions", &options).unwrap();
    let before: jiff::Timestamp = "2026-01-01T23:30:00Z".parse().unwrap();
    let after: jiff::Timestamp = "2026-01-02T00:30:00Z".parse().unwrap();
    match &mut program.statements {
        crate::query::Statements::Keyed(rows) => {
            rows.count = format!(
                "SELECT count(DISTINCT pl_day(at)), 0 FROM (SELECT {} AS at UNION ALL SELECT {})",
                before.as_millisecond(),
                after.as_millisecond()
            );
        }
        _ => panic!("session selection"),
    }
    assert_eq!(reader.query_count(&program).unwrap(), Some((2, 0)));
    program.clock.day = query_language::time::DaySetting::new(
        jiff::tz::TimeZone::get("Europe/Warsaw").unwrap(),
        jiff::civil::time(0, 0, 0, 0),
    )
    .unwrap();
    assert_eq!(reader.query_count(&program).unwrap(), Some((1, 0)));
}

#[test]
fn slow_count_still_returns_rows() {
    use crate::{cancellation::Cancellation, read::limit::ReadLimit};
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    fixture
        .session(SessionFacts {
            id: "test:only",
            ..Default::default()
        })
        .unwrap();
    let mut options = options();
    options.fields = vec!["id".into()];
    let mut program = crate::query::compile("sessions", &options).unwrap();
    match &mut program.statements {
        crate::query::Statements::Keyed(rows) => rows.count = "WITH RECURSIVE slow(n) AS (SELECT 0 UNION ALL SELECT n+1 FROM slow WHERE n<1000000000) SELECT count(*), 0 FROM slow".into(),
        _ => panic!("session selection"),
    }
    let pool = fixture.reads().unwrap();
    let limit = ReadLimit::new(std::time::Duration::from_secs(1), Cancellation::default());
    limit.scope(|| {
        let reader = pool.reader().unwrap();
        let answer = reader.query_page(&program, None).unwrap();
        assert_eq!(reader.query_count(&program).unwrap(), None);
        assert_eq!(answer.rows.len(), 1);
        assert_eq!(answer.rows[0]["id"], "test:only");
        assert!(matches!(
            limit.check(),
            Err(crate::StoreError::ReadDeadlineExceeded)
        ));
    });
}

#[test]
fn cancellation_is_not_a_deferred_count() {
    use crate::{cancellation::Cancellation, read::limit::ReadLimit};
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    fixture
        .session(SessionFacts {
            id: "test:only",
            ..Default::default()
        })
        .unwrap();
    let pool = fixture.reads().unwrap();
    let cancellation = Cancellation::default();
    let limit = ReadLimit::without_deadline(cancellation.clone());
    limit.scope(|| {
        let reader = pool.reader().unwrap();
        let mut options = options();
        options.fields = vec!["id".into()];
        let program = crate::query::compile("sessions", &options).unwrap();
        assert_eq!(reader.query_page(&program, None).unwrap().rows.len(), 1);
        cancellation.cancel();
        assert!(matches!(
            reader.query_count(&program),
            Err(SelectionError::Store(crate::StoreError::ReadCancelled))
        ));
    });
}

#[test]
fn cached_counts_follow_the_vector_snapshot() {
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:only",
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
                text: Some("synthetic stored text"),
                ..Default::default()
            },
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    let old_reader = pool.reader().unwrap();
    let key = "local/count-test/2/document/unit";
    let old = Arc::new(old_reader.vectors().unwrap().load(key).unwrap());
    let old_similarity = Arc::new(old.similarity(&[1.0, 0.0]).unwrap());
    let hash: Vec<u8> = old_reader
        .conn()
        .unwrap()
        .query_row("SELECT hash FROM text", [], |row| row.get(0))
        .unwrap();
    old_reader
        .vectors_writer()
        .unwrap()
        .put(
            key,
            &[crate::vectors::Vector {
                text_hash: hash.clone(),
                chunk: 0,
                values: vec![1.0, 0.0],
            }],
            &polylogue_model::WallClock::fixed(jiff::Timestamp::UNIX_EPOCH),
        )
        .unwrap();
    let mut options = options();
    options.fields = vec!["id".into()];
    let mut program = crate::query::compile("messages", &options).unwrap();
    program.near = Some(crate::query::compile::Near {
        text: "synthetic seed".into(),
        text_ranked: false,
    });
    program.counts_near = true;
    let hex = hash
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    match &mut program.statements {
        crate::query::Statements::Keyed(rows) => {
            rows.count = format!(
                "SELECT count(*), count(*) FILTER (WHERE pl_similarity(hash) IS NULL) FROM text WHERE hash=x'{hex}'"
            )
        }
        _ => panic!("message selection"),
    }
    // Both counts run after the write notification, but their vectors differ.
    assert_eq!(
        old_reader
            .with_similarity(old_similarity, || old_reader.query_count(&program))
            .unwrap(),
        Some((1, 1))
    );
    let new_reader = pool.reader().unwrap();
    let loaded = Arc::new(new_reader.vectors().unwrap().load(key).unwrap());
    let similarity = Arc::new(loaded.similarity(&[1.0, 0.0]).unwrap());
    assert_eq!(
        new_reader
            .with_similarity(similarity, || new_reader.query_count(&program))
            .unwrap(),
        Some((1, 0))
    );
}
