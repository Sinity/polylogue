use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

mod lineage;
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
        saved: Default::default(),
        fields: vec!["id".into(), "at".into()],
        kept: Default::default(),
        chat_pages: vec![],
    }
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
            _ => "1970-01-01T00:00:03Z",
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
    assert_eq!(error.code(), "bad_request");
}

#[test]
fn native_run_namespace_never_falls_back_to_a_provider_origin() {
    let mut options = options();
    options.origins.push("run".into());
    assert_eq!(
        crate::query::compile("run:not-a-run", &options)
            .unwrap_err()
            .kind(),
        "invalid_query"
    );
    assert_eq!(
        crate::query::compile("run:019a2f20-1234-7000-8000-0123456789ab", &options)
            .unwrap_err()
            .kind(),
        "not_built"
    );
}

#[test]
fn native_time_predicates_accept_a_declared_fixture_event_instant() {
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

#[test]
fn native_time_missing_related_metadata_keeps_unknown_truth() {
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
        ("links where child.started >= 2026-10-01", 0),
        ("links where NOT child.started >= 2026-10-01", 0),
        ("links where child.started:unknown", 1),
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
                "assertion:subject@revision".into(),
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
            .user(crate::fixture::authored::operator_request(
                id.into(),
                UserWrite::Deliver(Delivery {
                    recipient_kind: "session".into(),
                    recipient_id: Some("codex:not-yet-captured".into()),
                    sender: "hook".into(),
                    event: "advice".into(),
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
        (r#"deliveries where refs:"assertion:subject@revision""#, 1),
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
        if expected == 1 {
            assert_eq!(page.rows[0]["id"], "advice-with-refs", "{query}");
        }
    }
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
        _ => panic!("native session selection"),
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
    use crate::read::limit::{ReadCancellation, ReadLimit};
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
        _ => panic!("native session selection"),
    }
    let pool = fixture.reads().unwrap();
    let limit = ReadLimit::new(
        std::time::Duration::from_secs(1),
        ReadCancellation::default(),
    );
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
    use crate::read::limit::{ReadCancellation, ReadLimit};
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    fixture
        .session(SessionFacts {
            id: "test:only",
            ..Default::default()
        })
        .unwrap();
    let pool = fixture.reads().unwrap();
    let cancellation = ReadCancellation::default();
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
fn cached_counts_follow_the_actual_vector_snapshot() {
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
                text: Some("synthetic retained text"),
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
        )
        .unwrap();
    let mut options = options();
    options.fields = vec!["id".into()];
    let mut program = crate::query::compile("messages", &options).unwrap();
    program.near = Some(crate::query::compile::Near {
        text: "synthetic seed".into(),
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
        _ => panic!("native message selection"),
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
