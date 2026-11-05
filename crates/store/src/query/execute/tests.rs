use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

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
        assert_eq!(page.count, 3);
        assert_eq!(page.without, 0);
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
    assert_eq!(page.count, 1);
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
        .link(
            "test:parent",
            "test:missing-child",
            "spawn",
            "native-marker",
        )
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
            reader.query_page(&program, None).unwrap().count,
            count,
            "{text}"
        );
    }
}
