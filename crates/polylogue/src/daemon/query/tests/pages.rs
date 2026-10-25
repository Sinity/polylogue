use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};

fn page(home: &ClaudeHome, cursor: Option<&str>) -> Result<Value, Failure> {
    let pool = home.reads();
    let reader = pool.reader().unwrap();
    let mut request = ask("sessions | sort by id");
    request.page_size = Some(2);
    request.cursor = cursor;
    run(&reader, request)
}

#[test]
fn writes_between_pages_neither_refuse_nor_repeat_rows() {
    let home = ClaudeHome::new();
    for session in ["a", "b", "c"] {
        home.append(
            &format!("{session}.jsonl"),
            &claude_user(&format!("{session}1"), session, "synthetic prompt"),
        );
    }
    let (mut engine, _lock) = home.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    let first = page(&home, None).unwrap();
    assert_eq!(first["count"], 3);
    assert_eq!(first["rows"][0]["id"], "claude-code:a");
    assert_eq!(first["rows"][1]["id"], "claude-code:b");
    let cursor = first["next"].as_str().unwrap();
    // One session sorts before the cursor, one after it.
    home.append("aa.jsonl", &claude_user("aa1", "aa", "new earlier prompt"));
    home.append("d.jsonl", &claude_user("d1", "d", "new later prompt"));
    assert!(engine.converge().unwrap().errors.is_empty());
    let second = page(&home, Some(cursor)).unwrap();
    assert_eq!(second["count"], 5);
    let ids: Vec<&str> = second["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["claude-code:c", "claude-code:d"]);
    assert!(second["next"].is_null());
}

#[test]
fn pages_cover_every_shape_and_allow_new_page_sizes_and_the_original_clock() {
    let w = morrow();
    for query in [
        "kind:* | sort by id",
        "kind:*",
        "messages | sort by model desc",
        "messages | distinct",
        "tool_calls | group by action | count",
        "sessions where exists tool_call(action:shell) | matches",
        "sessions where exists tool_call(action:shell) | matches | sort by command desc",
        "sessions where seq(action:shell -> action:shell) | matches",
    ] {
        let mut request = ask(query);
        request.page_size = Some(2);
        let first = run_q(&w, request).unwrap();
        let count = first["count"].as_u64().unwrap();
        let mut rows = first["rows"].as_array().unwrap().clone();
        let mut cursor = first["next"].as_str().map(str::to_owned);
        let mut size = 1;
        while let Some(next) = cursor {
            let mut request = ask(query);
            request.cursor = Some(&next);
            request.page_size = Some(size);
            size += 1;
            request.now += DAY;
            let page = run_q(&w, request).unwrap();
            assert_eq!(page["count"], count, "{query}");
            assert_eq!(page["resolved"], first["resolved"], "{query}");
            rows.extend(page["rows"].as_array().unwrap().iter().cloned());
            cursor = page["next"].as_str().map(str::to_owned);
        }
        assert_eq!(rows.len() as u64, count, "{query}");
        assert_eq!(
            rows,
            q(&w, query)["rows"].as_array().unwrap().clone(),
            "{query}"
        );
    }
}

#[test]
fn continuation_binds_fields_caller_zone_day_and_semantic_options() {
    let w = morrow();
    let mut first = ask("kind:*");
    first.page_size = Some(1);
    let result = run_q(&w, first).unwrap();
    let cursor = result["next"].as_str().unwrap();
    let fields = vec!["id".to_owned()];
    for changed in [
        "fields",
        "caller",
        "zone",
        "day-start",
        "embeddings",
        "why",
        "selection",
    ] {
        let mut request = ask("kind:*");
        request.cursor = Some(cursor);
        match changed {
            "fields" => request.fields = &fields,
            "caller" => request.session = Some("claude-code:different-caller".into()),
            "zone" => request.day = LogicalDay::named("Europe/Warsaw", 0).unwrap(),
            "day-start" => request.day.start_minutes = 240,
            "embeddings" => request.embeddings = true,
            "why" => request.why = true,
            "selection" => request.text = "kind:* | limit 4",
            _ => unreachable!(),
        }
        assert_eq!(
            run_q(&w, request).unwrap_err().kind,
            "bad_request",
            "{changed}"
        );
    }
}

#[test]
fn edits_and_rebuilds_between_pages_continue_after_the_last_row() {
    for rebuild in [false, true] {
        let home = ClaudeHome::new();
        for session in ["a", "b", "c"] {
            home.append(
                &format!("{session}.jsonl"),
                &claude_user(&format!("{session}1"), session, "before"),
            );
        }
        let (mut engine, lock) = home.engine(64 << 20);
        assert!(engine.converge().unwrap().errors.is_empty());
        let first = page(&home, None).unwrap();
        let cursor = first["next"].as_str().unwrap();
        if rebuild {
            drop(engine);
            drop(lock);
            polylogue_testkit::delete_index(&home.root);
            let (mut reopened, _lock) = home.engine(64 << 20);
            assert!(reopened.converge().unwrap().errors.is_empty());
            assert_eq!(page(&home, Some(cursor)).unwrap()["rows"][0]["id"], "claude-code:c");
        } else {
            std::fs::write(
                home.file("b.jsonl"),
                claude_user("b1", "b", "after exact rewrite"),
            )
            .unwrap();
            assert!(engine.converge().unwrap().errors.is_empty());
            assert_eq!(page(&home, Some(cursor)).unwrap()["rows"][0]["id"], "claude-code:c");
        }
    }
}

fn fixture_query(
    fixture: &Fixture,
    text: &str,
    cursor: Option<&str>,
    size: u64,
) -> Result<Value, Failure> {
    let pool = fixture.reads().unwrap();
    let mut request = ask(text);
    request.cursor = cursor;
    request.page_size = Some(size);
    run(&pool.reader().unwrap(), request)
}

fn derived_fixture() -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    for name in ["a", "b", "c"] {
        let session = fixture
            .session(SessionFacts {
                id: &format!("claude-code:{name}"),
                ..Default::default()
            })
            .unwrap();
        fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("claude-code:{name}1"),
                    role: "user",
                    written_by: "human",
                    text: Some(&format!("derived {name}")),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    (dir, fixture)
}

#[test]
fn derived_rows_between_pages_continue_the_pages_and_report_each_session_once() {
    let (_dir, mut fixture) = derived_fixture();
    let first = fixture_query(&fixture, "sessions | sort by id", None, 1).unwrap();
    let page_cursor = first["next"].as_str().unwrap();
    let change_cursor = first["view"]["cursor"].as_str().unwrap();
    assert!(fixture.derive().unwrap() > 0);
    assert_eq!(
        fixture_query(&fixture, "sessions | sort by id", Some(page_cursor), 1).unwrap()["rows"]
            [0]["id"],
        "claude-code:b"
    );
    let changes = fixture_query(
        &fixture,
        &format!("changes where after:{change_cursor}"),
        None,
        100,
    )
    .unwrap();
    assert_eq!(changes["count"], 3);
    assert!(
        changes["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["kind"] == "session")
    );
    let current = fixture_query(&fixture, "sessions | sort by id", None, 100).unwrap();
    assert_eq!(current["rows"][0]["title"], "derived a");
    let settled = changes["view"].clone();
    assert_eq!(fixture.derive().unwrap(), 0);
    let after = fixture_query(
        &fixture,
        &format!(
            "changes where after:{}",
            settled["cursor"].as_str().unwrap()
        ),
        None,
        100,
    )
    .unwrap();
    assert_eq!(after["count"], 0);
    assert_eq!(after["view"], settled);
}

#[test]
fn search_text_committed_between_pages_continues_without_reporting_a_session_edit() {
    let (_dir, fixture) = derived_fixture();
    fixture
        .queue_search_text("queued independent search text")
        .unwrap();
    let first = fixture_query(&fixture, "sessions | sort by id", None, 1).unwrap();
    assert_eq!(fixture.publish_search().unwrap(), 1);
    assert_eq!(
        fixture_query(&fixture, "sessions | sort by id", first["next"].as_str(), 1).unwrap()
            ["rows"][0]["id"],
        "claude-code:b"
    );
    let changes = fixture_query(
        &fixture,
        &format!(
            "changes where after:{}",
            first["view"]["cursor"].as_str().unwrap()
        ),
        None,
        100,
    )
    .unwrap();
    assert_eq!(
        changes["count"], 0,
        "index availability did not edit session records"
    );
    assert!(changes["view"]["index"].as_i64().unwrap() > first["view"]["index"].as_i64().unwrap());
    assert_eq!(fixture.publish_search().unwrap(), 0);
    assert_eq!(
        fixture_query(&fixture, "sessions", None, 100).unwrap()["view"],
        changes["view"]
    );
}

#[test]
fn random_selection_count_rows_and_pages_use_one_resolved_seed() {
    for query in [
        "kind:* | sort by random | limit 6 | where origin:claude-code",
        "kind:* | group by origin | count | sort by random",
        "changes | sort by random",
    ] {
        let w = morrow();
        let mut request = ask(query);
        request.page_size = Some(1);
        let first = run_q(&w, request).unwrap();
        let seed = Cursor::decode(first["next"].as_str().unwrap())
            .unwrap()
            .seed;
        let reference = query.replace("random", &format!("random({seed})"));
        let expected = q(&w, &reference);
        assert_eq!(first["count"], expected["count"], "{query}");
        let mut rows = first["rows"].as_array().unwrap().clone();
        let mut cursor = first["next"].as_str().map(str::to_owned);
        while let Some(next) = cursor {
            let mut request = ask(query);
            request.cursor = Some(&next);
            request.page_size = Some(2);
            let page = run_q(&w, request).unwrap();
            assert_eq!(page["count"], expected["count"], "{query}");
            rows.extend(page["rows"].as_array().unwrap().iter().cloned());
            cursor = page["next"].as_str().map(str::to_owned);
        }
        assert_eq!(
            rows,
            expected["rows"].as_array().unwrap().clone(),
            "{query}"
        );
    }
}

#[test]
fn near_single_page_succeeds_but_semantic_continuation_never_calls_the_ranker() {
    use std::cell::Cell;
    let w = morrow();
    let calls = Cell::new(0);
    let rank = |_: &str| {
        calls.set(calls.get() + 1);
        Ok("[]".to_owned())
    };
    let text = "kind:* near:\"synthetic ranking\"";
    let mut single = ask(text);
    single.page_size = Some(100);
    single.embeddings = true;
    single.near = Some(&rank);
    assert_eq!(run_q(&w, single).unwrap()["count"], 8);
    assert_eq!(calls.get(), 1);
    let pool = w.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut request = ask(text);
    request.page_size = Some(1);
    request.embeddings = true;
    request.near = Some(&rank);
    let opts = options(&reader, &request, request.now, 7).unwrap();
    let program = polylogue_query::compile(text, &opts).unwrap();
    let cursor = Cursor::new(
        vec![Value::Null; program.order.len()],
        program.now,
        7,
        page::question(&program, &opts, false).unwrap(),
    )
    .encode()
    .unwrap();
    drop(reader);
    drop(pool);
    calls.set(0);
    request.cursor = Some(&cursor);
    assert_eq!(run_q(&w, request).unwrap_err().kind, "not_built");
    assert_eq!(calls.get(), 0);
}

#[test]
fn page_cursor_carries_no_query_or_caller_text_and_rejects_other_shapes() {
    use base64::Engine as _;
    let w = morrow();
    let text = "sessions where title:\"opaque private marker\" OR kind:*";
    let mut request = ask(text);
    request.page_size = Some(1);
    request.session = Some("claude-code:private-caller-marker".into());
    let first = run_q(&w, request).unwrap();
    let cursor = first["next"].as_str().unwrap();
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(cursor)
        .unwrap();
    let encoded = String::from_utf8(bytes).unwrap();
    assert!(!encoded.contains(text));
    assert!(!encoded.contains("private marker"));
    assert!(!encoded.contains("private-caller-marker"));
    let old = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(br#"{"offset":1,"now":0,"query":"old"}"#);
    let mut request = ask("sessions");
    request.cursor = Some(&old);
    assert_eq!(run_q(&w, request).unwrap_err().kind, "bad_request");
}
