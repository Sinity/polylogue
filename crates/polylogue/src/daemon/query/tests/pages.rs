use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};

#[test]
fn counting_ranked_rows_does_not_sort_or_project_the_answer() {
    let w = morrow();
    let pool = w.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    let request = ask("messages where parser");
    let opts = options(&reader, &request, request.now, 1).unwrap();
    let program = polylogue_query::compile(request.text, &opts).unwrap();
    let params = values(&program.params);
    let count_sql = program.count_sql();
    let (count, without) = reader.query_counts(&count_sql, &params).unwrap();
    let rows = reader.query(&program.sql, &params).unwrap();
    assert!(!rows.is_empty());
    assert_eq!(count, rows.len() as u64);
    assert_eq!(without, 0);
    // A row count needs selection, never the answer's presentation order.
    // Inspect SQLite's executable operations rather than timing a tiny fixture.
    let operations = reader
        .query(&format!("EXPLAIN {count_sql}"), &params)
        .unwrap();
    assert!(operations.iter().all(|op| op["opcode"] != "SorterOpen"));
}

#[test]
fn counting_fused_rows_does_not_rank_or_sort_the_answer() {
    let w = morrow();
    let pool = w.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    let request = ask(
        "messages where parser | union (messages where done) | fuse (messages where cwd:/home/op/morrow)",
    );
    let opts = options(&reader, &request, request.now, 1).unwrap();
    let program = polylogue_query::compile(request.text, &opts).unwrap();
    let params = values(&program.params);
    let count_sql = program.count_sql();
    let (count, without) = reader.query_counts(&count_sql, &params).unwrap();
    let rows = reader.query(&program.sql, &params).unwrap();
    assert!(!rows.is_empty());
    assert_eq!(count, rows.len() as u64);
    assert_eq!(without, 0);
    let operations = reader
        .query(&format!("EXPLAIN {count_sql}"), &params)
        .unwrap();
    assert!(operations.iter().all(|op| op["opcode"] != "SorterOpen"));
}

#[test]
fn counts_preserve_selected_rows_across_limits_offsets_and_shapes() {
    let w = morrow();
    let pool = w.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    for query in [
        "messages where parser",
        "messages where parser | limit 2",
        "messages where parser | offset 1",
        "messages where parser | sort by id | limit 2 | offset 1",
        "messages | sort by id | limit 2 | where written_by:human",
        "messages where parser | distinct",
        "messages where parser | group by written_by | count",
        "sessions where exists tool_call(action:shell) | matches | limit 2",
        "messages where parser | union (messages where done)",
        "messages where parser | union (messages where done) | limit 2",
    ] {
        let request = ask(query);
        let opts = options(&reader, &request, request.now, 1).unwrap();
        let program = polylogue_query::compile(query, &opts).unwrap();
        let params = values(&program.params);
        let rows = reader.query(&program.sql, &params).unwrap();
        let (count, without) = reader.query_counts(&program.count_sql(), &params).unwrap();
        assert_eq!(count, rows.len() as u64, "{query}");
        assert_eq!(without, 0, "{query}");
    }
}

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
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let first = page(&home, None).unwrap();
    assert_eq!(first["count"], 3);
    assert_eq!(first["rows"][0]["id"], "claude-code:a");
    assert_eq!(first["rows"][1]["id"], "claude-code:b");
    let cursor = first["next"].as_str().unwrap();
    // One session sorts before the cursor, one after it.
    home.append("aa.jsonl", &claude_user("aa1", "aa", "new earlier prompt"));
    home.append("d.jsonl", &claude_user("d1", "d", "new later prompt"));
    assert!(engine.run_pass().unwrap().errors.is_empty());
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
            "zone" => request.day = Some(LogicalDay::named("Europe/Warsaw", 0).unwrap()),
            "day-start" => {
                if let Some(day) = &mut request.day {
                    day.start_minutes = 240;
                }
            }
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
        assert!(engine.run_pass().unwrap().errors.is_empty());
        let first = page(&home, None).unwrap();
        let cursor = first["next"].as_str().unwrap();
        if rebuild {
            drop(engine);
            drop(lock);
            polylogue_testkit::delete_index(&home.root);
            let (mut reopened, _lock) = home.engine(64 << 20);
            assert!(reopened.run_pass().unwrap().errors.is_empty());
            assert_eq!(
                page(&home, Some(cursor)).unwrap()["rows"][0]["id"],
                "claude-code:c"
            );
        } else {
            std::fs::write(
                home.file("b.jsonl"),
                claude_user("b1", "b", "after exact rewrite"),
            )
            .unwrap();
            assert!(engine.run_pass().unwrap().errors.is_empty());
            assert_eq!(
                page(&home, Some(cursor)).unwrap()["rows"][0]["id"],
                "claude-code:c"
            );
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
    let change_cursor = first["read_at"]["cursor"].as_str().unwrap();
    assert!(fixture.derive().unwrap() > 0);
    assert_eq!(
        fixture_query(&fixture, "sessions | sort by id", Some(page_cursor), 1).unwrap()["rows"][0]
            ["id"],
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
    let finished = changes["read_at"].clone();
    assert_eq!(fixture.derive().unwrap(), 0);
    let after = fixture_query(
        &fixture,
        &format!(
            "changes where after:{}",
            finished["cursor"].as_str().unwrap()
        ),
        None,
        100,
    )
    .unwrap();
    assert_eq!(after["count"], 0);
    assert_eq!(after["read_at"], finished);
}

#[test]
fn search_text_committed_between_pages_continues_without_reporting_a_session_edit() {
    let (_dir, fixture) = derived_fixture();
    fixture
        .queue_search_text("queued independent search text")
        .unwrap();
    let first = fixture_query(&fixture, "sessions | sort by id", None, 1).unwrap();
    assert_eq!(fixture.commit_search().unwrap(), 1);
    assert_eq!(
        fixture_query(&fixture, "sessions | sort by id", first["next"].as_str(), 1).unwrap()["rows"]
            [0]["id"],
        "claude-code:b"
    );
    let changes = fixture_query(
        &fixture,
        &format!(
            "changes where after:{}",
            first["read_at"]["cursor"].as_str().unwrap()
        ),
        None,
        100,
    )
    .unwrap();
    assert_eq!(
        changes["count"], 0,
        "index availability did not edit session records"
    );
    assert!(
        changes["read_at"]["index"].as_i64().unwrap() > first["read_at"]["index"].as_i64().unwrap()
    );
    assert_eq!(fixture.commit_search().unwrap(), 0);
    assert_eq!(
        fixture_query(&fixture, "sessions", None, 100).unwrap()["read_at"],
        changes["read_at"]
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
fn near_ranked_pages_continue_like_any_other() {
    let w = morrow();
    let s = similarity(&w, "%0003-4000%");
    let near = |_: &str| Ok(std::sync::Arc::clone(&s));
    let text = "kind:* near:\"synthetic ranking\"";
    let mut whole = ask(text);
    whole.page_size = Some(100);
    whole.embeddings = true;
    whole.near = Some(&near);
    let expected = run_q(&w, whole).unwrap();
    assert_eq!(expected["count"], 8);
    let mut first = ask(text);
    first.page_size = Some(3);
    first.embeddings = true;
    first.near = Some(&near);
    let first = run_q(&w, first).unwrap();
    let mut rows = first["rows"].as_array().unwrap().clone();
    let mut cursor = first["next"].as_str().map(str::to_owned);
    while let Some(next) = cursor {
        let mut request = ask(text);
        request.page_size = Some(3);
        request.embeddings = true;
        request.near = Some(&near);
        request.cursor = Some(&next);
        let page = run_q(&w, request).unwrap();
        rows.extend(page["rows"].as_array().unwrap().iter().cloned());
        cursor = page["next"].as_str().map(str::to_owned);
    }
    assert_eq!(&rows, expected["rows"].as_array().unwrap());
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

#[test]
fn blocks_of_one_message_page_in_position_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
    let session = fx
        .session(SessionFacts {
            id: "claude-code:positions",
            started: at(1, 9),
            ..SessionFacts::default()
        })
        .unwrap();
    let message = fx
        .message(
            session,
            MessageFacts {
                id: "claude-code:positions-answer",
                role: "assistant",
                written_by: "model",
                at: at(1, 9),
                ..MessageFacts::default()
            },
        )
        .unwrap();
    // Twelve blocks: a call and its result six times.
    for _ in 0..6 {
        fx.tool_call(
            session,
            message,
            CallFacts {
                tool: "Bash",
                action: "shell",
                command: Some("true"),
                outcome: "ok",
                at: at(1, 9),
                output: Some("done"),
                ..CallFacts::default()
            },
        )
        .unwrap();
    }
    let w = TestArchive { _dir: dir, fx };
    let mut positions = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let mut request = ask("blocks");
        request.page_size = Some(5);
        request.cursor = cursor.as_deref();
        let page = run_q(&w, request).unwrap();
        positions.extend(
            page["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["idx"].as_i64().unwrap()),
        );
        cursor = page["next"].as_str().map(str::to_owned);
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(positions, (0..12).collect::<Vec<_>>());
}

/// Each page of `text` by `size`, read now, continuing after `cursor`.
fn next_page(w: &TestArchive, text: &str, size: u64, cursor: Option<&str>) -> Value {
    let mut request = ask(text);
    request.page_size = Some(size);
    request.cursor = cursor;
    run_q(w, request).unwrap_or_else(|f| panic!("{text}: {} {}", f.kind, f.message))
}

/// The `id` of each row: a session's or a message's.
fn row_ids(v: &Value) -> Vec<String> {
    v["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_owned())
        .collect()
}

/// Sessions written between pages, one sorting before every row and one
/// after, in each order (time, a number, a random order) and page sizes
/// 1, 2 and 7: every row there on the first page appears exactly once, no
/// new row that sorts before the last row shown appears, and each page
/// counts the rows there now.
#[test]
fn rows_written_between_pages_continue_after_the_last_row_shown() {
    for text in [
        "sessions where kind:*",
        "sessions where kind:* | sort by messages desc",
        "sessions where kind:* | sort by random(5)",
        "messages where active:*",
        "messages where active:* | sort by at desc",
    ] {
        for size in [1, 2, 7] {
            let mut w = morrow();
            let before = row_ids(&q(&w, text));
            let first = next_page(&w, text, size, None);
            let mut shown = row_ids(&first);
            let Some(mut cursor) = first["next"].as_str().map(str::to_owned) else {
                continue;
            };
            for (name, day, messages) in [("new-early", -5, 9), ("new-late", 20, 1)] {
                let s =
                    w.fx.session(SessionFacts {
                        id: &format!("claude-code:{name}"),
                        started: at(day, 9),
                        last: at(day, 10),
                        ..SessionFacts::default()
                    })
                    .unwrap();
                for n in 0..messages {
                    w.fx.message(
                        s,
                        msg(&format!("{name}-{n}"), "user", "human", "new", at(day, 9)),
                    )
                    .unwrap();
                }
            }
            let now = row_ids(&q(&w, text));
            let anchor = now.iter().position(|id| Some(id) == shown.last()).unwrap();
            loop {
                let page = next_page(&w, text, size, Some(&cursor));
                assert_eq!(page["count"], now.len() as u64, "{text} by {size}");
                shown.extend(row_ids(&page));
                match page["next"].as_str() {
                    Some(next) => cursor = next.to_owned(),
                    None => break,
                }
            }
            for id in &before {
                let times = shown.iter().filter(|s| *s == id).count();
                assert_eq!(times, 1, "{text} by {size}: {id} shown {times} times");
            }
            for id in now[..anchor].iter().filter(|id| !before.contains(id)) {
                assert!(
                    !shown.contains(id),
                    "{text} by {size}: {id} sorts before the last row shown"
                );
            }
        }
    }
}

/// Pages of a query with relative times, each requested a little later
/// and across midnight, concatenate to the answer at the first page's
/// time: the cursor carries the time the first page resolved.
#[test]
fn pages_of_relative_times_keep_the_first_pages_clock() {
    let w = morrow();
    // 2026-10-02 23:00 UTC.
    let first_now = OCT1 + DAY + 11 * 3_600_000;
    for text in [
        "since:7d",
        "messages where since:3d",
        "messages where at:today",
        "messages where at:yesterday",
        "tool_calls where since:7d",
    ] {
        let mut whole = ask(text);
        whole.now = first_now;
        whole.page_size = Some(1000);
        let whole = run_q(&w, whole).unwrap();
        let mut rows = vec![];
        let mut cursor: Option<String> = None;
        let mut now = first_now;
        loop {
            let mut request = ask(text);
            request.now = now;
            request.page_size = Some(1);
            request.cursor = cursor.as_deref();
            let page = run_q(&w, request).unwrap();
            assert_eq!(page["count"], whole["count"], "{text}");
            rows.extend(page["rows"].as_array().unwrap().iter().cloned());
            cursor = page["next"].as_str().map(str::to_owned);
            if cursor.is_none() {
                break;
            }
            now += 2 * 3_600_000;
        }
        assert_eq!(&rows, whole["rows"].as_array().unwrap(), "{text}");
    }
}
