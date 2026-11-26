use super::*;

fn explain(expression: Option<&str>) -> crate::test_daemon::Call {
    crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection":expression,"explain":true}),
    )
}

#[test]
fn declaration_pages_preserve_every_entry() {
    for subject in SUBJECTS {
        let expected = entries(subject, None).unwrap();
        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let page = answer(subject, None, Some(1), cursor.as_deref()).unwrap();
            assert_eq!(page["subject"], *subject);
            assert_eq!(page["total"], expected.len());
            assert!(page.get("syntax").is_none());
            seen.extend(page["entries"].as_array().unwrap().iter().cloned());
            let next = page["next"].as_str().map(str::to_owned);
            assert!(next.is_none() || next != cursor);
            cursor = next;
            if cursor.is_none() {
                break;
            }
            assert!(seen.len() <= expected.len());
        }
        assert_eq!(seen, expected);
    }
    let page = answer("fields", Some("messages"), Some(u64::MAX), None).unwrap();
    assert_eq!(
        page["entries"].as_array().unwrap().len(),
        fields::names(Unit::Messages).count()
    );
    let refusal = |result: Result<Value, crate::daemon::Failure>| result.unwrap_err().error.message;
    assert_eq!(
        refusal(answer("fields", None, Some(0), None)),
        "page_size must be positive so the declarations can advance"
    );
    assert_eq!(
        refusal(answer("fields", None, None, Some("units//1"))),
        "cursor belongs to another subject or is invalid"
    );
    assert_eq!(
        refusal(answer("fields", Some("bananas"), None, None)),
        "unknown unit \"bananas\""
    );
    let capabilities = entries("capabilities", None).unwrap();
    let views: Vec<_> = capabilities
        .iter()
        .filter(|v| v["kind"] == "read_view")
        .map(|v| v["name"].as_str().unwrap())
        .collect();
    assert_eq!(views, polylogue_model::read::View::NAMES);
}

#[test]
fn syntax_lists_time_functions_and_the_sort_transform_and_distinct_count() {
    let page = answer("syntax", None, None, None).unwrap();
    let keywords = page["entries"][0]["keywords"].as_array().unwrap();
    for function in [
        "day",
        "hour",
        "week",
        "weekday",
        "month",
        "month_day",
        "program",
    ] {
        assert!(keywords.contains(&json!(function)), "missing {function}");
    }
    assert!(
        page["entries"][0]["transforms"]
            .as_array()
            .unwrap()
            .contains(&json!("sort by"))
    );
    assert!(
        page["entries"][0]["aggregates"]
            .as_array()
            .unwrap()
            .contains(&json!("count_distinct"))
    );
}

#[test]
fn query_explanation_uses_archive_definitions_session_and_day_without_writes() {
    let day = query_language::time::DaySetting::new(
        jiff::tz::TimeZone::get("Pacific/Kiritimati").unwrap(),
        jiff::civil::Time::constant(7, 30, 0, 0),
    )
    .unwrap();
    let w = crate::test_daemon::start_with_clock_and_day(
        "synthetic query explanation",
        polylogue_model::WallClock::fixed("2026-10-29T08:00:00Z".parse().unwrap()),
        day,
    );
    let save = crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection":"self","steps":[{"verb":"save","name":"mine"}]}),
    )
    .id("save-explain");
    let caller = polylogue_model::authorship::Authorship {
        author: polylogue_model::Author::Agent(Some(
            polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
        )),
        via: polylogue_model::authorship::Channel::Mcp,
    };
    let as_caller = |call: crate::test_daemon::Call| {
        call.authorship(caller.clone())
            .calling_session("claude-code:s1")
    };
    crate::test_answers::ok(as_caller(save).answer(&w.daemon));
    let kept = crate::test_answers::step(
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection":"sessions","steps":[{"verb":"save"}]}),
        )
        .id("keep-explain")
        .answer(&w.daemon),
    );
    let crate::protocol::line::StepAnswer::Saved(kept) = kept else {
        panic!("expected kept result")
    };
    let position = w.daemon.reads.reader().unwrap().change_position().unwrap();
    for expression in [
        "tool_calls where outcome:error | group by session | count".to_owned(),
        "sessions where exists tool_calls(outcome:error)".into(),
        "messages where written_by:human chrono".into(),
        "sessions where cost:unknown".into(),
        "tool_calls | union (outcome:error)".into(),
        "sessions | tool_calls | union (outcome:error)".into(),
        "query:mine".into(),
        kept.kept.result.clone(),
    ] {
        let result =
            crate::test_answers::ok(as_caller(explain(Some(&expression))).answer(&w.daemon));
        let explanation = &result["reply"]["plan"]["explanation"];
        let program = &explanation["program"];
        // A unit's rows are read key first; a derived relation whole.
        let statements = &program["statements"];
        let selecting = match statements["kind"].as_str().unwrap() {
            "keyed" => &statements["keys"],
            "derived" => &statements["rows"],
            kind => panic!("{expression}: {kind}"),
        };
        assert!(!selecting.as_str().unwrap().is_empty(), "{expression}");
        assert!(explanation["parsed"].is_object());
        let clock: query_language::time::ClockWire =
            serde_json::from_value(program["clock"].clone()).unwrap();
        assert_eq!(
            clock.now,
            "2026-10-29T08:00:00Z".parse::<jiff::Timestamp>().unwrap()
        );
        assert_eq!(clock.day.zone, "Pacific/Kiritimati");
        assert_eq!(clock.day.start, jiff::civil::Time::constant(7, 30, 0, 0));
        assert!(program["incomplete"].is_array());
        assert!(program["params"].is_array());
        if expression == "query:mine" {
            assert_eq!(program["definitions"][0]["name"], "mine");
            assert!(program["params"].to_string().contains("claude-code:s1"));
        }
        if expression.starts_with("tool_calls") {
            assert_eq!(explanation["units"], json!(["tool_calls"]));
        }
        if expression.contains("exists") || expression.starts_with("sessions |") {
            assert_eq!(explanation["units"], json!(["sessions", "tool_calls"]));
        }
    }
    assert_eq!(
        w.daemon.reads.reader().unwrap().change_position().unwrap(),
        position
    );
    let none = crate::test_answers::ok(explain(None).answer(&w.daemon));
    assert_eq!(none["reply"]["form"], "none");
    assert!(none["reply"]["plan"].is_null());
    for expression in [
        Some(""),
        Some("messages where nonsense:x"),
        Some("bananas where role:user"),
    ] {
        assert!(
            matches!(
                (explain(expression)).answer(&w.daemon),
                command_surface::Answer {
                    outcome: command_surface::Outcome::Error(_),
                    ..
                }
            ),
            "{expression:?}"
        );
    }
}

#[test]
fn derived_pages_and_fields_expose_the_same_validity_and_parameters() {
    let page = answer("derived", None, Some(100), None).unwrap();
    let entries = page["entries"].as_array().unwrap();
    assert_eq!(entries.len(), polylogue_model::derived::all().len());
    assert!(entries.iter().any(|e| e["name"] == "session_span"
        && e["validity"] == "parameterized"
        && e["parameters"][0]["value"] == 300000));
    let page = answer("fields", Some("sessions"), Some(1000), None).unwrap();
    let field = page["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == "clustered_time")
        .unwrap();
    assert_eq!(field["construct"]["catalog"], "K15");
}

#[test]
fn capabilities_list_each_operation_with_unlisted_arguments_and_answer() {
    let capabilities = entries("capabilities", None).unwrap();
    let read = capabilities
        .iter()
        .find(|entry| entry["kind"] == "command" && entry["name"] == "rows")
        .unwrap();
    assert_eq!(read["tool"], "polylogue_read");
    let unlisted = read["unlisted"].as_array().unwrap();
    assert!(
        unlisted.contains(&json!("return_selection")),
        "{unlisted:?}"
    );
    assert!(read["arguments"]["properties"]["steps"].is_object());
    assert!(read["answer"].is_object());
    let status = capabilities
        .iter()
        .find(|entry| entry["kind"] == "command" && entry["name"] == "status")
        .unwrap();
    assert_eq!(status["tool"], "polylogue_read");
    let envelope = capabilities
        .iter()
        .find(|entry| entry["kind"] == "answer")
        .unwrap();
    assert_eq!(
        envelope["schema"],
        schemars::schema_for!(command_surface::Answer<Value>).to_value()
    );
}

#[test]
fn transforms_subject_lists_the_pipe_stages() {
    let page = answer("transforms", None, Some(100), None).unwrap();
    let completions: Vec<_> = page["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry["completion"].as_str())
        .collect();
    for stage in ["where", "sort by", "group by"] {
        assert!(completions.contains(&stage), "{completions:?}");
    }
}

#[test]
fn transforms_list_every_measure_with_its_help() {
    let entries = entries("transforms", None).unwrap();
    for m in crate::daemon::measure::MEASURES {
        let name = format!("measure {}", m.name);
        let entry = entries
            .iter()
            .find(|entry| entry["name"] == name)
            .unwrap_or_else(|| panic!("{name} is not listed"));
        assert_eq!(entry["help"], m.help);
    }
}

/// `help WORD` for a removed word prints its row of docs/vocabulary.md.
#[test]
fn a_removed_word_names_what_says_it_now() {
    assert_eq!(
        removed_word("timeline").as_deref(),
        Some(
            "timeline, cost, measure NAME (commands): no longer part of the surface; now queries with | measure\n"
        )
    );
    assert_eq!(removed_word("cost"), removed_word("timeline"));
    assert!(
        removed_word("chat list")
            .unwrap()
            .ends_with("the unit chat_actions\n")
    );
    assert_eq!(removed_word("sessions"), None);
}
