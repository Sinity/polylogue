use super::*;

#[test]
fn shell_stages_count_occurrences_calls_and_sessions_in_the_query_population() {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
    let west = [
        (
            "shell",
            Some("sh -c 'git status && git status; cargo test'"),
            1,
        ),
        ("shell", Some("git -C /work status"), 1),
        ("shell", None, 1),
        ("shell", Some("for file in *.txt; do cat file; done"), 1),
        ("file_read", Some("git gc"), 1),
        ("shell", Some("rg word"), 2),
    ];
    for (id, calls) in [
        ("claude-code:west", west.as_slice()),
        (
            "claude-code:east",
            &[("shell", Some("git status && cargo test"), 1)],
        ),
        (
            "codex:other",
            &[("shell", Some("git status && nix build"), 1)],
        ),
    ] {
        let session = fx
            .session(SessionFacts {
                id,
                started: at(1, 12),
                last: at(calls.last().unwrap().2, 12),
                ..SessionFacts::default()
            })
            .unwrap();
        for (i, (action, command, day)) in calls.iter().enumerate() {
            let when = at(*day, 12);
            let message_id = scoped(id, &format!("{id}-stage-{i}"));
            let message = fx
                .message(session, msg(&message_id, "assistant", "model", "", when))
                .unwrap();
            fx.tool_call(
                session,
                message,
                CallFacts {
                    tool: if *action == "shell" { "Bash" } else { "Read" },
                    action,
                    command: *command,
                    path: None,
                    outcome: "unknown",
                    exit_code: None,
                    at: when,
                    output: None,
                },
            )
            .unwrap();
        }
    }
    let archive = TestArchive { _dir: dir, fx };
    let all = q(&archive, "sessions | measure shell-stages");
    let all = &all["data"];
    assert_eq!(all["shell_calls"], 7);
    assert_eq!(all["calls_without_recognized_stages"], 2);
    assert_eq!(
        all["stages"]["git status"],
        serde_json::json!({"occurrences":5,"calls":4,"sessions":3})
    );
    assert_eq!(all["stages"]["nix build"]["occurrences"], 1);
    assert_eq!(all["stages"]["rg"]["occurrences"], 1);
    let selected = q(
        &archive,
        "sessions where origin:claude-code | measure shell-stages",
    );
    assert_eq!(selected["data"]["shell_calls"], 6);
    assert_eq!(selected["data"]["calls_without_recognized_stages"], 2);
    assert_eq!(
        selected["data"]["stages"],
        serde_json::json!({
            "git status":{"occurrences":4,"calls":3,"sessions":2},
            "cargo test":{"occurrences":2,"calls":2,"sessions":2},
            "rg":{"occurrences":1,"calls":1,"sessions":1}
        })
    );
    let day = q(
        &archive,
        "sessions where since:2026-10-02 until:2026-10-02 | measure shell-stages",
    );
    assert_eq!(day["data"]["shell_calls"], 1);
    assert_eq!(day["data"]["calls_without_recognized_stages"], 0);
    assert_eq!(
        day["data"]["stages"],
        serde_json::json!({"rg":{"occurrences":1,"calls":1,"sessions":1}})
    );
}

#[test]
fn catalog_measures_use_selected_sessions_and_typed_fields() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let all = q(&archive, "sessions | measure facets:deep");
    let selected = q(
        &archive,
        "sessions where origin:claude-code | measure facets:deep",
    );
    assert!(
        selected["data"]["summary"]["selected_sessions"]
            .as_u64()
            .unwrap()
            < all["data"]["summary"]["selected_sessions"]
                .as_u64()
                .unwrap()
    );
    assert!(
        selected["data"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v["idf"].is_null() || v["idf"].is_number())
    );
    let time = q(&archive, "sessions where origin:claude-code | measure time");
    assert!(
        time["data"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["session"].as_str().unwrap().starts_with("claude-code:"))
    );
    assert!(
        q(&archive, "sessions | measure \"compare:origin,words\"")["data"]["sessions"].is_array()
    );
    let time = run_q(&archive, ask("sessions | measure time:unsupported")).unwrap_err();
    assert_eq!(time.code(), "bad_request");
    assert_eq!(time.message, "time takes no arguments");
    let comparison = run_q(&archive, ask("sessions | measure compare:unsupported")).unwrap_err();
    assert_eq!(comparison.code(), "bad_value");
    let reason = comparison.reason().unwrap();
    assert_eq!(reason.name, "unknown_choice");
    assert_eq!(reason.fields["field"], "compare");
    assert_eq!(reason.fields["value"], "unsupported");
}

#[test]
fn outlook_setting_method_and_integer_money_reach_the_public_query() {
    let mut archive = morrow();
    archive.fx.setting("plan.example",r#"{"name":"max-20x","origin":"claude-code","cycle_anchor_day":1,"price_nusd":200000000000,"budget":{"population":"responses","unit":"input_tokens","amount":1000000,"overage_rate_nusd_per_unit":null}}"#).unwrap();
    let answer = q(
        &archive,
        "sessions | measure outlook:example:trailing-7d-mean",
    );
    assert_eq!(answer["data"]["projection_method"], "trailing-7d-mean");
    assert_eq!(answer["data"]["trailing_observed_days"], 7);
    assert_eq!(answer["data"]["estimate"], true);
    assert_eq!(
        answer["data"]["segments"][0]["plan"]["price_nusd"],
        200_000_000_000_i64
    );
    assert_eq!(
        q(&archive, "sessions | measure outlook:missing")["data"]["reason"],
        "plan_not_configured"
    );
    assert_eq!(
        q(&archive, "sessions | measure outlook")["data"]["reason"],
        "provider_required"
    );
}

#[test]
fn time_comparison_and_facets_page_with_the_query_cursor() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    for (text, field) in [
        ("sessions | sort by id | measure time", "sessions"),
        ("sessions | sort by id | measure compare", "sessions"),
        ("sessions | measure facets:deep", "values"),
    ] {
        let mut cursor = None::<String>;
        let mut ids = std::collections::BTreeSet::new();
        let count = loop {
            let mut request = ask(text);
            request.page_size = Some(1);
            request.cursor = cursor.as_deref();
            let page = run_q(&archive, request).unwrap();
            let rows = page["data"][field].as_array().unwrap();
            assert!(rows.len() <= 1);
            for row in rows {
                let id = if field == "values" {
                    format!("{}:{}", row["family"], row["value"])
                } else {
                    row["session"].as_str().unwrap().to_owned()
                };
                assert!(ids.insert(id));
            }
            match page["next"].as_str() {
                Some(next) => {
                    // The test helper puts the typed outcome beside the document.
                    let mut document = page.clone();
                    document.as_object_mut().unwrap().remove("outcome");
                    assert!(
                        crate::commands::declarations()
                            .render_text(
                                "rows",
                                &command_surface::Answer::ok(
                                    serde_json::to_value(
                                        crate::protocol::line::LineAnswer::Listing(Box::new(
                                            serde_json::from_value(document).unwrap()
                                        ))
                                    )
                                    .unwrap()
                                ),
                                command_surface::TextStyle::Plain
                            )
                            .unwrap()
                            .contains(&format!("--cursor {next}"))
                    );
                    cursor = Some(next.into());
                }
                None => break page["count"].as_u64().unwrap(),
            }
        };
        assert_eq!(ids.len() as u64, count);
        assert!(count > 1);
    }
}

#[test]
fn facets_use_the_selection_units_default_population() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let ordinary = q(&archive, "sessions");
    let facets = q(&archive, "sessions | measure facets");
    assert_eq!(
        facets["data"]["summary"]["selected_sessions"],
        ordinary["count"]
    );
    assert_eq!(
        facets["data"]["summary"]["global_sessions"],
        ordinary["count"]
    );
    let subagents = q(&archive, "sessions where kind:subagent | measure facets");
    assert!(
        subagents["data"]["summary"]["global_sessions"]
            .as_u64()
            .unwrap()
            > ordinary["count"].as_u64().unwrap()
    );
    let empty = q(&archive, "sessions where origin:absent | measure facets");
    assert_eq!(empty["data"]["summary"]["outcome"], "empty");
    assert!(
        empty["data"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v["selected_count"] == 0)
    );
}

/// The archive's logical sessions are #1, #2 with its subagent #3 and
/// resumed #4, #5 with its fork #6, #7 and #8 (examples/morrow/FACTS.md).
#[test]
fn facet_counts_count_logical_sessions() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let count = |query: &str, family: &str, value: &str| {
        let facets = q(&archive, query);
        let row = facets["data"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["family"] == family && row["value"] == value)
            .unwrap_or_else(|| panic!("no {family} {value} in {facets}"))
            .clone();
        (row["selected_count"].clone(), row["global_count"].clone())
    };
    // #1 and the work of #2, #3 and #4 are two claude-code logical sessions.
    assert_eq!(
        count(
            "sessions | measure facets:origins",
            "origins",
            "claude-code"
        ),
        (2.into(), 2.into())
    );
    assert_eq!(
        count("sessions | measure facets:origins", "origins", "codex"),
        (1.into(), 1.into())
    );
    // Subagent #3 alone answers with this model; its logical session is #2's.
    assert_eq!(
        count(
            "sessions | measure facets:models",
            "models",
            "claude-opus-5-5"
        ),
        (1.into(), 1.into())
    );
    // A physical listing counts each physical session itself.
    let physical = q(&archive, "sessions where physical:true origin:claude-code");
    assert!(physical["count"].as_u64().unwrap() > 2);
    assert_eq!(
        count(
            "sessions where physical:true | measure facets:origins",
            "origins",
            "claude-code"
        )
        .0,
        physical["count"]
    );
}

#[test]
fn paged_measures_preserve_query_order_when_display_fields_omit_the_id() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let mut request = ask("sessions | sort by id desc");
    request.page_size = Some(2);
    let ordinary = run_q(&archive, request).unwrap();
    let expected: Vec<_> = ordinary["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect();
    let fields = vec!["title".into()];
    for name in ["time", "compare"] {
        let text = format!("sessions | sort by id desc | measure {name}");
        let mut request = ask(&text);
        request.page_size = Some(2);
        request.fields = &fields;
        let measured = run_q(&archive, request).unwrap();
        let ids: Vec<_> = measured["data"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["session"].as_str().unwrap())
            .collect();
        assert_eq!(ids, expected, "{name}");
    }
}

#[test]
fn comparisons_keep_whole_selection_differences_on_every_page() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let text = "sessions | sort by id | measure \"compare:origin,words,wall_span\"";
    let mut request = ask(text);
    request.page_size = Some(1);
    let first = run_q(&archive, request).unwrap();
    assert!(
        first["data"]["varies"]
            .as_array()
            .unwrap()
            .contains(&json!("origin"))
    );
    let mut request = ask(text);
    request.page_size = Some(1);
    request.cursor = first["next"].as_str();
    let next = run_q(&archive, request).unwrap();
    assert_eq!(first["data"]["varies"], next["data"]["varies"]);
    assert_eq!(first["data"]["unknown"], next["data"]["unknown"]);
}

#[test]
fn measures_over_near_ranked_sessions_keep_page_counts() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let vectors = super::generated::vectors_for(&archive, |_| true);
    let near = |_: &str| Ok(std::sync::Arc::clone(&vectors));
    for (name, field) in [
        ("time", "sessions"),
        ("compare", "sessions"),
        ("facets", "values"),
    ] {
        let text = format!("sessions where similar:\"synthetic ranking\" | measure {name}");
        let mut request = ask(&text);
        request.embeddings = true;
        request.near = Some(&near);
        request.page_size = Some(1);
        let answer = run_q(&archive, request).unwrap();
        assert!(answer["count"].as_u64().unwrap() > 1);
        assert_eq!(answer["data"][field].as_array().unwrap().len(), 1);
        let mut request = ask(&text);
        request.embeddings = true;
        request.near = Some(&near);
        request.page_size = Some(1);
        request.cursor = answer["next"].as_str();
        let next = run_q(&archive, request).unwrap();
        assert_eq!(next["count"], answer["count"]);
        assert_eq!(next["data"][field].as_array().unwrap().len(), 1);
    }
}

#[test]
fn stored_counts_and_time_match_morrow_after_the_real_loop() {
    let archive = polylogue_testkit::Morrow::build(polylogue_testkit::Set::Main);
    let pool = archive.reads();
    let reader = pool.reader().unwrap();
    let query =
        r#"sessions where id:"claude-code:7f3a2c10-0002-4000-8000-000000000002" | measure time"#;
    let answer = run(&reader, ask(query)).unwrap();
    let time = &answer["data"]["sessions"][0];
    assert_eq!(time["span"]["wall_ms"], 4_500_000);
    assert_eq!(time["span"]["clustered_ms"], 3_720_000);
    assert_eq!(time["span"]["tool_union_ms"], 1_704_000);
    assert_eq!(time["span"]["tool_sum_ms"], 3_434_000);
    let parties = time["parties"].as_array().unwrap();
    let party = |name: &str| {
        parties
            .iter()
            .filter(|l| l["party"] == name)
            .map(|l| l["ms"].as_i64().unwrap())
            .sum::<i64>()
    };
    assert_eq!(
        (
            party("tool"),
            party("model"),
            party("human"),
            party("unattributed")
        ),
        (1_704_000, 2_348_000, 448_000, 0)
    );
    let fields = vec![
        "messages".into(),
        "human_prompts".into(),
        "tool_calls".into(),
        "errors".into(),
        "wall_span".into(),
        "clustered_time".into(),
    ];
    let mut request =
        ask(r#"sessions where id:"claude-code:7f3a2c10-0002-4000-8000-000000000002""#);
    request.fields = &fields;
    let answer = run(&reader, request).unwrap();
    assert_eq!(answer["rows"][0]["messages"], 18);
    assert_eq!(answer["rows"][0]["human_prompts"], 3);
    assert_eq!(answer["rows"][0]["tool_calls"], 13);
    assert_eq!(answer["rows"][0]["errors"], 1);
}

#[test]
fn aggregate_counts_report_pending_and_compute_current_rows() {
    let archive = morrow();
    let answer = q(&archive, "sessions | agg sum(messages)");
    assert!(answer["rows"][0]["sum_messages"].as_i64().unwrap() > 0);
    assert!(answer["rows"][0]["pending_messages"].as_i64().unwrap() > 0);
}

#[test]
fn clustered_fields_print_the_parameter_in_query_output() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let result = q(&archive, "sessions | agg p50(clustered_time)");
    assert!(
        result["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n.as_str().is_some_and(|n| n.contains("5-minute gap")))
    );
    let printed = crate::cli::print::query(
        &serde_json::from_value(super::document(
            &archive,
            "sessions | agg p50(clustered_time)",
        ))
        .unwrap(),
        command_surface::TextStyle::Plain,
    );
    assert!(printed.contains("5-minute gap"));
}

#[test]
fn tool_time_aggregates_and_default_comparisons_report_unmeasured_calls() {
    use polylogue_store::fixture::writer::{WriterArchive, bash};
    let mut archive = WriterArchive::new();
    let mut measured = bash("tool-times", 1, 0, "cargo test", true, "ok");
    measured[1].at = polylogue_model::Timestamp::from_unix_ms(120);
    let mut unmeasured = bash("tool-times", 2, 180, "cargo build", true, "ok");
    unmeasured[1].at = polylogue_model::Timestamp::from_unix_ms(180);
    measured.extend(unmeasured);
    archive.write("tool-times", measured);
    let pool = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let answer = run(
        &reader,
        ask("sessions | agg sum(tool_time), sum(tool_time_union)"),
    )
    .unwrap();
    assert_eq!(answer["rows"][0]["sum_tool_time"], 120);
    assert_eq!(answer["rows"][0]["unmeasured_calls_tool_time"], 1);
    assert_eq!(answer["rows"][0]["unmeasured_calls_tool_time_union"], 1);
    let answer = run(&reader, ask("sessions | measure compare")).unwrap();
    assert_eq!(answer["data"]["sessions"][0]["values"]["tool_untimed"], 1);
}

#[test]
fn unknown_session_endings_and_clock_sources_match_unknown_conditions() {
    let mut archive = morrow();
    archive
        .fx
        .session(SessionFacts {
            id: "codex:untimed",
            ..Default::default()
        })
        .unwrap();
    archive.fx.derive().unwrap();
    // Each condition selects the untimed session and not every session.
    let total = q(&archive, "sessions where kind:*")["count"]
        .as_u64()
        .unwrap();
    for condition in [
        "ends_with:unknown",
        "time_source:unknown",
        // The stored word for an untimed session is also accepted as a value.
        "time_source:absent",
    ] {
        let result = q(&archive, &format!("sessions where kind:* {condition}"));
        assert!(
            ids(&result).contains(&"codex:untimed".to_owned()),
            "{condition}: {result}"
        );
        assert!(result["count"].as_u64().unwrap() < total, "{condition}");
    }
}

fn session_pk(archive: &TestArchive, like: &str) -> i64 {
    let pool = archive.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    reader
        .query(
            "SELECT pk FROM session WHERE id LIKE ?1",
            &[SqlValue::Text(like.into())],
        )
        .unwrap()[0]["pk"]
        .as_i64()
        .unwrap()
}

#[test]
fn sessions_day_includes_subagent_usage() {
    let archive = morrow();
    // #2 on Oct 2 spawned subagent #3, the only session that used
    // claude-opus-5-5: the day's usage over main sessions includes it.
    let v = q(
        &archive,
        "sessions where since:2026-10-02 until:2026-10-02 | measure usage",
    );
    let models: Vec<&str> = v["data"]["by_model"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["model"].as_str())
        .collect();
    assert!(models.contains(&"claude-opus-5-5"), "{v}");
    assert_eq!(
        ids(&q(&archive, "since:2026-10-02 until:2026-10-02")),
        ["#2"]
    );
}

#[test]
fn usage_window_excludes_responses_outside() {
    let mut archive = morrow();
    let day = "sessions where since:2026-10-01 until:2026-10-01 | measure usage";
    let before = q(&archive, day);
    // A response in #1 on the next day: #1 is still selected (its recorded
    // activity is Oct 1), and the response stays out of Oct 1's figures.
    let s1 = session_pk(&archive, "%0001-4000%");
    archive
        .fx
        .message(
            s1,
            MessageFacts {
                cost_nusd: Some(990_000_000),
                ..msg(
                    &scoped("claude-code:7f3a2c10-0001-4000-8000-000000000001", "m1late"),
                    "assistant",
                    "model",
                    "late",
                    at(2, 10),
                )
            },
        )
        .unwrap();
    let after = q(&archive, day);
    assert_eq!(
        after["data"]["known_response_messages"],
        before["data"]["known_response_messages"]
    );
    assert_eq!(after["data"]["api_cost"], before["data"]["api_cost"]);
    // The whole archive does count it.
    let all = q(&archive, "sessions | measure usage");
    assert!(
        all["data"]["known_response_messages"].as_u64().unwrap()
            > after["data"]["known_response_messages"].as_u64().unwrap()
    );
}

#[test]
fn usage_equals_sum_across_surfaces() {
    let mut archive = morrow();
    let s1 = session_pk(&archive, "%0001-4000%");
    // A priced response on an abandoned branch is still one of the
    // session's own responses; a priced copy shown elsewhere is not.
    let branch = scoped(
        "claude-code:7f3a2c10-0001-4000-8000-000000000001",
        "m1branch",
    );
    let copy = scoped("claude-code:7f3a2c10-0001-4000-8000-000000000001", "m1copy");
    for (id, active, cost) in [
        (&branch, Some(false), 70_000_000),
        (&copy, None, 30_000_000),
    ] {
        archive
            .fx
            .message(
                s1,
                MessageFacts {
                    cost_nusd: Some(cost),
                    on_active_path: active,
                    ..msg(id, "assistant", "model", "another answer", at(1, 13))
                },
            )
            .unwrap();
    }
    polylogue_store::archive::open_write(
        &archive.fx.archive.index_db(),
        polylogue_store::archive::cache::SETUP,
    )
    .unwrap()
    .execute(
        "UPDATE message SET duplicate_of = ?2, duplicate_reason = 'synthetic copy' WHERE id = ?1",
        [copy.as_str(), branch.as_str()],
    )
    .unwrap();
    // Measures write money as `{"nusd": N, "usd": D}`, `compare` as
    // nano-dollars and query columns as dollars; all compared in nano-dollars.
    // One day's sessions, #1 alone, which spawned no subagent: usage adds
    // spawned sessions' responses to the sessions that spawned them.
    let day = "sessions where since:2026-10-01 until:2026-10-01";
    let measured = q(&archive, &format!("{day} | measure usage"));
    let usage = measured["data"]["api_cost"]["known"]["nusd"]
        .as_i64()
        .unwrap_or_else(|| panic!("{measured}"));
    let summed = q(&archive, &format!("{day} | agg sum(cost)"));
    let field = summed["rows"][0]["sum_cost"]
        .as_f64()
        .map(|usd| (usd * 1e9).round() as i64)
        .unwrap_or_else(|| panic!("{summed}"));
    let compared: i64 =
        q(&archive, &format!("{day} | measure compare:cost_nusd"))["data"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s["values"]["cost_nusd"].as_i64())
            .sum();
    assert_eq!(summed["rows"][0]["sum_cost"], 1.33, "{summed}");
    assert_eq!((field, compared), (usage, usage));
}

#[test]
fn measured_population_same_with_and_without_conditions() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let conditioned = "sessions where origin:claude-code | measure usage";
    let before = q(&archive, conditioned);
    archive
        .fx
        .note(
            "hidden",
            None,
            None,
            "session:claude-code:7f3a2c10-0002-4000-8000-000000000002",
        )
        .unwrap();
    // Hiding is a display preference: the measured population stays, with
    // or without conditions, and says how many hidden sessions it includes.
    let after = q(&archive, conditioned);
    let everything = q(&archive, "sessions | measure usage");
    assert_eq!(before["data"]["hidden_sessions"], 0);
    assert_eq!(after["data"]["hidden_sessions"], 1);
    assert_eq!(everything["data"]["hidden_sessions"], 1);
    for field in ["known_response_messages", "api_cost", "tokens"] {
        assert_eq!(after["data"][field], before["data"][field], "{field}");
    }
}

#[test]
fn usage_includes_continuation_once() {
    let w = logical_work();
    let usage = q(&w, "sessions where cwd:/w/c | measure usage");
    assert_eq!(usage["data"]["known_response_messages"], 5);
    // Explicitly selecting both a root and its child still counts own responses once.
    let explicit = q(
        &w,
        "sessions where physical:true (cwd:/w/a OR cwd:/w/c) | measure usage",
    );
    assert_eq!(explicit["data"]["known_response_messages"], 5);
    let day = q(
        &w,
        "sessions where cwd:/w/c since:2026-10-03 until:2026-10-03 | measure usage",
    );
    assert_eq!(day["data"]["known_response_messages"], 1);
}

#[test]
fn failing_measure_keeps_array_data_and_prints_in_the_resolved_day_zone() {
    let mut w = morrow();
    w.fx.derive().unwrap();
    let pool = w.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut request = ask("sessions | measure failing");
    request.context = fixture_context(OCT1 + 7 * DAY, "Europe/Warsaw", 0);
    let answer = crate::daemon::query::run(&reader, request).unwrap();
    let crate::protocol::query::Answer::Measure(measure) = &answer.value else {
        panic!("expected a measurement");
    };
    assert!(measure.data.is_array());
    assert!(!measure.data.as_array().unwrap().is_empty());
    assert_eq!(measure.resolved.context.clock.day.zone, "Europe/Warsaw");
    let text = crate::cli::print::query(&answer.value, command_surface::TextStyle::Plain);
    assert!(text.contains("+02:00"), "{text}");
}

/// The responses `| measure responses` lists are the ones `usage` sums:
/// the same count and known cost, in time order, inside the day's bounds.
#[test]
fn responses_are_the_rows_behind_usage() {
    let archive = morrow();
    for selection in [
        "sessions",
        "sessions where since:2026-10-02 until:2026-10-02",
        "origin:claude-code",
    ] {
        let usage = q(&archive, &format!("{selection} | measure usage"));
        let listed = q(&archive, &format!("{selection} | measure responses"));
        assert!(listed["next"].is_null(), "{listed}");
        let rows = listed["data"]["responses"].as_array().unwrap();
        assert_eq!(
            rows.len() as u64,
            usage["data"]["known_response_messages"].as_u64().unwrap(),
            "{selection}: {listed}"
        );
        let known: i64 = rows.iter().filter_map(|r| r["cost"]["nusd"].as_i64()).sum();
        assert_eq!(
            Some(known),
            usage["data"]["api_cost"]["known"]["nusd"].as_i64(),
            "{selection}"
        );
        let times: Vec<i64> = rows.iter().map(|r| r["at"].as_i64().unwrap()).collect();
        assert!(
            times.windows(2).all(|w| w[0] <= w[1]),
            "{selection}: {times:?}"
        );
    }
    // Oct 2 is #2 and its subagent #3, the only session that used
    // claude-opus-5-5.
    let day = "sessions where since:2026-10-02 until:2026-10-02";
    let opus = q(
        &archive,
        &format!("{day} | measure responses:claude-opus-5-5"),
    );
    let rows = opus["data"]["responses"].as_array().unwrap();
    assert!(!rows.is_empty());
    assert!(
        rows.iter().all(|r| r["model"] == "claude-opus-5-5"
            && r["session"].as_str().unwrap().contains("0003-4000"))
    );
    // Morrow's #2 has one response with a stated cost and no recorded model.
    let unknown = q(&archive, &format!("{day} | measure responses:unknown"));
    let unknown = unknown["data"]["responses"].as_array().unwrap();
    assert_eq!(
        unknown
            .iter()
            .map(|r| (r["id"].as_str().unwrap(), r["model"].is_null()))
            .collect::<Vec<_>>(),
        [("claude-code:m2a0", true)]
    );
}
