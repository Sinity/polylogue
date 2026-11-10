use super::*;

#[test]
fn catalog_measures_use_selected_sessions_and_typed_fields() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    for removed in ["profiles", "pace", "similar", "correlation", "distribution"] {
        assert!(run_q(&archive, ask(&format!("sessions | measure {removed}"))).is_err());
    }
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
    assert!(run_q(&archive, ask("sessions | measure time:unsupported")).is_err());
    assert!(run_q(&archive, ask("sessions | measure compare:unsupported")).is_err());
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
                                "query",
                                &command_surface::Answer::ok(document),
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
    let lanes = time["lanes"].as_array().unwrap();
    let lane = |name: &str| {
        lanes
            .iter()
            .filter(|l| l["lane"] == name)
            .map(|l| l["ms"].as_i64().unwrap())
            .sum::<i64>()
    };
    assert_eq!(
        (
            lane("tool"),
            lane("model"),
            lane("human"),
            lane("unattributed")
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
        &super::document(&archive, "sessions | agg p50(clustered_time)"),
        false,
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
    for field in ["ends_with", "time_source"] {
        let result = q(
            &archive,
            &format!(r#"sessions where id:"codex:untimed" {field}:unknown"#),
        );
        assert_eq!(result["count"], 1);
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
