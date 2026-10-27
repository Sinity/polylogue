use super::*;

#[test]
fn new_measures_use_selected_sessions_and_profile_fields() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    assert!(run_q(&archive, ask("sessions | measure profiles:unsupported")).is_err());
    let all = q(&archive, "sessions | measure facets:deep");
    assert!(all["data"]["global"]["total_sessions"].as_u64().unwrap() > 1);
    let selected = q(
        &archive,
        "sessions where origin:claude-code | measure facets:deep-no-idf",
    );
    assert!(selected["data"]["idf"].as_object().unwrap().is_empty());
    assert!(
        selected["data"]["selected"]["total_sessions"]
            .as_u64()
            .unwrap()
            < all["data"]["global"]["total_sessions"].as_u64().unwrap()
    );
    let all_profiles = q(&archive, "sessions | measure profiles");
    assert!(
        all_profiles["data"]["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["usage"]["parents"]["complete"] == false
                && p["usage"]["with_parents"].is_null()
                && p["usage"]["own"].is_object())
    );
    let profiles = q(
        &archive,
        "sessions where workflow_shape:chat | measure profiles",
    );
    assert!(!profiles["data"]["profiles"].as_array().unwrap().is_empty());
    assert!(
        profiles["data"]["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["facts"]["workflow_shape"] == "chat")
    );
    let pace = q(
        &archive,
        "sessions where origin:claude-code | measure pace:60",
    );
    assert!(pace["data"]["gaps"].is_array());
    assert!(
        pace["data"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["session"].as_str().unwrap().starts_with("claude-code:"))
    );
    let distribution = q(
        &archive,
        "sessions | measure distribution:median_tool_call_ms",
    );
    assert!(distribution["data"]["total_sessions"].as_u64().is_some());
    let correlation = q(
        &archive,
        "sessions | measure correlation:message_count:word_count",
    );
    assert!(correlation["data"]["sample_count"].as_u64().unwrap() > 0);
    assert!(q(&archive, "sessions | measure compare")["data"]["profiles"].is_array());
    let similar = q(
        &archive,
        r#"sessions where id:"claude-code:7f3a2c10-0001-4000-8000-000000000001" | measure similar"#,
    );
    assert!(!similar["data"]["matches"].as_array().unwrap().is_empty());
    assert!(run_q(&archive, ask("sessions | measure similar")).is_err());
    assert_eq!(
        q(&archive, "sessions | measure outlook")["data"]["reason"],
        "plans_not_configured"
    );
}

#[test]
fn outlook_settings_and_method_reach_the_public_query() {
    let mut archive = morrow();
    archive.fx.setting("cost.plans",r#"{"sample":{"origin":"claude-code","cycle_anchor_day":1,"quota_unit":"credits","quota":1000,"price_usd":20,"overage_rate_usd_per_unit":null}}"#).unwrap();
    let answer = q(
        &archive,
        "sessions | measure outlook:sample:trailing-7d-mean",
    );
    assert_eq!(answer["data"]["projection_method"], "trailing-7d-mean");
    assert_eq!(answer["data"]["estimate"], true);
    assert_eq!(
        answer["data"]["quota_pressure"]["reason"],
        "quota_unit_unobserved"
    );
    assert!(run_q(&archive, ask("sessions | measure outlook:missing")).is_err());
}

#[test]
fn profiles_comparison_pace_and_similarity_page_with_the_query_cursor() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    for text in [
        "sessions | sort by id | measure profiles",
        "sessions | sort by id | measure compare",
        "sessions | sort by id | measure pace",
        r#"sessions where id:"claude-code:7f3a2c10-0001-4000-8000-000000000001" | measure similar"#,
    ] {
        let mut cursor = None::<String>;
        let mut ids = std::collections::BTreeSet::new();
        let count = loop {
            let mut request = ask(text);
            request.page_size = Some(1);
            request.cursor = cursor.as_deref();
            let page = run_q(&archive, request).unwrap();
            let rows = if text.ends_with("similar") {
                &page["data"]["matches"]
            } else if text.ends_with("pace") {
                &page["data"]["gaps"]
            } else {
                &page["data"]["profiles"]
            };
            assert!(rows.as_array().unwrap().len() <= 1);
            for row in rows.as_array().unwrap() {
                let id = if text.ends_with("pace") {
                    format!(
                        "{}:{}:{}",
                        row["session"], row["from_message"], row["to_message"]
                    )
                } else {
                    row["session"].as_str().unwrap().to_owned()
                };
                assert!(ids.insert(id));
            }
            match page["next"].as_str() {
                Some(next) => {
                    let command = crate::commands::command("query").unwrap();
                    assert!((command.print)(&page).contains(&format!("--cursor {next}")));
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
        facets["data"]["selected"]["total_sessions"],
        ordinary["count"]
    );
    assert_eq!(
        facets["data"]["global"]["total_sessions"],
        ordinary["count"]
    );
    let subagents = q(&archive, "sessions where kind:subagent | measure facets");
    assert!(
        subagents["data"]["global"]["total_sessions"]
            .as_u64()
            .unwrap()
            > ordinary["count"].as_u64().unwrap()
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
    for name in ["profiles", "compare"] {
        let text = format!("sessions | sort by id desc | measure {name}");
        let mut request = ask(&text);
        request.page_size = Some(2);
        request.fields = &fields;
        let measured = run_q(&archive, request).unwrap();
        let rows = &measured["data"][if name == "pace" {
            "sessions"
        } else {
            "profiles"
        }];
        let ids: Vec<_> = rows
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["session"].as_str().unwrap())
            .collect();
        assert_eq!(ids, expected, "{name}");
    }
}

#[test]
fn similarity_can_take_a_reference_selected_by_near_ranking() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let vectors = super::generated::vectors_for(&archive, |_| true);
    let near = |_: &str| Ok(std::sync::Arc::clone(&vectors));
    let text = "sessions where near:\"synthetic ranking\" | limit 1 | measure similar";
    let mut request = ask(text);
    request.embeddings = true;
    request.near = Some(&near);
    let answer = run_q(&archive, request).unwrap();
    assert!(!answer["data"]["matches"].as_array().unwrap().is_empty());
}

#[test]
fn pace_pages_global_gaps_and_keeps_summaries_for_the_full_selection() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let text = "sessions | measure pace";
    let mut whole = ask(text);
    whole.page_size = Some(1000);
    let expected = run_q(&archive, whole).unwrap();
    let mut first = ask(text);
    first.page_size = Some(1);
    let first = run_q(&archive, first).unwrap();
    assert_eq!(first["data"]["gaps"].as_array().unwrap().len(), 1);
    assert_eq!(first["data"]["gaps"][0], expected["data"]["gaps"][0]);
    assert_eq!(first["data"]["sessions"], expected["data"]["sessions"]);
    assert_eq!(first["count"], expected["data"]["total_gaps"]);
    let mut gaps = first["data"]["gaps"].as_array().unwrap().clone();
    let mut cursor = first["next"].as_str().map(str::to_owned);
    while let Some(next) = cursor {
        let mut request = ask(text);
        request.page_size = Some(2);
        request.cursor = Some(&next);
        let page = run_q(&archive, request).unwrap();
        assert_eq!(page["data"]["sessions"], expected["data"]["sessions"]);
        gaps.extend(page["data"]["gaps"].as_array().unwrap().iter().cloned());
        cursor = page["next"].as_str().map(str::to_owned);
    }
    assert_eq!(gaps, *expected["data"]["gaps"].as_array().unwrap());
}

#[test]
fn measures_over_near_ranked_sessions_keep_page_counts() {
    let mut archive = morrow();
    archive.fx.derive().unwrap();
    let vectors = super::generated::vectors_for(&archive, |_| true);
    let near = |_: &str| Ok(std::sync::Arc::clone(&vectors));
    for (name, field) in [("profiles", "profiles"), ("compare", "profiles"), ("pace", "gaps")] {
        let text = format!("sessions where near:\"synthetic ranking\" | measure {name}");
        let mut request = ask(&text);
        request.embeddings = true;
        request.near = Some(&near);
        request.page_size = Some(1);
        let answer = run_q(&archive, request).unwrap();
        assert!(answer["count"].as_u64().unwrap() > 1, "{name}");
        assert_eq!(answer["data"][field].as_array().unwrap().len(), 1, "{name}");
        let mut request = ask(&text);
        request.embeddings = true;
        request.near = Some(&near);
        request.page_size = Some(1);
        request.cursor = answer["next"].as_str();
        let next = run_q(&archive, request).unwrap();
        assert_eq!(next["count"], answer["count"], "{name}");
        assert_eq!(next["data"][field].as_array().unwrap().len(), 1, "{name}");
    }
}
