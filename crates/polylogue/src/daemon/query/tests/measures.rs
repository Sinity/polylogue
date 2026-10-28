use super::*;

#[test]
fn catalog_measures_use_selected_sessions_and_typed_fields() {
    let mut archive=morrow();archive.fx.derive().unwrap();
    for removed in ["profiles","pace","similar","correlation","distribution"] {
        assert!(run_q(&archive,ask(&format!("sessions | measure {removed}"))).is_err());
    }
    let all=q(&archive,"sessions | measure facets:deep");
    let selected=q(&archive,"sessions where origin:claude-code | measure facets:deep");
    assert!(selected["data"]["summary"]["selected_sessions"].as_u64().unwrap()<all["data"]["summary"]["selected_sessions"].as_u64().unwrap());
    assert!(selected["data"]["values"].as_array().unwrap().iter().all(|v|v["idf"].is_null()||v["idf"].is_number()));
    let time=q(&archive,"sessions where origin:claude-code | measure time");
    assert!(time["data"]["sessions"].as_array().unwrap().iter().all(|s|s["session"].as_str().unwrap().starts_with("claude-code:")));
    assert!(q(&archive,"sessions | measure compare:origin,words")["data"]["sessions"].is_array());
    assert!(run_q(&archive,ask("sessions | measure time:unsupported")).is_err());
    assert!(run_q(&archive,ask("sessions | measure compare:unsupported")).is_err());
}

#[test]
fn outlook_setting_method_and_integer_money_reach_the_public_query() {
    let mut archive=morrow();archive.fx.setting("plan.example",r#"{"name":"max-20x","origin":"claude-code","cycle_anchor_day":1,"price_nusd":200000000000,"budget":{"population":"responses","unit":"input_tokens","amount":1000000,"overage_rate_nusd_per_unit":null}}"#).unwrap();
    let answer=q(&archive,"sessions | measure outlook:example:trailing-7d-mean");
    assert_eq!(answer["data"]["projection_method"],"trailing-7d-mean");assert_eq!(answer["data"]["trailing_observed_days"],7);
    assert_eq!(answer["data"]["estimate"],true);assert_eq!(answer["data"]["segments"][0]["plan"]["price_nusd"],200_000_000_000_i64);
    assert_eq!(q(&archive,"sessions | measure outlook:missing")["data"]["reason"],"plan_not_configured");
    assert_eq!(q(&archive,"sessions | measure outlook")["data"]["reason"],"provider_required");
}

#[test]
fn time_comparison_and_facets_page_with_the_query_cursor() {
    let mut archive=morrow();archive.fx.derive().unwrap();
    for (text,field) in [("sessions | sort by id | measure time","sessions"),("sessions | sort by id | measure compare","sessions"),("sessions | measure facets:deep","values")] {
        let mut cursor=None::<String>;let mut ids=std::collections::BTreeSet::new();
        let count=loop {
            let mut request=ask(text);request.page_size=Some(1);request.cursor=cursor.as_deref();let page=run_q(&archive,request).unwrap();
            let rows=page["data"][field].as_array().unwrap();assert!(rows.len()<=1);
            for row in rows {let id=if field=="values"{format!("{}:{}",row["family"],row["value"])}else{row["session"].as_str().unwrap().to_owned()};assert!(ids.insert(id));}
            match page["next"].as_str(){Some(next)=>{let command=crate::commands::command("query").unwrap();assert!((command.print)(&page).contains(&format!("--cursor {next}")));cursor=Some(next.into());},None=>break page["count"].as_u64().unwrap()}
        };
        assert_eq!(ids.len() as u64,count);assert!(count>1);
    }
}

#[test]
fn facets_use_the_selection_units_default_population() {
    let mut archive=morrow();archive.fx.derive().unwrap();let ordinary=q(&archive,"sessions");let facets=q(&archive,"sessions | measure facets");
    assert_eq!(facets["data"]["summary"]["selected_sessions"],ordinary["count"]);assert_eq!(facets["data"]["summary"]["global_sessions"],ordinary["count"]);
    let subagents=q(&archive,"sessions where kind:subagent | measure facets");assert!(subagents["data"]["summary"]["global_sessions"].as_u64().unwrap()>ordinary["count"].as_u64().unwrap());
    let empty=q(&archive,"sessions where origin:absent | measure facets");assert_eq!(empty["data"]["summary"]["outcome"],"empty");assert!(empty["data"]["values"].as_array().unwrap().iter().all(|v|v["selected_count"]==0));
}

#[test]
fn paged_measures_preserve_query_order_when_display_fields_omit_the_id() {
    let mut archive=morrow();archive.fx.derive().unwrap();let mut request=ask("sessions | sort by id desc");request.page_size=Some(2);let ordinary=run_q(&archive,request).unwrap();
    let expected:Vec<_>=ordinary["rows"].as_array().unwrap().iter().map(|r|r["id"].as_str().unwrap()).collect();
    let fields=vec!["title".into()];
    for name in ["time","compare"] {
        let text=format!("sessions | sort by id desc | measure {name}");let mut request=ask(&text);request.page_size=Some(2);request.fields=&fields;let measured=run_q(&archive,request).unwrap();
        let ids:Vec<_>=measured["data"]["sessions"].as_array().unwrap().iter().map(|r|r["session"].as_str().unwrap()).collect();assert_eq!(ids,expected,"{name}");
    }
}

#[test]
fn comparisons_keep_whole_selection_differences_on_every_page() {
    let mut archive=morrow();archive.fx.derive().unwrap();let text="sessions | sort by id | measure compare:origin,words,wall_span";
    let mut request=ask(text);request.page_size=Some(1);let first=run_q(&archive,request).unwrap();
    assert!(first["data"]["varies"].as_array().unwrap().contains(&json!("origin")));
    let mut request=ask(text);request.page_size=Some(1);request.cursor=first["next"].as_str();let next=run_q(&archive,request).unwrap();
    assert_eq!(first["data"]["varies"],next["data"]["varies"]);assert_eq!(first["data"]["unknown"],next["data"]["unknown"]);
}

#[test]
fn measures_over_near_ranked_sessions_keep_page_counts() {
    let mut archive=morrow();archive.fx.derive().unwrap();let vectors=super::generated::vectors_for(&archive,|_|true);let near=|_:&str|Ok(std::sync::Arc::clone(&vectors));
    for (name,field) in [("time","sessions"),("compare","sessions"),("facets","values")] {
        let text=format!("sessions where near:\"synthetic ranking\" | measure {name}");let mut request=ask(&text);request.embeddings=true;request.near=Some(&near);request.page_size=Some(1);let answer=run_q(&archive,request).unwrap();
        assert!(answer["count"].as_u64().unwrap()>1);assert_eq!(answer["data"][field].as_array().unwrap().len(),1);
        let mut request=ask(&text);request.embeddings=true;request.near=Some(&near);request.page_size=Some(1);request.cursor=answer["next"].as_str();let next=run_q(&archive,request).unwrap();
        assert_eq!(next["count"],answer["count"]);assert_eq!(next["data"][field].as_array().unwrap().len(),1);
    }
}

#[test]
fn stored_counts_and_time_match_morrow_after_the_real_loop() {
    let archive=polylogue_testkit::Morrow::build(polylogue_testkit::Set::Main);let pool=archive.reads();let reader=pool.reader().unwrap();
    let query=r#"sessions where id:"claude-code:7f3a2c10-0002-4000-8000-000000000002" | measure time"#;
    let answer=run(&reader,ask(query)).unwrap();let time=&answer["data"]["sessions"][0];
    assert_eq!(time["span"]["wall_ms"],4_500_000);assert_eq!(time["span"]["clustered_ms"],3_720_000);
    assert_eq!(time["span"]["tool_union_ms"],1_704_000);assert_eq!(time["span"]["tool_sum_ms"],3_434_000);
    let lanes=time["lanes"].as_array().unwrap();let lane=|name:&str|lanes.iter().filter(|l|l["lane"]==name).map(|l|l["ms"].as_i64().unwrap()).sum::<i64>();
    assert_eq!((lane("tool"),lane("model"),lane("human"),lane("unattributed")),(1_704_000,2_348_000,448_000,0));
    let fields=vec!["messages".into(),"human_prompts".into(),"tool_calls".into(),"errors".into(),"wall_span".into(),"clustered_time".into()];
    let mut request=ask(r#"sessions where id:"claude-code:7f3a2c10-0002-4000-8000-000000000002""#);request.fields=&fields;let answer=run(&reader,request).unwrap();
    assert_eq!(answer["rows"][0]["messages"],18);assert_eq!(answer["rows"][0]["human_prompts"],3);assert_eq!(answer["rows"][0]["tool_calls"],13);assert_eq!(answer["rows"][0]["errors"],1);
}

#[test]
fn aggregate_counts_report_pending_and_compute_current_rows() {
    let archive=morrow();let answer=q(&archive,"sessions | agg sum:messages");
    assert!(answer["rows"][0]["sum_messages"].as_i64().unwrap()>0);assert!(answer["rows"][0]["pending_messages"].as_i64().unwrap()>0);
}

#[test]
fn clustered_fields_print_the_parameter_in_query_output() {
    let mut archive=morrow();archive.fx.derive().unwrap();
    let result=q(&archive,"sessions | agg p50:clustered_time");
    assert!(result["notes"].as_array().unwrap().iter().any(|n|n.as_str().is_some_and(|n|n.contains("5-minute gap"))));
    assert!(crate::cli::print::query(&result).contains("5-minute gap"));
}
