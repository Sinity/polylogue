use super::*;
fn user(id: &str, parent: Option<&str>) -> Value {
    let mut v = json!({"type":"user","uuid":id,"sessionId":SID,"promptSource":"typed","message":{"content":id}});
    if let Some(p) = parent {
        v["parentUuid"] = json!(p);
    }
    v
}
fn assistant(id: &str, response: &str, parent: &str, text: &str) -> Value {
    json!({"type":"assistant","uuid":id,"sessionId":SID,"parentUuid":parent,"requestId":"request","message":{"id":response,"model":"synthetic","content":[{"type":"text","text":text}]}})
}
fn marker(kind: &str, id: Value) -> Value {
    json!({"type":kind,"sessionId":SID,"leafUuid":id})
}
fn import(world: &World, records: Vec<Value>, messages: usize) -> (Value, Value) {
    let text = records
        .iter()
        .map(|v| v.to_string() + "\n")
        .collect::<String>();
    let path = world.input("active.jsonl", text.as_bytes());
    let receipt = world.import(&path, "01900000-0000-7000-8000-000000009101");
    let doc = world.read(
        &reference(),
        receipt["retained"]["id"].as_str().unwrap(),
        messages,
    );
    (receipt, doc)
}
fn page(world: &World, branches: bool) -> Value {
    let mut call = transcript_call(&reference());
    if branches {
        call["branches"] = json!(true);
    }
    model_call(world, &call, Some(65000))
}
fn text(page: &Value) -> Vec<String> {
    page["shown"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v["text"].as_str().map(str::to_owned))
        .collect()
}
fn fork() -> Vec<Value> {
    vec![
        user("root", None),
        assistant("old", "old-response", "root", "Old answer"),
        assistant("chosen", "chosen-response", "root", "Chosen answer"),
    ]
}
#[test]
fn authoritative_leaf_selects_record_path_without_adding_file_tail_and_survives_restart() {
    let mut world = World::new();
    let mut records = fork();
    records.push(marker("last-prompt", json!("chosen")));
    records.push(user("tail", Some("chosen")));
    let (_, doc) = import(&world, records, 4);
    let before = page(&world, false);
    assert_ne!(before["outcome"], "error", "{before}");
    assert_eq!(text(&before), vec!["root", "Chosen answer"]);
    let all = page(&world, true);
    assert_eq!(
        text(&all),
        vec!["root", "Old answer", "Chosen answer", "tail"]
    );
    assert_eq!(
        all["shown"][2]["version_hash"],
        doc["messages"][2]["version_hash"]
    );
    fs::remove_file(world.work.path().join("active.jsonl")).unwrap();
    world.restart();
    let after = page(&world, false);
    assert_eq!(before["shown"], after["shown"]);
}
#[test]
fn later_summary_changes_path_and_invalid_later_marker_never_falls_back() {
    for value in [
        json!("old"),
        Value::Null,
        json!(false),
        json!("missing"),
        json!(""),
    ] {
        let world = World::new();
        let mut records = fork();
        records.push(marker("last-prompt", json!("chosen")));
        records.push(marker("summary", value.clone()));
        import(&world, records, 3);
        let selected = page(&world, false);
        if value == "old" {
            assert_eq!(text(&selected), vec!["root", "Old answer"]);
        } else {
            assert_eq!(selected["outcome"], "error", "{value}: {selected}");
            assert_eq!(text(&page(&world, true)).len(), 3);
        }
    }
}
#[test]
fn non_message_ancestors_are_walked_and_folded_message_keeps_its_complete_body() {
    let world = World::new();
    let records = vec![
        json!({"type":"attachment","uuid":"anchor","sessionId":SID}),
        user("root", Some("anchor")),
        assistant("part1", "same-response", "root", "First block"),
        json!({"type":"system","subtype":"local_command","uuid":"event","parentUuid":"part1","sessionId":SID}),
        assistant("part2", "same-response", "event", "Other block"),
        marker("last-prompt", json!("part1")),
    ];
    let (_, doc) = import(&world, records, 2);
    let selected = page(&world, false);
    assert_ne!(selected["outcome"], "error", "{selected}");
    // The owning rule is any active record -> whole grouped Message. The
    // off-path second contribution retains the same selected body/version.
    assert_eq!(text(&selected), vec!["root", "First block", "Other block"]);
    for shown in selected["shown"].as_array().unwrap().iter().skip(1) {
        assert_eq!(shown["version_hash"], doc["messages"][1]["version_hash"]);
        assert_page_field_origins(shown, &doc["messages"][1]);
    }
}
fn call(id: &str) -> Value {
    json!({"type":"tool_use","id":id,"name":"Read","input":{"path":"synthetic.txt"}})
}
fn result(id: &str, parent: &str, call: &str) -> Value {
    json!({"type":"user","uuid":id,"parentUuid":parent,"sessionId":SID,"message":{"content":[{"type":"tool_result","tool_use_id":call,"content":id}]}})
}
#[test]
fn results_of_proved_calls_are_peers_but_unrelated_or_ambiguous_ids_do_not_grant_membership() {
    for mode in [
        "known",
        "unrelated",
        "duplicate",
        "reemitted",
        "duplicate-answer",
        "missing-id",
    ] {
        let world = World::new();
        let mut a = assistant("calls", "calls-response", "root", "");
        let mut calls = vec![call("one"), call("two")];
        if mode == "duplicate" {
            let mut conflict = call("one");
            conflict["input"] = json!({"path":"different.txt"});
            calls.push(conflict);
        }
        if mode == "reemitted" {
            calls.push(call("one"));
        }
        a["message"]["content"] = json!(calls);
        let records = vec![
            user("root", None),
            a,
            result(
                "first",
                "calls",
                if mode == "unrelated" {
                    "other"
                } else if mode == "missing-id" {
                    ""
                } else {
                    "one"
                },
            ),
            result(
                "second",
                "calls",
                if mode == "duplicate-answer" {
                    "one"
                } else {
                    "two"
                },
            ),
            marker("last-prompt", json!("second")),
        ];
        import(&world, records, 4);
        let selected = page(&world, false);
        if matches!(mode, "known" | "reemitted") {
            assert_ne!(selected["outcome"], "error", "{selected}");
            assert_eq!(text(&selected), vec!["root", "first", "second"]);
        } else {
            assert_eq!(selected["outcome"], "error", "{selected}");
        }
    }
}

#[test]
fn current_record_correction_is_used_and_real_record_cycles_refuse() {
    for cycle in [false, true] {
        let world = World::new();
        let mut records = fork();
        records.push(marker("last-prompt", json!("chosen")));
        records.push(assistant(
            "chosen",
            "chosen-response",
            "old",
            "Chosen answer",
        ));
        if cycle {
            records.push(assistant("old", "old-response", "chosen", "Old answer"));
        }
        import(&world, records, 3);
        let selected = page(&world, false);
        if cycle {
            assert_eq!(selected["outcome"], "error", "{selected}");
            assert_ne!(page(&world, true)["outcome"], "error");
        } else {
            assert_eq!(text(&selected), vec!["root", "Old answer", "Chosen answer"]);
        }
    }
}

#[test]
fn direct_call_record_authority_supports_identical_reemission_and_refuses_conflicting_witnesses() {
    for mode in [
        "identical",
        "conflict",
        "missing",
        "future",
        "malformed",
        "contradictory-parent",
    ] {
        let world = World::new();
        let mut original = assistant("original", "original-response", "root", "");
        original["message"]["content"] = json!([call("one")]);
        let mut current = assistant("current", "current-response", "original", "");
        let mut repeated = call("one");
        if mode == "conflict" {
            repeated["input"] = json!({"path":"different.txt"});
        }
        current["message"]["content"] = json!([repeated, call("two")]);
        let mut first = result("first", "current", "one");
        first["sourceToolAssistantUUID"] = match mode {
            "missing" => json!("absent"),
            "future" => json!("future"),
            "malformed" => Value::Null,
            "identical" => json!("current"),
            _ => json!("original"),
        };
        let mut records = vec![
            user("root", None),
            original,
            current,
            first,
            result("second", "current", "two"),
            marker("last-prompt", json!("second")),
        ];
        if mode == "future" {
            let mut later = assistant("future", "future-response", "root", "");
            later["message"]["content"] = json!([call("one")]);
            records.push(later);
        }
        import(&world, records, if mode == "future" { 6 } else { 5 });
        let selected = page(&world, false);
        if mode == "identical" {
            assert_ne!(selected["outcome"], "error", "{selected}");
            assert_eq!(text(&selected), vec!["root", "first", "second"]);
        } else {
            assert_eq!(selected["outcome"], "error", "{mode}: {selected}");
        }
    }
}

#[test]
fn pending_observed_summary_does_not_borrow_old_leaf_and_completion_reapplies_current_marker() {
    let mut world = World::new();
    world.kill();
    let mut records = fork();
    records.push(marker("last-prompt", json!("chosen")));
    let mut raw = records
        .iter()
        .map(|v| v.to_string() + "\n")
        .collect::<String>()
        .into_bytes();
    let main = world.input(&format!("{SID}.jsonl"), &raw);
    let config=world.input("active-config.json",json!({"claude_main":{"path":main,"machine":"synthetic-machine","host":"synthetic-host","acquisition_quantum_bytes":65536}}).to_string().as_bytes());
    world.child = Some(OwnedChild(
        world
            .command()
            .args(["daemon", "--config", config.to_str().unwrap()])
            .stdout(Stdio::null())
            .stderr(Stdio::from(File::create(&world.log).unwrap()))
            .spawn()
            .unwrap(),
    ));
    let until = Instant::now() + WAIT;
    let source = loop {
        let out = world.cli(&[&reference(), "read", "--format", "json"]);
        if matches!(out.status.code(), Some(0 | 5)) {
            let doc: Value = serde_json::from_slice(&out.stdout).unwrap();
            if doc["messages"].as_array().is_some_and(|m| m.len() == 3)
                && page(&world, false).get("shown").is_some()
            {
                let source = doc["messages"][0]["sources"][0]["input"]
                    .as_str()
                    .unwrap()
                    .to_owned();
                let p = world.raw_progress(&source);
                if p["active"] == false && p["source_rows"]["state"] != "pending" {
                    assert!(p["failure"].is_null(), "{p}");
                    break source;
                }
            }
        }
        assert!(Instant::now() < until, "initial leaf: {}", world.log());
        thread::sleep(Duration::from_millis(10));
    };
    let later = (marker("summary", Value::Null).to_string() + "\n").into_bytes();
    let split = later.len() / 2;
    raw.extend_from_slice(&later[..split]);
    fs::write(&main, &raw).unwrap();
    let until = Instant::now() + WAIT;
    loop {
        let p = world.raw_progress(&source);
        if p["parser"]["acquired_upto"] == raw.len() as u64
            && p["parser"]["status"] == "waiting_tail"
            && p["active"] == false
        {
            break;
        }
        assert!(Instant::now() < until, "{p}");
        thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(page(&world, false)["outcome"], "error");
    raw.extend_from_slice(&later[split..]);
    raw.extend_from_slice((marker("summary", json!("old")).to_string() + "\n").as_bytes());
    fs::write(&main, &raw).unwrap();
    let until = Instant::now() + WAIT;
    loop {
        let p = world.raw_progress(&source);
        if p["parser"]["acquired_upto"] == raw.len() as u64
            && p["active"] == false
            && p["source_rows"]["state"] != "pending"
        {
            let selected = page(&world, false);
            if selected.get("shown").is_some() {
                assert_eq!(text(&selected), vec!["root", "Old answer"]);
                break;
            }
        }
        assert!(Instant::now() < until, "{p}; {}", world.log());
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn direct_caller_and_tree_parent_may_name_different_fragments_of_same_response() {
    let world = World::new();
    let mut first = assistant("first-call", "same-response", "root", "");
    first["message"]["content"] = json!([call("one")]);
    let mut second = assistant("second-call", "same-response", "first-call", "");
    second["message"]["content"] = json!([call("two")]);
    let mut result1 = result("result1", "second-call", "one");
    result1["sourceToolAssistantUUID"] = json!("first-call");
    import(
        &world,
        vec![
            user("root", None),
            first,
            second,
            result1,
            result("result2", "second-call", "two"),
            marker("last-prompt", json!("result2")),
        ],
        4,
    );
    assert_eq!(
        text(&page(&world, false)),
        vec!["root", "result1", "result2"]
    );
}

#[test]
fn off_path_result_requires_established_caller_before_it_can_be_inactive() {
    for mode in ["absent", "null", "known-inactive", "inactive-mismatch"] {
        let world = World::new();
        let mut active = assistant("active", "active-response", "root", "");
        active["message"]["content"] = json!([call("one"), call("two")]);
        let mut inactive = assistant("inactive", "inactive-response", "root", "");
        inactive["message"]["content"] = json!([call("off")]);
        let mut extra = result(
            "extra",
            "inactive",
            if mode == "known-inactive" {
                "off"
            } else {
                "one"
            },
        );
        if mode == "absent" {
            extra.as_object_mut().unwrap().remove("parentUuid");
        } else if mode == "null" {
            extra["parentUuid"] = Value::Null;
        }
        let (_, doc) = import(
            &world,
            vec![
                user("root", None),
                active,
                inactive,
                extra,
                result("leaf", "active", "two"),
                marker("last-prompt", json!("leaf")),
            ],
            5,
        );
        // Root results with absent/null tree parent are real selected Messages,
        // so whole-row projection readiness cannot settle their caller scope.
        assert_eq!(doc["messages"].as_array().unwrap().len(), 5);
        assert_ne!(page(&world, true)["outcome"], "error", "{mode}");
        let selected = page(&world, false);
        if mode == "known-inactive" {
            assert_ne!(selected["outcome"], "error", "{selected}");
            assert_eq!(text(&selected), vec!["root", "leaf"]);
        } else {
            assert_eq!(selected["outcome"], "error", "{mode}: {selected}");
        }
    }
}
