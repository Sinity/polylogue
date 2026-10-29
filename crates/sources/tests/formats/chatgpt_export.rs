//! ChatGPT export knowledge from the Python version's
//! `test_parsers_chatgpt.py`, restated against this parser's model.

use crate::support::{blocks, events, message, messages, parse};
use serde_json::{Value, json};

const PATH: &str = "Downloads/chatgpt-export/conversations.json";

fn node(id: &str, role: &str, content: Value, meta: Value) -> Value {
    json!({"id": id, "children": [], "message": {"id": id, "author": {"role": role},
        "create_time": 1785578400.0, "content": content, "metadata": meta, "status": "finished_successfully"}})
}

fn text(parts: &[&str]) -> Value {
    json!({"content_type": "text", "parts": parts})
}

/// Links the nodes into one chain, in order, under a conversation.
fn conversation(nodes: Vec<Value>, extra: Value) -> Value {
    let mut mapping = serde_json::Map::new();
    let mut prev: Option<String> = None;
    for mut n in nodes {
        let id = n["id"].as_str().unwrap().to_owned();
        n["parent"] = json!(prev);
        if let Some(p) = &prev {
            mapping[p]["children"]
                .as_array_mut()
                .unwrap()
                .push(json!(id));
        }
        mapping.insert(id.clone(), n);
        prev = Some(id);
    }
    let mut c = json!({"id": "c1", "title": "t", "create_time": 1785578400.0, "mapping": mapping});
    c.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    c
}

fn run(conv: Value) -> Value {
    parse("chatgpt", PATH, json!([conv]).to_string().as_bytes())
}

fn with(mut n: Value, key: &str, value: Value) -> Value {
    n["message"][key] = value;
    n
}

#[test]
fn media_pointers_become_attachments_with_their_file_id() {
    let p = run(conversation(
        vec![node(
            "m1",
            "assistant",
            json!({"content_type": "multimodal_text", "parts": [
                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-IMG1", "size_bytes": 10},
                {"content_type": "audio_asset_pointer", "asset_pointer": "file-service://file-AUDIO7",
                 "mime_type": "audio/wav", "size_bytes": 4096},
            ]}),
            json!({}),
        )],
        json!({}),
    ));
    let b = &message(&p, ":m1")["blocks"];
    assert_eq!(b[0]["type"], "image");
    assert_eq!(b[0]["source_id"], "file-IMG1");
    assert_eq!(b[1]["type"], "file");
    assert_eq!(b[1]["source_id"], "file-AUDIO7");
    assert_eq!(b[1]["media_type"], "audio/wav");
    assert_eq!(b[1]["size"], 4096);
    assert_eq!(b[1]["direction"], "output");
}

#[test]
fn a_media_part_without_a_pointer_invents_no_attachment() {
    let p = run(conversation(
        vec![node(
            "m1",
            "assistant",
            json!({"content_type": "multimodal_text",
        "parts": [{"content_type": "image_asset_pointer", "width": 1}]}),
            json!({}),
        )],
        json!({}),
    ));
    let b = &message(&p, ":m1")["blocks"][0];
    assert_eq!(b["type"], "other");
    assert_eq!(b["type_name"], "image_asset_pointer");
}

#[test]
fn a_tool_result_takes_its_outcome_from_the_message_status() {
    let outcome = |status: Option<&str>| {
        let mut out = node(
            "out",
            "tool",
            json!({"content_type": "execution_output", "text": "1"}),
            json!({}),
        );
        match status {
            Some(s) => out["message"]["status"] = json!(s),
            None => {
                out["message"].as_object_mut().unwrap().remove("status");
            }
        }
        let call = with(
            node(
                "call",
                "assistant",
                json!({"content_type": "code", "language": "python", "text": "print(1)"}),
                json!({}),
            ),
            "recipient",
            json!("python"),
        );
        let p = run(conversation(vec![call, out], json!({})));
        blocks(&p)
            .into_iter()
            .find(|b| b["type"] == "tool_result")
            .unwrap()["outcome"]
            .clone()
    };
    assert_eq!(
        outcome(Some("finished_successfully")),
        json!({"kind": "ok"})
    );
    assert_eq!(
        outcome(Some("finished_partial_completion"))["kind"],
        "error"
    );
    assert_ne!(
        outcome(Some("in_progress"))["kind"],
        "ok",
        "a running call is not a success"
    );
    assert_eq!(
        outcome(None),
        json!({"kind": "unknown", "reason": "status_field_unreadable"}),
        "no status is no verdict, never a guessed one"
    );
}

#[test]
fn a_failed_code_run_is_an_error_whatever_the_message_status() {
    let call = with(
        node(
            "call",
            "assistant",
            json!({"content_type": "code", "text": "1/0"}),
            json!({}),
        ),
        "recipient",
        json!("python"),
    );
    let out = node(
        "out",
        "tool",
        json!({"content_type": "execution_output", "text": "ZeroDivisionError"}),
        json!({"aggregate_result": {"status": "failed_with_in_kernel_exception", "code": "1/0"}}),
    );
    let p = run(conversation(vec![call, out], json!({})));
    let result = blocks(&p)
        .into_iter()
        .find(|b| b["type"] == "tool_result")
        .unwrap();
    assert_eq!(result["outcome"]["kind"], "error");
    assert_eq!(
        result["call_id"], "call",
        "the result pairs with the call it answers"
    );
}

#[test]
fn a_system_error_is_an_error_result() {
    let call = with(
        node(
            "call",
            "assistant",
            json!({"content_type": "code", "text": "x"}),
            json!({}),
        ),
        "recipient",
        json!("python"),
    );
    let err = node(
        "err",
        "tool",
        json!({"content_type": "system_error", "name": "tool_error", "text": "boom"}),
        json!({}),
    );
    let p = run(conversation(vec![call, err], json!({})));
    let result = blocks(&p)
        .into_iter()
        .find(|b| b["type"] == "tool_result")
        .unwrap();
    assert_eq!(result["outcome"]["kind"], "error");
    assert_eq!(result["output"], "boom");
}

#[test]
fn a_recipient_addressed_json_payload_is_a_tool_call_with_parsed_input() {
    let call = with(
        node(
            "a",
            "assistant",
            text(&[r#"{"search_query":[{"q":"prices"}],"response_length":"medium"}"#]),
            json!({}),
        ),
        "recipient",
        json!("web"),
    );
    let p = run(conversation(vec![call], json!({})));
    let b = &message(&p, ":a")["blocks"];
    assert_eq!(b.as_array().unwrap().len(), 1);
    assert_eq!(b[0]["type"], "tool_call");
    assert_eq!(b[0]["tool"], "web");
    assert_eq!(
        b[0]["input"],
        json!({"search_query": [{"q": "prices"}], "response_length": "medium"})
    );
}

#[test]
fn code_sent_to_the_code_tool_is_its_command() {
    let call = with(
        node(
            "c",
            "assistant",
            json!({"content_type": "code", "language": "python", "text": "print(1)"}),
            json!({}),
        ),
        "recipient",
        json!("python"),
    );
    let p = run(conversation(vec![call], json!({})));
    let b = &message(&p, ":c")["blocks"][0];
    assert_eq!(b["tool"], "python");
    assert_eq!(b["command"], "print(1)");
}

#[test]
fn citation_marks_and_structured_references_are_kept_as_written() {
    let marked = "Per the brief. \u{e200}filecite\u{e202}turn0file0\u{e201} More text \u{e200}cite\u{e202}turn1search2\u{e202}L10-L20\u{e201} end.";
    let p = run(conversation(
        vec![node(
            "a",
            "assistant",
            text(&[marked]),
            json!({"citations": [{"start_ix": 15, "end_ix": 43, "metadata": {"title": "brief.md"}}]}),
        )],
        json!({}),
    ));
    let t = message(&p, ":a")["blocks"][0]["text"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(t, marked);
    assert_eq!(
        message(&p, ":a")["blocks"][1],
        json!({"type":"other","type_name":"citations","data":[{"start_ix":15,"end_ix":43,"metadata":{"title":"brief.md"}}]})
    );
}

#[test]
fn an_unterminated_citation_opener_is_kept_as_written() {
    let p = run(conversation(
        vec![node(
            "a",
            "assistant",
            text(&["A \u{e200}cite\u{e202}x\u{e201} B \u{e200}open"]),
            json!({}),
        )],
        json!({}),
    ));
    assert_eq!(
        message(&p, ":a")["blocks"][0]["text"],
        "A \u{e200}cite\u{e202}x\u{e201} B \u{e200}open"
    );
}

#[test]
fn the_model_falls_back_through_the_default_slugs() {
    let p = run(conversation(
        vec![
            node("asked", "user", text(&["Q"]), json!({})),
            node(
                "exact",
                "assistant",
                text(&["A"]),
                json!({"model_slug": "o3"}),
            ),
            node(
                "defaulted",
                "assistant",
                text(&["A"]),
                json!({"default_model_slug": "gpt-4o"}),
            ),
            node("bare", "assistant", text(&["A"]), json!({})),
        ],
        json!({"default_model_slug": "gpt-5-pro"}),
    ));
    assert_eq!(message(&p, ":exact")["model"], "o3");
    assert_eq!(message(&p, ":defaulted")["model"], "gpt-4o");
    assert_eq!(message(&p, ":bare")["model"], "gpt-5-pro");
    assert!(message(&p, ":asked").get("model").is_none());
}

#[test]
fn finish_details_are_kept_as_written() {
    let p = run(conversation(
        vec![node(
            "a",
            "assistant",
            text(&["A"]),
            json!({"finish_details": {"type": "max_tokens"}}),
        )],
        json!({}),
    ));
    assert_eq!(message(&p, ":a")["stop_reason"], "max_tokens");
}

#[test]
fn every_branch_is_kept_and_the_active_leaf_is_marked() {
    let mut conv = conversation(
        vec![
            node("u", "user", text(&["Q"]), json!({})),
            node("a1", "assistant", text(&["first"]), json!({})),
        ],
        json!({"current_node": "a1"}),
    );
    let mut second = node("a2", "assistant", text(&["second"]), json!({}));
    second["parent"] = json!("u");
    conv["mapping"]["u"]["children"]
        .as_array_mut()
        .unwrap()
        .push(json!("a2"));
    conv["mapping"]["a2"] = second;
    let p = run(conv);
    assert_eq!(message(&p, ":a1")["parent"], "chatgpt:u");
    assert_eq!(message(&p, ":a2")["parent"], "chatgpt:u");
    assert_eq!(events(&p, "active_leaf").len(), 1);
}

#[test]
fn without_a_current_node_every_node_is_kept_and_no_leaf_is_claimed() {
    let p = run(conversation(
        vec![
            node("u", "user", text(&["Q"]), json!({})),
            node("a", "assistant", text(&["A"]), json!({})),
        ],
        json!({}),
    ));
    assert_eq!(messages(&p).len(), 2);
    assert!(events(&p, "active_leaf").is_empty());
}

#[test]
fn a_child_keeps_its_parent_link_through_an_empty_message() {
    let p = run(conversation(
        vec![
            node("u", "user", text(&["Q"]), json!({})),
            node("e", "assistant", text(&[""]), json!({})),
            node("f", "user", text(&["after empty"]), json!({})),
        ],
        json!({}),
    ));
    assert_eq!(message(&p, ":f")["parent"], "chatgpt:e");
}

#[test]
fn an_idless_message_takes_its_node_id() {
    let mut conv = conversation(
        vec![node("n1", "user", text(&["no id"]), json!({}))],
        json!({}),
    );
    conv["mapping"]["n1"]["message"]
        .as_object_mut()
        .unwrap()
        .remove("id");
    assert_eq!(messages(&run(conv))[0]["id"], "chatgpt:n1");
}

#[test]
fn reasoning_reaches_thinking_blocks() {
    let p = run(conversation(
        vec![
            node(
                "t",
                "assistant",
                json!({"content_type": "thoughts", "thoughts": [{"summary": "S", "content": "C"}]}),
                json!({}),
            ),
            node(
                "r",
                "assistant",
                json!({"content_type": "reasoning_recap", "content": "Thought for 5s"}),
                json!({}),
            ),
        ],
        json!({}),
    ));
    assert_eq!(
        message(&p, ":t")["blocks"][0],
        json!({"type": "thinking", "text": "S\n\nC"})
    );
    assert_eq!(
        message(&p, ":r")["blocks"][0],
        json!({"type": "thinking", "text": "Thought for 5s"})
    );
}

#[test]
fn custom_instructions_keep_human_authorship() {
    let p = run(conversation(
        vec![node(
            "uec",
            "user",
            json!({"content_type": "user_editable_context", "user_profile": "P", "user_instructions": "I"}),
            json!({}),
        )],
        json!({}),
    ));
    let m = message(&p, ":uec");
    assert_eq!(m["role"], "user");
    assert_eq!(m["blocks"][0]["text"], "P\n\nI");
}

#[test]
fn a_temporary_chat_is_recorded_in_the_conversation_settings() {
    let p = run(conversation(
        vec![node("m1", "user", text(&["hi"]), json!({}))],
        json!({"is_temporary": true}),
    ));
    assert_eq!(
        events(&p, "conversation_settings")[0]["data"]["is_temporary"],
        true
    );
}

#[test]
fn overflowing_optional_duration_stays_unknown_without_losing_the_answer() {
    for seconds in [1e308, 1e20, u64::MAX as f64 / 1000.0] {
        let parsed = run(conversation(
            vec![node(
                "answer",
                "assistant",
                text(&["Neutral authored answer"]),
                json!({"finished_duration_sec": seconds}),
            )],
            json!({}),
        ));
        let answer = message(&parsed, ":answer");
        assert!(
            answer.get("duration_ms").is_none(),
            "overflowing seconds: {seconds}"
        );
        assert_eq!(answer["blocks"][0]["text"], "Neutral authored answer");
        assert_eq!(messages(&parsed).len(), 1);
    }
}

#[test]
fn optional_duration_keeps_seconds_units_zero_and_unknown_distinct() {
    for (supplied, expected) in [
        (None, None),
        (Some(json!(null)), None),
        (Some(json!(true)), None),
        (Some(json!("7")), None),
        (Some(json!({})), None),
        (Some(json!([])), None),
        (Some(json!(-0.0001)), None),
        (Some(json!(0)), Some(0)),
        (Some(json!(0.0)), Some(0)),
        (Some(json!(0.00049)), Some(0)),
        (Some(json!(0.00051)), Some(1)),
        (Some(json!(0.5)), Some(500)),
        (Some(json!(7.0)), Some(7000)),
        (Some(json!(1.2346)), Some(1235)),
    ] {
        let mut metadata = json!({});
        if let Some(value) = supplied {
            metadata["finished_duration_sec"] = value;
        }
        let parsed = run(conversation(
            vec![node(
                "answer",
                "assistant",
                text(&["Neutral authored answer"]),
                metadata,
            )],
            json!({}),
        ));
        let answer = message(&parsed, ":answer");
        assert_eq!(answer.get("duration_ms").and_then(Value::as_u64), expected);
        assert_eq!(answer["blocks"][0]["text"], "Neutral authored answer");
        assert_eq!(messages(&parsed).len(), 1);
    }
}
