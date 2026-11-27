use super::*;

use serde_json::json;

/// The rule as a file found at a location meets it: its first bytes alone.
fn recognize(path: &std::path::Path, head: &[u8]) -> bool {
    super::recognize(path, &crate::document::Head::new(head))
}

fn node(id: &str, parent: Option<&str>, message: Json) -> (String, Json) {
    (
        id.to_owned(),
        json!({"id": id, "parent": parent, "children": [], "message": message}),
    )
}

fn msg(id: &str, role: &str, time: f64, content: Json) -> Json {
    json!({"id": id, "author": {"role": role, "name": null}, "create_time": time,
           "content": content, "status": "finished_successfully", "recipient": "all",
           "metadata": {}})
}

fn text(parts: &[&str]) -> Json {
    json!({"content_type": "text", "parts": parts})
}

fn export(convs: &[Json]) -> crate::parse::Step {
    let bytes = serde_json::to_vec(&Json::Array(convs.to_vec())).unwrap();
    crate::export::fixture::parse(&mut Parse::Export(Box::new(Export)), &bytes).unwrap()
}

fn conv(id: &str, current: &str, nodes: Vec<(String, Json)>) -> Json {
    let mapping: serde_json::Map<String, Json> = nodes.into_iter().collect();
    json!({"id": id, "conversation_id": id, "title": "T", "create_time": 1700000000.0,
           "update_time": 1700000009.0, "current_node": current, "mapping": mapping})
}

fn ids(step: &crate::parse::Step) -> Vec<String> {
    step.output
        .messages
        .iter()
        .map(|r| r.message.id.native().to_string())
        .collect()
}

#[test]
fn every_branch_is_kept_and_the_active_leaf_is_marked() {
    let c = conv(
        "c1",
        "n-edit-answer",
        vec![
            node("root", None, Json::Null),
            node("n-q", Some("root"), msg("q", "user", 1.0, text(&["first"]))),
            node(
                "n-a",
                Some("n-q"),
                msg("a", "assistant", 2.0, text(&["one"])),
            ),
            node(
                "n-edit",
                Some("root"),
                msg("q2", "user", 3.0, text(&["edited"])),
            ),
            node(
                "n-edit-answer",
                Some("n-edit"),
                msg("a2", "assistant", 4.0, text(&["two"])),
            ),
        ],
    );
    let step = export(&[c]);
    assert_eq!(ids(&step), ["q", "a", "q2", "a2"]);
    let edited = &step.output.messages[2].message;
    assert_eq!(edited.parent, None, "the root holds no message");
    let answer = &step.output.messages[3].message;
    assert_eq!(answer.parent.as_ref().unwrap().native().as_str(), "q2");
    let leaf = step
        .output
        .events
        .iter()
        .find(|e| e.kind == "active_leaf")
        .unwrap();
    assert_eq!(leaf.after.as_ref().unwrap().native().as_str(), "a2");
    assert_eq!(step.counts.records, 5, "one conversation, four messages");
    assert_eq!(step.counts.used["text"], 4);
}

#[test]
fn every_result_of_a_tool_call_points_at_the_call() {
    let mut call = msg(
        "call",
        "assistant",
        2.0,
        json!({"content_type": "code", "language": "python", "text": "print(1)"}),
    );
    call["recipient"] = json!("python");
    let mut r1 = msg(
        "r1",
        "tool",
        3.0,
        json!({"content_type": "execution_output", "text": "1"}),
    );
    r1["author"]["name"] = json!("python");
    r1["metadata"] = json!({"aggregate_result": {"status": "failed_with_in_kernel_exception"}});
    let mut r2 = msg("r2", "tool", 4.0, text(&["more"]));
    r2["author"]["name"] = json!("python");
    let c = conv(
        "c2",
        "n-r2",
        vec![
            node("n-q", None, msg("q", "user", 1.0, text(&["run it"]))),
            node("n-call", Some("n-q"), call),
            node("n-r1", Some("n-call"), r1),
            node("n-r2", Some("n-r1"), r2),
        ],
    );
    let step = export(&[c]);
    let blocks: Vec<&Block> = step
        .output
        .messages
        .iter()
        .flat_map(|r| &r.message.blocks)
        .collect();
    match blocks[1] {
        Block::ToolCall {
            run_by_script: false,
            call_id,
            tool,
            command,
            ..
        } => {
            assert_eq!(call_id.as_deref(), Some("call"));
            assert_eq!(tool.as_deref(), Some("python"));
            assert_eq!(command.as_deref(), Some("print(1)"));
        }
        other => panic!("{other:?}"),
    }
    for (block, ok) in [(blocks[2], false), (blocks[3], true)] {
        match block {
            Block::ToolResult {
                call_id, outcome, ..
            } => {
                assert_eq!(call_id.as_deref(), Some("call"));
                assert_eq!(*outcome == ToolOutcome::Ok, ok);
            }
            other => panic!("{other:?}"),
        }
    }
    let r = &step.output.messages[2].message;
    assert_eq!(r.role, Role::User);
    assert_eq!(r.prompt_kind, Some(PromptKind::Harness));
}

#[test]
fn messages_sort_by_time_and_untimed_ones_keep_their_place() {
    let mut untimed = msg("b", "assistant", 0.0, text(&["b"]));
    untimed["create_time"] = Json::Null;
    let c = conv(
        "c3",
        "n-c",
        vec![
            node("n-c", Some("n-b"), msg("c", "user", 5.0, text(&["c"]))),
            node("n-a", None, msg("a", "user", 1.0, text(&["a"]))),
            node("n-b", Some("n-a"), untimed),
        ],
    );
    assert_eq!(ids(&export(&[c])), ["a", "b", "c"]);
}

#[test]
fn messages_with_rfc3339_times_sort_by_those_times() {
    let mut late = msg("late", "user", 0.0, text(&["late"]));
    late["create_time"] = Json::String("2024-03-01T10:00:05Z".into());
    let mut early = msg("early", "user", 0.0, text(&["early"]));
    early["create_time"] = Json::String("2024-03-01T10:00:01Z".into());
    let c = conv(
        "c4",
        "n-late",
        vec![
            node("n-late", Some("n-early"), late),
            node("n-early", None, early),
        ],
    );
    assert_eq!(ids(&export(&[c])), ["early", "late"]);
}

#[test]
fn empty_objects_and_broken_items_are_accounted_for() {
    let mut no_role = msg("x", "user", 1.0, text(&["x"]));
    no_role["author"] = json!({});
    let c = conv("c4", "n-x", vec![node("n-x", None, no_role)]);
    let step = export(&[json!({}), json!({"id": "c5"}), c]);
    assert_eq!(step.counts.records, 4);
    assert_eq!(
        step.counts.skipped["conversation: empty object in the export"],
        1
    );
    assert_eq!(step.counts.unreadable["conversation: no mapping"], 1);
    assert_eq!(step.counts.unreadable["text: no author role"], 1);
    assert_eq!(step.counts.used["conversation"], 1);
}

#[test]
fn images_and_uploads_keep_their_file_ids() {
    let mut user = msg(
        "u",
        "user",
        1.0,
        json!({"content_type": "multimodal_text", "parts": [
            {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-AbC",
             "size_bytes": 1234, "width": 10, "height": 10},
            "what is this?"
        ]}),
    );
    user["metadata"] = json!({"attachments": [
        {"id": "file-Doc", "name": "notes.pdf", "size": 99, "mime_type": "application/pdf"}
    ]});
    let c = conv("c6", "n-u", vec![node("n-u", None, user)]);
    let step = export(&[c]);
    let blocks = &step.output.messages[0].message.blocks;
    match &blocks[0] {
        Block::Image(a) => {
            assert_eq!(a.source_id.as_deref(), Some("file-AbC"));
            assert_eq!(a.size, Some(1234));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        blocks[1],
        Block::Text {
            text: "what is this?".into()
        }
    );
    match &blocks[2] {
        Block::File(a) => {
            assert_eq!(a.name.as_deref(), Some("notes.pdf"));
            assert_eq!(a.source_id.as_deref(), Some("file-Doc"));
            assert_eq!(a.size, Some(99));
            assert_eq!(a.media_type.as_deref(), Some("application/pdf"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_chat_branched_into_a_new_conversation_keeps_only_its_tail() {
    let shared = || {
        vec![
            node("n-q", None, msg("q", "user", 1.0, text(&["question"]))),
            node(
                "n-a",
                Some("n-q"),
                msg("a", "assistant", 2.0, text(&["answer"])),
            ),
        ]
    };
    let mut original = conv("orig", "n-a", shared());
    original["create_time"] = json!(1.0);
    let mut nodes = shared();
    nodes.push(node(
        "n-b",
        Some("n-a"),
        msg("b", "user", 3.0, text(&["branch"])),
    ));
    let mut branch = conv("fork", "n-b", nodes);
    branch["create_time"] = json!(2.0);
    // The export lists newest first.
    let step = export(&[branch, original]);
    let by_session: Vec<(String, String)> = step
        .output
        .messages
        .iter()
        .map(|r| {
            (
                r.message.session.native().to_string(),
                r.message.id.native().to_string(),
            )
        })
        .collect();
    assert_eq!(
        by_session,
        [
            ("orig".into(), "q".into()),
            ("orig".into(), "a".into()),
            ("fork".into(), "b".into())
        ]
    );
    let link = &step.output.links[0];
    assert_eq!(link.kind, LinkKind::Fork);
    assert_eq!(link.parent.native().as_str(), "orig");
    assert_eq!(link.child.native().as_str(), "fork");
    assert_eq!(link.branch_point.as_ref().unwrap().native().as_str(), "a");
    let tail = &step.output.messages[2].message;
    assert_eq!(tail.parent.as_ref().unwrap().native().as_str(), "a");
    let [copy] = &step.output.prefix_copies[..] else {
        panic!("one copied prefix")
    };
    assert!(copy.explicit_members);
    let members: Vec<_> = step
        .output
        .prefix_members
        .iter()
        .map(|member| {
            (
                member.parent.to_string(),
                member.child.to_string(),
                member.message.to_string(),
                member.source.is_none(),
                member.position,
            )
        })
        .collect();
    assert_eq!(
        members,
        [
            (
                "chatgpt:orig".into(),
                "chatgpt:fork".into(),
                "chatgpt:q".into(),
                true,
                [1, 0]
            ),
            (
                "chatgpt:orig".into(),
                "chatgpt:fork".into(),
                "chatgpt:a".into(),
                true,
                [1, 1]
            ),
        ]
    );
}

#[test]
fn recognizes_the_export_and_its_shards_only() {
    assert!(recognize(Path::new("/x/conversations.json"), b" [{"));
    assert!(recognize(
        Path::new("/x.zip!/conversations-012.json"),
        b"[]"
    ));
    assert!(!recognize(Path::new("/x/conversations.json"), b"{"));
    assert!(!recognize(Path::new("/x/shared_conversations.json"), b"["));
    assert!(recognize(
        Path::new("/x/conversations.json"),
        b"[\n  {\n    \"conversation_id\": \"0012\""
    ));
    // claude.ai's export has the same file name.
    assert!(!recognize(
        Path::new("/x/conversations.json"),
        b"[{\"uuid\": \"3b33\", \"name\": \"x\"}]"
    ));
}

#[test]
fn project_gpt_and_template_facts_keep_their_conversation_urls() {
    let mut project = conv("project-chat", "", vec![]);
    project["gizmo_id"] = json!("g-p-project");
    project["conversation_template_id"] = json!("template-opaque");
    let mut gpt = conv("gpt-chat", "", vec![]);
    gpt["gizmo_id"] = json!("g-custom");
    let mut plain = conv("plain-chat", "", vec![]);
    plain["gizmo_id"] = Json::Null;
    plain["conversation_template_id"] = json!("");
    plain.as_object_mut().unwrap().remove("update_time");
    let step = export(&[project, gpt, plain]);
    let facts: Vec<_> = step
        .output
        .events
        .iter()
        .filter(|event| event.kind == "conversation_facts")
        .collect();
    assert_eq!(facts.len(), 3);
    assert_eq!(
        facts.iter().map(|event| &event.data).collect::<Vec<_>>(),
        [
            &json!({"project":"g-p-project", "template":"template-opaque",
                    "web_url":"https://chatgpt.com/g/g-p-project/c/project-chat"}),
            &json!({"gpt":"g-custom", "web_url":"https://chatgpt.com/g/g-custom/c/gpt-chat"}),
            &json!({"web_url":"https://chatgpt.com/c/plain-chat"}),
        ],
    );
    for fact in &facts {
        let settings = step
            .output
            .events
            .iter()
            .find(|event| event.kind == "conversation_settings" && event.session == fact.session)
            .unwrap();
        assert_eq!(fact.at, settings.at);
        assert_eq!(fact.at_quality, settings.at_quality);
        assert_eq!(fact.at_precision, settings.at_precision);
        assert_eq!(fact.at_until, settings.at_until);
    }
    assert_eq!(facts[2].at, None);
    let original = step
        .output
        .events
        .iter()
        .find(|event| {
            event.kind == "conversation_settings"
                && event.session.as_ref().unwrap().native().as_str() == "project-chat"
        })
        .unwrap();
    assert_eq!(original.data["gizmo_id"], "g-p-project");
    assert_eq!(original.data["conversation_template_id"], "template-opaque");
}

#[test]
fn research_task_fields_are_message_facts_and_keep_the_original_metadata() {
    let mut research = msg("research", "assistant", 1.0, text(&["Research started"]));
    research["metadata"] = json!({"async_task_id":"task-opaque",
        "async_task_title":"Compare the records", "async_task_type":"deep_research",
        "future_task_field":{"value":7}});
    let mut partial = msg("partial", "assistant", 2.0, text(&["Still running"]));
    partial["metadata"] = json!({"async_task_id":"", "async_task_title":"Known title"});
    let step = export(&[conv(
        "research-chat",
        "n2",
        vec![node("n1", None, research), node("n2", Some("n1"), partial)],
    )]);
    let rows = &step.output.messages;
    let tasks: Vec<_> = rows
        .iter()
        .flat_map(|row| row.message.blocks.iter())
        .filter_map(|block| match block {
            Block::Other { type_name, data } if type_name == "research_task" => Some(data),
            _ => None,
        })
        .collect();
    assert_eq!(
        tasks,
        [
            &json!({"id":"task-opaque", "title":"Compare the records", "kind":"deep_research"}),
            &json!({"title":"Known title"}),
        ]
    );
    assert_eq!(
        rows[0].message.attrs["metadata"]["future_task_field"],
        json!({"value":7})
    );
    assert!(
        matches!(&rows[0].message.blocks[0], Block::Text { text } if text == "Research started")
    );
}

#[test]
fn browsing_content_uses_the_same_fields_in_messages_and_tool_results() {
    let cases = [
        (
            json!({"content_type":"tether_quote", "url":"https://example.test/one",
                "title":"Named result", "domain":"ignored.example", "text":"Main text",
                "snippet":"Ignored snippet", "result":"Ignored result", "tether_id":"tether-1",
                "ref_id":"ignored-ref"}),
            json!({"content_type":"tether_quote", "url":"https://example.test/one",
                "title":"Named result", "text":"Main text", "source_id":"tether-1"}),
        ),
        (
            json!({"content_type":"tether_browsing_display", "url":"https://example.test/two",
                "title":"", "domain":"fallback.example", "text":"", "snippet":"Snippet text",
                "result":"Ignored result", "tether_id":"", "ref_id":"ref-2"}),
            json!({"content_type":"tether_browsing_display", "url":"https://example.test/two",
                "title":"fallback.example", "text":"Snippet text", "source_id":"ref-2"}),
        ),
        (
            json!({"content_type":"sonic_webpage", "domain":"sonic.example", "result":"Result text"}),
            json!({"content_type":"sonic_webpage", "title":"sonic.example", "text":"Result text"}),
        ),
    ];
    for (content, expected) in cases {
        let mut call = msg("call", "assistant", 2.0, text(&["{}"]));
        call["recipient"] = json!("web.run");
        let step = export(&[conv(
            "browsing",
            "result-node",
            vec![
                node(
                    "visible-node",
                    None,
                    msg("visible", "assistant", 1.0, content.clone()),
                ),
                node("call-node", Some("visible-node"), call),
                node(
                    "result-node",
                    Some("call-node"),
                    msg("result", "tool", 3.0, content),
                ),
            ],
        )]);
        let rows = &step.output.messages;
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows[0].message.blocks,
            [Block::Other {
                type_name: "web_result".into(),
                data: expected.clone(),
            }]
        );
        let Block::ToolResult {
            call_id,
            outcome,
            output,
            data,
            ..
        } = &rows[2].message.blocks[0]
        else {
            panic!("browsing tool output must remain a tool result")
        };
        assert_eq!(call_id.as_deref(), Some("call"));
        assert_eq!(*outcome, ToolOutcome::Ok);
        assert_eq!(output.as_deref(), expected.str("text"));
        assert_eq!(data.as_ref(), Some(&expected));
        assert_eq!(rows[2].message.role, Role::User);
        assert_eq!(rows[2].message.prompt_kind, Some(PromptKind::Harness));
        assert_eq!(
            rows[2].message.parent.as_ref().unwrap().native().as_str(),
            "call"
        );
    }
}

#[test]
fn partial_browsing_content_does_not_invent_fields_or_change_unknown_content() {
    let missing = json!({"content_type":"tether_quote", "title":"", "text":"", "ref_id":""});
    let future = json!({"content_type":"future_browsing", "text":"still retained", "new":7});
    let step = export(&[conv(
        "partial-browsing",
        "n2",
        vec![
            node("n1", None, msg("partial", "assistant", 1.0, missing)),
            node(
                "n2",
                Some("n1"),
                msg("future", "assistant", 2.0, future.clone()),
            ),
        ],
    )]);
    assert_eq!(
        step.output.messages[0].message.blocks,
        [Block::Other {
            type_name: "web_result".into(),
            data: json!({"content_type":"tether_quote"}),
        }]
    );
    assert_eq!(
        step.output.messages[1].message.blocks,
        [Block::Other {
            type_name: "future_browsing".into(),
            data: future,
        }]
    );
}

#[test]
fn conversation_headers_can_start_with_project_or_template_fields() {
    for bytes in [
        br#"[{"gizmo_id":"g-p-project","conversation_id":"c","mapping":{}}]"#.as_slice(),
        br#"[{"conversation_template_id":"template","conversation_id":"c","mapping":{}}]"#
            .as_slice(),
    ] {
        assert!(recognize(Path::new("conversations.json"), bytes));
    }
}

#[test]
fn browsing_and_project_facts_match_between_export_and_browser_capture() {
    let mut message = msg(
        "shared-message",
        "assistant",
        1.0,
        json!({"content_type":"tether_quote", "text":"Recorded result", "ref_id":"shared-ref"}),
    );
    message["metadata"] = json!({"async_task_id":"shared-task", "async_task_type":"deep_research"});
    let mut conversation = conv("shared-chat", "n1", vec![node("n1", None, message)]);
    conversation["gizmo_id"] = json!("g-p-shared");
    conversation["conversation_template_id"] = json!("shared-template");
    let exported = export(&[conversation.clone()]);
    let bytes = serde_json::to_vec(&json!({
        "polylogue_capture_kind":"conversation", "schema_version":1, "source":"browser-extension",
        "provenance":{"source_url":"https://chatgpt.com/g/g-p-shared/c/shared-chat",
                      "captured_at":"2026-02-01T00:00:00Z", "capture_mode":"auto"},
        "provider_meta":{}, "raw_provider_payload":conversation,
        "session":{"provider":"chatgpt", "provider_session_id":"shared-chat", "turns":[]}
    }))
    .unwrap();
    let captured = crate::parse::run_document(&crate::browser_capture::Capture, &bytes).unwrap();
    assert_eq!(captured.output.messages.len(), 1);
    assert_eq!(
        captured.output.messages[0].message.id,
        exported.output.messages[0].message.id
    );
    assert_eq!(
        captured.output.messages[0].message.blocks,
        exported.output.messages[0].message.blocks
    );
    let facts = |step: &crate::parse::Step| {
        step.output
            .events
            .iter()
            .filter(|event| event.kind == "conversation_facts")
            .map(|event| (event.session.clone(), event.data.clone()))
            .collect::<Vec<_>>()
    };
    let exported_facts = facts(&exported);
    assert_eq!(exported_facts.len(), 1);
    assert_eq!(facts(&captured), exported_facts);
}
