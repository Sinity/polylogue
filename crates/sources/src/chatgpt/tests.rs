use super::*;

use serde_json::json;

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
    crate::array::fixture::parse(&mut Parse::Array(Box::new(Export)), &bytes).unwrap()
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
    assert_eq!(step.accounting.framed, 5, "one conversation, four messages");
    assert_eq!(step.accounting.materialized["text"], 4);
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
fn empty_objects_and_broken_items_are_accounted_for() {
    let mut no_role = msg("x", "user", 1.0, text(&["x"]));
    no_role["author"] = json!({});
    let c = conv("c4", "n-x", vec![node("n-x", None, no_role)]);
    let step = export(&[json!({}), json!({"id": "c5"}), c]);
    assert_eq!(step.accounting.framed, 4);
    assert_eq!(
        step.accounting.unknown["conversation: empty object in the export"],
        1
    );
    assert_eq!(step.accounting.refused["conversation: no mapping"], 1);
    assert_eq!(step.accounting.refused["text: no author role"], 1);
    assert_eq!(step.accounting.materialized["conversation"], 1);
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
    assert_eq!(link.at.as_ref().unwrap().native().as_str(), "a");
    let tail = &step.output.messages[2].message;
    assert_eq!(tail.parent.as_ref().unwrap().native().as_str(), "a");
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
