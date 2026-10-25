use super::*;
use polylogue_model::{ErrorReason, MessageId, SessionId, SessionKind, Usage};

pub(crate) fn message(
    id: &str,
    role: Role,
    kind: Option<PromptKind>,
    blocks: Vec<Block>,
) -> Message {
    Message {
        id: MessageId::parse(&format!("claude-code:s/{id}")).unwrap(),
        session: SessionId::parse("claude-code:s").unwrap(),
        parent: None,
        role,
        at: None,
        model: None,
        prompt_kind: kind,
        blocks,
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
    }
}

pub(crate) fn conversation() -> Vec<Message> {
    vec![
        message(
            "1",
            Role::User,
            Some(PromptKind::Prose),
            vec![Block::Text {
                text: "run the tests; token sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-AAAAAAAA".into(),
            }],
        ),
        message(
            "2",
            Role::Assistant,
            None,
            vec![
                Block::Thinking {
                    text: Some("plan".into()),
                    signature: None,
                    redacted: false,
                },
                Block::Text {
                    text: "Running them.".into(),
                },
                Block::ToolCall {
                    run_by_script: false,
                    call_id: Some("c1".into()),
                    tool: Some("Bash".into()),
                    action: polylogue_model::ToolAction::Shell,
                    input: serde_json::json!({"command": "cargo test"}),
                    command: Some("cargo test".into()),
                    path: None,
                },
            ],
        ),
        message(
            "3",
            Role::User,
            Some(PromptKind::Harness),
            vec![Block::ToolResult {
                call_id: Some("c1".into()),
                outcome: ToolOutcome::Error {
                    reason: Some(ErrorReason::ExitCode(101)),
                },
                output: Some("test failed".into()),
                data: None,
                attachments: vec![],
            }],
        ),
    ]
}

fn render_to(format: Format, view: View) -> String {
    let session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    let ms = conversation();
    let value = crate::session(&session, None, ms, view, format, &[]).unwrap();
    if format == Format::Json {
        value.to_string()
    } else {
        value["content"].as_str().unwrap().to_owned()
    }
}

#[test]
fn prose_formats_mask_credentials_before_rendering() {
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        let t = render_to(format, View::Full);
        assert!(!t.contains("sk-ant-api03-AAAA"), "{format:?}: {t}");
        assert!(t.contains("run the tests"), "{format:?}");
    }
}

#[test]
fn views_filter_the_one_stream() {
    let prompts = render_to(Format::Markdown, View::Prompts);
    assert!(prompts.contains("run the tests"));
    assert!(!prompts.contains("Running them."));
    assert!(!prompts.contains("test failed"));

    let dialogue = render_to(Format::Markdown, View::Dialogue);
    assert!(dialogue.contains("Running them."));
    assert!(dialogue.contains("> plan"));
    assert!(!dialogue.contains("cargo test"));

    let actions = render_to(Format::Markdown, View::Actions);
    assert!(actions.contains("→ **Bash**"));
    assert!(actions.contains("← **error (exit 101)**"));
    assert!(!actions.contains("Running them."));
}

#[test]
fn markdown_fences_survive_backticks_in_output() {
    assert_eq!(fence("a ``` b"), "````");
    assert_eq!(fence("plain"), "```");
}

#[test]
fn html_escapes_text() {
    let session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    let m = message(
        "1",
        Role::User,
        None,
        vec![Block::Text {
            text: "<script>".into(),
        }],
    );
    let value = crate::session(&session, None, vec![m], View::Full, Format::Html, &[]).unwrap();
    let t = value["content"].as_str().unwrap();
    assert!(t.contains("&lt;script&gt;"));
    assert!(!t.contains("<script>"));
}

#[test]
fn json_keeps_typed_action_blocks() {
    let t = render_to(Format::Json, View::Actions);
    let v: serde_json::Value = serde_json::from_str(&t).unwrap();
    let blocks: Vec<&str> = v["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|message| message["blocks"].as_array().into_iter().flatten())
        .map(|block| block["type"].as_str().unwrap())
        .collect();
    assert_eq!(blocks, ["tool_call", "tool_result"]);
}

mod structured;
