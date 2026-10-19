use super::*;
use serde_json::json;

fn result(record: Json, item: Json) -> ToolOutcome {
    let mut blobs = Vec::new();
    match block(&record, &item, true, &mut blobs) {
        Block::ToolResult { outcome, .. } => outcome,
        other => panic!("{other:?}"),
    }
}

#[test]
fn outcome_prefers_denial_then_interruption_then_is_error() {
    let item = json!({"type":"tool_result","tool_use_id":"t","is_error":true,"content":"no"});
    assert_eq!(
        result(json!({"toolDenialKind":"permission_rule"}), item.clone()),
        ToolOutcome::Error {
            reason: Some(ErrorReason::Denied("permission_rule".into()))
        }
    );
    assert_eq!(
        result(json!({"toolUseResult":{"interrupted":true}}), item.clone()),
        ToolOutcome::Error {
            reason: Some(ErrorReason::Interrupted)
        }
    );
    assert_eq!(result(json!({}), item), ToolOutcome::Error { reason: None });
}

#[test]
fn exit_code_comes_from_the_last_line_that_states_one() {
    let item =
        json!({"type":"tool_result","is_error":true,"content":"Exit code 1\nmore\nexit code: 2"});
    assert_eq!(
        result(json!({}), item),
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(2))
        }
    );
}

#[test]
fn background_launch_and_unreadable_status_are_unknown() {
    let item = json!({"type":"tool_result","backgroundTaskId":"b1","content":"started"});
    assert_eq!(
        result(json!({}), item),
        ToolOutcome::Unknown {
            reason: UnknownReason::RunningInBackground
        }
    );
    let item = json!({"type":"tool_result","is_error":"yes"});
    assert_eq!(
        result(json!({}), item),
        ToolOutcome::Unknown {
            reason: UnknownReason::StatusFieldUnreadable
        }
    );
}

#[test]
fn tool_use_result_applies_only_to_a_sole_result() {
    let record = json!({"toolUseResult":{"interrupted":true}});
    let content = json!([
        {"type":"tool_result","tool_use_id":"a"},
        {"type":"tool_result","tool_use_id":"b"}
    ]);
    let mut blobs = Vec::new();
    for b in blocks(&record, &content, &mut blobs) {
        let Block::ToolResult { outcome, data, .. } = b else {
            panic!()
        };
        assert_eq!(outcome, ToolOutcome::Ok);
        assert_eq!(data, None);
    }
}

#[test]
fn inline_images_become_blobs() {
    let data = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG fake");
    let content =
        json!([{"type":"image","source":{"type":"base64","media_type":"image/png","data":data}}]);
    let mut blobs = Vec::new();
    let b = blocks(&json!({}), &content, &mut blobs);
    assert_eq!(blobs.len(), 1);
    let Block::Image(a) = &b[0] else { panic!() };
    assert_eq!(a.blob, Some(BlobHash::of(b"\x89PNG fake")));
    assert_eq!(a.size, Some(9));
}

#[test]
fn prompt_kinds() {
    let r = json!({});
    assert_eq!(prompt_kind(&r, &json!("fix the build")), PromptKind::Prose);
    assert_eq!(prompt_kind(&r, &json!("/model opus")), PromptKind::Command);
    assert_eq!(
        prompt_kind(&r, &json!("/usr/bin/env is weird\nsee")),
        PromptKind::Prose
    );
    assert_eq!(
        prompt_kind(&r, &json!("<command-name>/clear</command-name>")),
        PromptKind::Command
    );
    assert_eq!(
        prompt_kind(&r, &json!("[Request interrupted by user]")),
        PromptKind::Harness
    );
    assert_eq!(
        prompt_kind(&r, &json!([{"type":"tool_result","tool_use_id":"t"}])),
        PromptKind::Harness
    );
    assert_eq!(
        prompt_kind(&json!({"isMeta":true}), &json!("hi")),
        PromptKind::Harness
    );
}
