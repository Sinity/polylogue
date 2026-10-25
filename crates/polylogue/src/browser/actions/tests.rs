use super::*;

fn request(op: &str, conversation: &str, policy: &str) -> ActionRequest {
    serde_json::from_value(serde_json::json!({
        "provider": "chatgpt",
        "operation": op,
        "target": {"conversation_id": conversation},
        "text": "summarize this",
        "presentation": {"model_slug": "x", "model_label": "X", "effort_label": "standard"},
        "submit_policy": policy
    }))
    .unwrap()
}

#[test]
fn a_request_is_checked_before_it_is_stored() {
    let none = |_: &BlobHash| None;
    let new = new_action(&request("conversation.create", "new", "stage_only"), none).unwrap();
    assert!(new.stage_only && !new.needs_approval);
    assert_eq!(new.request_key, new.request_hash.to_string());
    let held = new_action(&request("conversation.reply", "c1", "submit_once"), none).unwrap();
    assert!(held.needs_approval);
    assert_eq!(
        new_action(&request("conversation.reply", "new", "stage_only"), none).unwrap_err(),
        Refused(400, "target_does_not_match_operation")
    );
    let mut other = request("conversation.create", "new", "stage_only");
    other.provider = "grok".into();
    assert_eq!(
        new_action(&other, none).unwrap_err(),
        Refused(400, "unsupported_action")
    );
    let mut attached = request("conversation.create", "new", "stage_only");
    let blob = BlobHash::of(b"bytes");
    attached.attachments = vec![AttachmentInput {
        name: "../escape".into(),
        mime_type: "text/plain".into(),
        attachment_ref: blob.to_string(),
    }];
    let stored = |b: &BlobHash| (*b == blob).then_some(5);
    assert_eq!(
        new_action(&attached, stored).unwrap_err(),
        Refused(400, "attachment_name_not_a_basename")
    );
    attached.attachments[0].name = "notes.txt".into();
    assert_eq!(
        new_action(&attached, stored).unwrap().attachments[0].size,
        5
    );
    assert_eq!(
        new_action(&attached, none).unwrap_err(),
        Refused(400, "unknown_attachment")
    );
}
