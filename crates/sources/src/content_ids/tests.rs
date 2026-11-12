use super::*;
use serde_json::json;

/// The native ID [`ContentIds`] gives, computed from hand-written canonical
/// JSON of `[role, content]`, independently of the encoder.
pub(crate) fn expected(
    session_native: &str,
    role: &str,
    canonical: &str,
    occurrence: u32,
) -> String {
    let mut bytes = b"polylogue-message-content\0".to_vec();
    bytes.extend_from_slice(canonical.as_bytes());
    let hash = blake3::hash(&bytes).to_hex();
    format!(
        "{session_native}~{role}-{}-{occurrence}",
        &hash.as_str()[..16]
    )
}

fn session(native: &str) -> SessionId {
    SessionId::parse(&format!("grok-cli:{native}")).unwrap()
}

#[test]
fn a_message_is_named_by_session_role_content_and_occurrence() {
    let mut ids = ContentIds::default();
    let blocks = json!([{"type": "text", "text": "hi"}]);
    let first = ids.next(&session("s"), Role::User, &blocks).unwrap();
    let canonical = r#"["user",[{"text":"hi","type":"text"}]]"#;
    assert_eq!(first.native().as_str(), expected("s", "user", canonical, 0));
    assert_eq!(first.origin().as_str(), "grok-cli");
    let again = ids.next(&session("s"), Role::User, &blocks).unwrap();
    assert_eq!(again.native().as_str(), expected("s", "user", canonical, 1));
    let answered = ids.next(&session("s"), Role::Assistant, &blocks).unwrap();
    assert_eq!(
        answered.native().as_str(),
        expected(
            "s",
            "assistant",
            r#"["assistant",[{"text":"hi","type":"text"}]]"#,
            0
        )
    );
    let elsewhere = ids.next(&session("t"), Role::User, &blocks).unwrap();
    assert_eq!(
        elsewhere.native().as_str(),
        expected("t", "user", canonical, 0)
    );
}

#[test]
fn carried_counts_continue_after_a_resumed_parse() {
    let mut ids = ContentIds::default();
    let blocks = json!(["same"]);
    ids.next(&session("s"), Role::User, &blocks).unwrap();
    let carried: ContentIds = serde_json::from_slice(&serde_json::to_vec(&ids).unwrap()).unwrap();
    let mut resumed = carried;
    assert_eq!(
        resumed.next(&session("s"), Role::User, &blocks).unwrap(),
        ids.next(&session("s"), Role::User, &blocks).unwrap()
    );
    assert_eq!(
        serde_json::to_string(&ContentIds::default()).unwrap(),
        "{}",
        "an unused counter adds nothing to a parser's carried state"
    );
}

#[test]
fn removing_an_earlier_equal_message_renumbers_the_later_copies() {
    let a = json!(["A"]);
    let x = json!(["X"]);
    let ids = |contents: &[&serde_json::Value]| -> Vec<MessageId> {
        let mut ids = ContentIds::default();
        contents
            .iter()
            .map(|content| ids.next(&session("s"), Role::User, *content).unwrap())
            .collect()
    };
    let before = ids(&[&a, &x, &a]);
    let after = ids(&[&x, &a]);
    // X keeps its ID; the remaining A takes the first A's number, and the
    // second A's ID no longer names anything (docs/archive.md).
    assert_eq!(after[0], before[1]);
    assert_eq!(after[1], before[0]);
    assert!(!after.contains(&before[2]));
}

#[test]
fn normalized_written_text_preserves_message_identity_and_original_block() {
    let session = crate::document::session_id("codex", "synthetic").unwrap();
    let mut blocks = vec![polylogue_model::Block::ToolCall {
        written_text: None,
        call_id: None,
        tool: Some("apply_patch".into()),
        action: polylogue_model::ToolAction::FileEdit,
        input: serde_json::json!("exact patch"),
        command: None,
        path: None,
        run_by_script: false,
    }];
    let expected = ContentIds::default().next(&session, Role::Assistant, &blocks);
    let polylogue_model::Block::ToolCall { written_text, .. } = &mut blocks[0] else {
        unreachable!()
    };
    *written_text = Some("exact patch".into());
    assert_eq!(
        ContentIds::default().next_blocks(&session, Role::Assistant, &mut blocks),
        expected
    );
    let polylogue_model::Block::ToolCall { written_text, .. } = &blocks[0] else {
        unreachable!()
    };
    assert_eq!(written_text.as_deref(), Some("exact patch"));
}
