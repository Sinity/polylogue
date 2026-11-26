use super::*;
use serde_json::json;

/// The content name [`content_name`] gives, computed from hand-written
/// canonical JSON of `[role, content]`, independently of the encoder.
pub(crate) fn expected_name(session_native: &str, role: &str, canonical: &str) -> String {
    let mut bytes = b"polylogue-message-content\0".to_vec();
    bytes.extend_from_slice(canonical.as_bytes());
    let hash = blake3::hash(&bytes).to_hex();
    format!("{session_native}~{role}-{}", &hash.as_str()[..16])
}

/// The settled native ID of a content-named message.
pub(crate) fn expected(
    session_native: &str,
    role: &str,
    canonical: &str,
    occurrence: u32,
) -> String {
    format!(
        "{}-{occurrence}",
        expected_name(session_native, role, canonical)
    )
}

fn session(native: &str) -> SessionId {
    SessionId::parse(&format!("grok-cli:{native}")).unwrap()
}

#[test]
fn a_message_is_written_as_its_session_role_content_and_position() {
    let blocks = json!([{"type": "text", "text": "hi"}]);
    let canonical = r#"["user",[{"text":"hi","type":"text"}]]"#;
    let id = named(&session("s"), Role::User, &blocks, 42).unwrap();
    assert_eq!(
        id.native().as_str(),
        format!("{}@42", expected_name("s", "user", canonical))
    );
    assert_eq!(id.origin().as_str(), "grok-cli");
    let name = expected_name("s", "user", canonical);
    assert_eq!(as_written(&id), Some((name.as_str(), 42)));
    let answered = named(&session("s"), Role::Assistant, &blocks, 42).unwrap();
    assert!(answered.native().as_str().starts_with(&expected_name(
        "s",
        "assistant",
        r#"["assistant",[{"text":"hi","type":"text"}]]"#
    )));
    let elsewhere = named(&session("t"), Role::User, &blocks, 42).unwrap();
    assert!(elsewhere.native().as_str().starts_with("t~user-"));
}

#[test]
fn a_settled_id_gives_back_its_content_name() {
    let name = expected_name("s", "user", r#"["user",["same"]]"#);
    let id = numbered("grok-cli", &name, 3).unwrap();
    assert_eq!(id.native().as_str(), format!("{name}-3"));
    assert_eq!(settled_name(id.native().as_str()), Some(name.as_str()));
    assert_eq!(as_written(&id), None);
    assert_eq!(settled_name("toolu_01-7"), None);
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
    let expected = named(&session, Role::Assistant, &blocks, 0);
    let polylogue_model::Block::ToolCall { written_text, .. } = &mut blocks[0] else {
        unreachable!()
    };
    *written_text = Some("exact patch".into());
    assert_eq!(
        named_blocks(&session, Role::Assistant, &mut blocks, 0),
        expected
    );
    let polylogue_model::Block::ToolCall { written_text, .. } = &blocks[0] else {
        unreachable!()
    };
    assert_eq!(written_text.as_deref(), Some("exact patch"));
}

mod goldens;
