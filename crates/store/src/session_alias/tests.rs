use super::*;

#[test]
fn a_copy_message_takes_its_id_in_the_session() {
    assert_eq!(
        renamed(
            "ai-studio:content-ab~user-0123-0",
            "ai-studio:content-ab",
            "ai-studio:1AbC"
        ),
        Some("ai-studio:1AbC~user-0123-0".into())
    );
    assert_eq!(
        renamed(
            "ai-studio:10:content-ab#id:6:answer",
            "ai-studio:content-ab",
            "ai-studio:1AbC"
        ),
        Some("ai-studio:4:1AbC#id:6:answer".into())
    );
    assert_eq!(
        renamed(
            "ai-studio:other~user-0-0",
            "ai-studio:content-ab",
            "ai-studio:1AbC"
        ),
        None
    );
    assert_eq!(
        message_session("ai-studio:10:content-ab#id:6:answer").as_deref(),
        Some("ai-studio:content-ab")
    );
    assert_eq!(message_session("claude-code:plain-uuid"), None);
}
