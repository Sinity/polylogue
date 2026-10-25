use super::*;

#[test]
fn targets_parse_and_a_text_target_refuses_plainly() {
    assert!(matches!(
        Scope::parse("session:claude-code:abc").unwrap(),
        Scope::Session(_)
    ));
    assert!(matches!(
        Scope::parse("claude-code:abc").unwrap(),
        Scope::Session(_)
    ));
    assert!(matches!(
        Scope::parse("input:0193").unwrap(),
        Scope::Input(_)
    ));
    let text = Scope::parse("text:sk-secret").unwrap_err();
    assert!(text.contains("not supported"), "{text}");
    assert!(!text.contains("sk-secret"));
    assert!(Scope::parse("nonsense").is_err());
}
