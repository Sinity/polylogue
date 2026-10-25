use super::*;

#[test]
fn targets_parse_and_a_text_target_refuses_plainly() {
    assert!(matches!(
        ExciseTarget::parse("session:claude-code:abc").unwrap(),
        ExciseTarget::Session(_)
    ));
    assert!(matches!(
        ExciseTarget::parse("claude-code:abc").unwrap(),
        ExciseTarget::Session(_)
    ));
    assert!(matches!(
        ExciseTarget::parse("input:0193").unwrap(),
        ExciseTarget::Input(_)
    ));
    let text = ExciseTarget::parse("text:sk-secret").unwrap_err();
    assert!(text.contains("not supported"), "{text}");
    assert!(!text.contains("sk-secret"));
    assert!(ExciseTarget::parse("nonsense").is_err());
}
