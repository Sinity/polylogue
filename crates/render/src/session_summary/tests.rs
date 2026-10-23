use super::*;
use crate::tests::conversation;
use polylogue_model::{SessionId, SessionKind};

#[test]
fn a_card_counts_outcomes_not_words() {
    let mut s = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    s.offer_title("Fix the tests", polylogue_model::TitleSource::Summary);
    let c = summarize(
        &s,
        s.title.as_ref().map(|t| t.text.as_str()),
        &conversation(),
    );
    assert_eq!(c.title, "Fix the tests");
    assert_eq!(c.lines[0], "3 messages, 1 prompts, 1 tool calls");
    assert!(c.lines.contains(&"tools: Bash 1".to_owned()));
    assert!(
        c.lines
            .contains(&"ran: cargo test ×1 (1 failed)".to_owned()),
        "{:?}",
        c.lines
    );
    assert!(
        c.lines
            .contains(&"last failure: cargo test: error (exit 101)".to_owned())
    );
}
