use super::*;

const HOSTILE: &str = "\x1b]0;title\x07\x1b]2;title\x1b\\\x1b]52;c;YQ==\x07\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\x1b[2J\x1b\u{9b}2J\u{202e}abc\u{2066}xyz\u{2069}\r\x7f";

#[test]
fn controls_are_visible_and_escaping_is_idempotent() {
    let escaped = escape(HOSTILE);
    assert!(escaped.contains("\\x1b]52;c;YQ==\\x07"));
    assert!(escaped.contains("\\x9b2J\\u{202e}"));
    assert!(escaped.ends_with("\\x0d\\x7f"));
    assert_eq!(escape(&escaped), escaped);
    assert_eq!(escape("λ🙂\n\t"), "λ🙂\n\t");
    for code in 0..=159 {
        let c = char::from_u32(code).unwrap();
        if c.is_control() && c != '\n' && c != '\t' {
            assert_eq!(escape(&c.to_string()), format!("\\x{code:02x}"));
        }
    }
}

#[test]
fn color_requires_a_terminal_and_no_color_takes_precedence() {
    assert!(color_for(true, false));
    assert!(!color_for(false, false));
    assert!(!color_for(true, true));
    assert!(!color_for(false, true));
}

#[test]
fn json_controls_round_trip_without_terminal_sequences() {
    let value = serde_json::json!({HOSTILE: HOSTILE});
    for pretty in [true, false] {
        let text = json(&value, pretty);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text).unwrap(),
            value
        );
        assert!(
            !text
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t')
        );
        assert!(!text.contains('\u{202e}'));
    }
}
