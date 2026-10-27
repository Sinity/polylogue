use super::*;
use serde_json::json;

const HOSTILE: &str = "\x1b]0;title\x07\x1b]2;title\x1b\\\x1b]52;c;YQ==\x07\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\x1b[2J\x1b\u{9b}2J\u{202e}";

fn safe(text: &str) {
    assert!(
        !text
            .chars()
            .any(|c| (c.is_control() && c != '\n' && c != '\t') || c == '\u{202e}'),
        "{text:?}"
    );
}

#[test]
fn terminal_printers_escape_archived_text() {
    let data = json!({
        "content":HOSTILE, "text":HOSTILE, "next":HOSTILE,
        "prepared":{"text":HOSTILE},
        "actions":[{"id":HOSTILE,"conversation":HOSTILE,"last_error":HOSTILE}],
        "jobs":[{"id":HOSTILE,"detail":HOSTILE}],
        "harness_command":{"program":HOSTILE,"args":[HOSTILE],"cwd":HOSTILE},
        "context_assisted":{"program":HOSTILE,"args":[HOSTILE],"cwd":HOSTILE}
    });
    for printer in [read, context, resume, chat_actions, backfill_jobs] {
        let text = printer(&data);
        safe(&text);
        assert!(text.contains("\\x1b"), "{text:?}");
    }
    safe(&read(&json!({"text":HOSTILE,"next":HOSTILE})));
    safe(&resume(
        &json!([{"session":{"title":HOSTILE,"id":HOSTILE},"open":{"todo":HOSTILE,"waiting_on":HOSTILE},"follow_ups":[HOSTILE]}]),
    ));
    let status = json!({"notices":[HOSTILE], "errors":[HOSTILE]});
    let text = super::json(&status);
    safe(&text);
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), status);
}
