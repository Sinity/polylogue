use super::*;
use serde_json::json;

const FIELDS: Fields = &[
    ("type", Rule::Read),
    ("message", Rule::Read),
    ("message.content", Rule::Read),
    ("message.content[].text", Rule::Read),
    ("message.content[].type", Rule::Read),
    ("message.usage", Rule::Read),
    ("slug", Rule::Excluded("display label only")),
];

#[test]
fn lists_fields_nobody_declared() {
    let record = json!({
        "type": "assistant",
        "slug": "x",
        "extra": 1,
        "message": {
            "usage": {"input_tokens": 1, "anything": 2},
            "content": [{"type": "text", "text": "hi", "citations": []}],
            "stop_reason": null
        }
    });
    let mut got = unaccounted(&record, FIELDS);
    got.sort();
    assert_eq!(
        got,
        [
            "extra",
            "message.content[].citations",
            "message.stop_reason"
        ]
    );
}
