use super::*;
use serde_json::json;

static FIELDS: Fields = Fields::new(&[
    ("type", Rule::Read),
    ("message", Rule::Read),
    ("message.content", Rule::Read),
    ("message.content[].text", Rule::Read),
    ("message.content[].type", Rule::Read),
    ("message.usage", Rule::Read),
    ("slug", Rule::Excluded("display label only")),
]);

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
    let mut got = unaccounted(&record, &FIELDS);
    got.sort();
    assert_eq!(
        got,
        [
            "extra",
            "message.content[].citations",
            "message.stop_reason",
            "message.usage.anything",
            "message.usage.input_tokens"
        ]
    );
}

mod equivalence;

#[test]
fn unread_descendants_include_nested_arrays_and_empty_values() {
    static RULES: Fields = Fields::new(&[
        ("read", Rule::Read),
        ("read.used", Rule::Read),
        ("indexed", Rule::Indexed("index JSON")),
        ("kept", Rule::Kept("raw input")),
        ("ignored", Rule::Excluded("UI setting")),
    ]);
    let value = json!({"read":{"used":1,"nested":[{"empty":{},"values":[[1,2]]}],"array":[]},"indexed":{"any":[{"depth":1}]},"kept":{"any":2},"ignored":{"any":3}});
    let mut got = unaccounted(&value, &RULES);
    got.sort();
    assert_eq!(
        got,
        [
            "read.array",
            "read.nested[].empty",
            "read.nested[].values[][]",
            "read.nested[].values[][]"
        ]
    );
}
