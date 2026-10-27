use super::*;
use sinex_model::{
    Class, EventPayload, KeyKind, MaterialId, Origin, ReadingOrder, RecordLocation, ReplaceRule,
};

fn at(seconds: i64) -> Timestamp {
    Timestamp::from_unix_secs(seconds).unwrap()
}

fn message(text: &str) -> Message {
    Message {
        message: "claude-code:m1".into(),
        session: "claude-code:s1".into(),
        parent: None,
        role: "user".into(),
        written_by: "human".into(),
        prompt_kind: Some("prose".into()),
        at: Some(at(1_790_000_000)),
        model: None,
        text: text.into(),
        usage: Usage::default(),
        cost_nusd: None,
        stop_reason: None,
        duration_ms: None,
        attachments: vec![],
        copies: vec![],
    }
}

fn origin() -> Origin {
    Origin::Record {
        location: RecordLocation {
            material: MaterialId::new(),
            anchor: 0,
        },
        output_index: 0,
    }
}

#[devkit::test]
fn sessions_messages_and_calls_are_keyed_by_polylogue_ids_and_replaced_when_newer() {
    let keys: [(KeyKind, ReplaceRule, &'static [&'static str]); 4] = [
        (Session::KEY, Session::REPLACE, &["session"]),
        (Message::KEY, Message::REPLACE, &["message"]),
        (ToolCall::KEY, ToolCall::REPLACE, &["call"]),
        (Turn::KEY, Turn::REPLACE, &["turn"]),
    ];
    for (key, replace, field) in keys {
        assert_eq!(key, KeyKind::Fields(field));
        assert_eq!(replace, ReplaceRule::ReplaceWhenNewer);
    }
    assert_eq!(Note::REPLACE, ReplaceRule::SuppressRepeat);
    let a = Message::payload_schema();
    assert!(
        a["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f == "text")
    );
}

#[devkit::test]
fn message_text_is_stored_guarded_whatever_the_producer_asks() {
    assert_eq!(Message::CLASS, Class::Guarded);
    assert_eq!(Session::CLASS, Class::Open);
    let event = sinex_model::Event::build(
        &message("a prompt"),
        "polylogue",
        origin(),
        ReadingOrder(1),
        [0; 32],
        Class::Open,
    )
    .unwrap();
    assert_eq!(event.class, Class::Guarded);
    let sealed = sinex_model::Event::build(
        &message("a prompt"),
        "polylogue",
        origin(),
        ReadingOrder(1),
        [0; 32],
        Class::Sealed,
    )
    .unwrap();
    assert_eq!(sealed.class, Class::Sealed);
}

#[devkit::test]
fn the_same_message_read_twice_has_one_equivalence_key() {
    let build = |text: &str| {
        sinex_model::Event::build(
            &message(text),
            "polylogue",
            origin(),
            ReadingOrder(1),
            [0; 32],
            Class::Guarded,
        )
        .unwrap()
    };
    assert_eq!(
        build("first reading").equivalence_key,
        build("a newer parser's reading").equivalence_key
    );
}

use serde_json::{Value, json};

fn example(schema: &Value, root: &Value, optional: bool) -> Value {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        return example(
            root.pointer(reference.strip_prefix('#').unwrap()).unwrap(),
            root,
            optional,
        );
    }
    if let Some(v) = schema.get("const") {
        return v.clone();
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return values[0].clone();
    }
    if let Some(branches) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(Value::as_array)
    {
        let branch = if optional {
            branches
                .iter()
                .find(|s| s.get("type").and_then(Value::as_str) != Some("null"))
                .unwrap()
        } else {
            &branches[0]
        };
        return example(branch, root, optional);
    }
    let ty = schema.get("type").unwrap_or(&Value::Null);
    let ty = ty
        .as_str()
        .or_else(|| {
            ty.as_array()
                .and_then(|a| a.iter().filter_map(Value::as_str).find(|s| *s != "null"))
        })
        .unwrap_or("object");
    match ty {
        "string" => match schema.get("format").and_then(Value::as_str) {
            Some("date-time") => json!("2026-10-27T14:00:00+02:00"),
            Some("uuid") => json!("019a26cd-e800-7000-8000-000000000001"),
            _ => json!("sample"),
        },
        "integer" | "number" => schema.get("minimum").cloned().unwrap_or(json!(0)),
        "boolean" => json!(optional),
        "null" => Value::Null,
        "array" => json!([]),
        _ => {
            let mut object = serde_json::Map::new();
            if let Some(fields) = schema.get("properties").and_then(Value::as_object) {
                let required = schema.get("required").and_then(Value::as_array);
                for (name, field) in fields {
                    if optional || required.is_some_and(|fields| fields.contains(&json!(name))) {
                        object.insert(name.clone(), example(field, root, optional));
                    }
                }
            }
            Value::Object(object)
        }
    }
}

fn parity<P: EventPayload>() {
    let d = super::declarations("polylogue")
        .into_iter()
        .find(|d| d.kind.name == P::EVENT_TYPE)
        .unwrap();
    for optional in [false, true] {
        let schema = d.payload_schema.as_ref().unwrap();
        let payload = example(schema, schema, optional);
        let typed: P = serde_json::from_value(payload).unwrap();
        let event = sinex_model::Event::build(
            &typed,
            "polylogue",
            origin(),
            ReadingOrder(1),
            [0; 32],
            Class::Open,
        )
        .unwrap();
        let evaluated = d.evaluate(&event.payload, Class::Open).unwrap();
        assert_eq!(evaluated.equivalence_key, event.equivalence_key);
        assert_eq!(evaluated.ts_orig, event.ts_orig);
        assert_eq!(evaluated.record_state, event.record_state);
        assert_eq!(evaluated.owned_blobs, event.owned_blobs);
        assert_eq!(evaluated.class, event.class);
    }
}

#[test]
fn all_ten_producer_declarations_preserve_typed_keys_clocks_and_classes() {
    assert_eq!(super::declarations("polylogue").len(), 10);
    parity::<Session>();
    parity::<Message>();
    parity::<ToolCall>();
    parity::<Turn>();
    parity::<Delivery>();
    parity::<DeliveryOutcome>();
    parity::<Note>();
    parity::<Setting>();
    parity::<KeptResult>();
    parity::<Excision>();
}
