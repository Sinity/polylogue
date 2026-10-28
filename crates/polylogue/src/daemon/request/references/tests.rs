use super::*;
use polylogue_model::{MessageId, SessionId, Target};

#[derive(Serialize)]
enum Example {
    Unit,
    Tuple(MessageId, String),
    #[serde(rename = "new/~")]
    Object {
        target: Target,
        text: String,
    },
    One(SessionId),
}

#[derive(Serialize)]
struct Answer {
    newly_added_field: Vec<Option<MessageId>>,
    variants: Vec<Example>,
    authored: Value,
}

#[test]
fn nested_typed_references_survive_without_declaring_their_json_paths() {
    let secret = "AKIAIOSFODNN7EXAMPLE";
    let message = MessageId::parse(&format!("claude-code:{secret}")).unwrap();
    let session = SessionId::parse(&format!("claude-code:{secret}")).unwrap();
    let answer = Answer {
        newly_added_field: vec![Some(message.clone()), None],
        variants: vec![
            Example::Unit,
            Example::Tuple(message.clone(), secret.into()),
            Example::Object {
                target: Target::Message(message.clone()),
                text: secret.into(),
            },
            Example::One(session.clone()),
        ],
        authored: serde_json::json!({"session":{"id":session},"polylogue_reference":secret}),
    };
    let mut references = References::default();
    let data = references.value("", &answer);
    assert_eq!(data, serde_json::to_value(&answer).unwrap());
    let response = super::super::Response::Ok { data, references }
        .masked()
        .clean(|text| text.replace(secret, "cleaned"));
    let super::super::Response::Ok { data, .. } = response else {
        panic!()
    };
    assert_eq!(data["newly_added_field"][0], message.to_string());
    assert_eq!(data["variants"][1]["Tuple"][0], message.to_string());
    assert_eq!(
        data["variants"][2]["new/~"]["target"],
        Target::Message(message).to_string()
    );
    assert_eq!(data["variants"][3]["One"], session.to_string());
    assert_ne!(data["variants"][1]["Tuple"][1], secret);
    assert_ne!(data["authored"]["session"]["id"], session.to_string());
    assert_ne!(data["authored"]["polylogue_reference"], secret);
}

#[test]
fn exact_numbers_and_raw_json_keep_their_types() {
    #[derive(Serialize)]
    struct Answer {
        number: serde_json::Number,
        raw: Box<serde_json::value::RawValue>,
    }
    let answer = Answer {
        number: serde_json::from_str("123456789012345678901234567890.123456789").unwrap(),
        raw: serde_json::from_str(r#"{"nested":[1,0.1234567890123456789,"authored"]}"#).unwrap(),
    };
    let mut references = References::default();
    assert_eq!(
        references.value("", &answer),
        serde_json::to_value(answer).unwrap()
    );
}

#[test]
fn typed_references_survive_masked_map_keys_and_key_collisions() {
    let secret = "AKIAIOSFODNN7EXAMPLE";
    let id = MessageId::parse(&format!("claude-code:/synthetic/archive/{secret}")).unwrap();
    let label = polylogue_render::mask(secret);
    let map = std::collections::BTreeMap::from([
        (secret.to_owned(), id.clone()),
        (label.clone(), id.clone()),
    ]);
    let response = References::default()
        .response(map)
        .masked()
        .clean(|text| text.replace("/synthetic/archive", "[archive]"))
        .clean(polylogue_render::mask);
    let super::super::Response::Ok { data, .. } = response else {
        panic!()
    };
    assert_eq!(data[&label], id.to_string());
    assert_eq!(data[format!("{label} #2")], id.to_string());
    assert!(data.get(secret).is_none());
}
