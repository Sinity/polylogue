use super::*;
use serde_json::json;

const SECRET: &str = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-AAAAAAAA";

#[test]
fn decoded_nested_strings_are_masked_without_changing_json_types_or_empty_fields() {
    let encoded = format!(
        r#"{{"nested":["\u0073{}","quotes \" and slash \\ and newline\n"],"empty":"","array":[],"object":{{}},"unknown":null,"zero":0,"flag":false}}"#,
        &SECRET[1..]
    );
    let mut value: Value = serde_json::from_str(&encoded).unwrap();
    let ordinary = value["nested"][1].clone();
    mask_json(&mut value);
    assert_eq!(value["nested"][0], mask(SECRET));
    assert_eq!(value["nested"][1], ordinary);
    assert_eq!(value["empty"], "");
    assert_eq!(value["array"], json!([]));
    assert_eq!(value["object"], json!({}));
    assert_eq!(value["unknown"], Value::Null);
    assert_eq!(value["zero"], 0);
    assert_eq!(value["flag"], false);
    assert_eq!(
        serde_json::from_str::<Value>(&value.to_string()).unwrap(),
        value
    );
}

#[test]
fn masking_object_keys_preserves_colliding_values_and_ordinary_keys() {
    let other = SECRET.replace('A', "B");
    let label = mask(SECRET);
    assert_eq!(mask(&other), label);
    let mut value = json!({ SECRET: 1, other: 2, label.clone(): 3, "": "" });
    mask_json(&mut value);
    let fields = value.as_object().unwrap();
    assert_eq!(fields.len(), 4);
    assert_eq!(fields[&label], 3);
    assert_eq!(fields[""], "");
    assert!(fields.values().any(|v| v == 1));
    assert!(fields.values().any(|v| v == 2));
    assert!(!value.to_string().contains(SECRET));
    let once = value.clone();
    mask_json(&mut value);
    assert_eq!(value, once);
}
