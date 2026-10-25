use super::*;
use serde_json::json;

const SECRET: &str = "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-AAAAAAAA";

#[test]
fn crossing_quoted_credentials_mask_nested_values_and_colliding_keys() {
    let label = "--secret=[credential: CLI secret, masked] tail";
    let double = "--secret=\"unfinished --token=\"synthetic-later-value\" tail";
    let single = "--secret='unfinished --token='synthetic-later-value' tail";
    let raw =
        json!({"nested": [{double: 1, single: 2, "ordinary": 3, "value": double}], "empty": ""});
    let mut output = raw.clone();
    mask_json(&mut output);
    let nested = output["nested"][0].as_object().unwrap();
    assert_eq!(nested.len(), 4);
    assert_eq!(nested["ordinary"], 3);
    assert!(nested.values().any(|v| v == 1));
    assert!(nested.values().any(|v| v == 2));
    assert_eq!(nested["value"], label);
    assert_eq!(output["empty"], "");
    assert!(!output.to_string().contains("synthetic-later-value"));
    assert_eq!(raw["nested"][0]["value"], double);
    assert!(raw["nested"][0].get(single).is_some());
}

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

#[test]
fn tool_projection_aliases_reuse_redaction_and_independent_values_keep_their_types() {
    use crate::{Block, ToolAction, ToolOutcome};
    let tail = "private-command-----END PRIVATE KEY----- visible command";
    let result = "-----BEGIN PRIVATE KEY-----private-result";
    let mut blocks = vec![
        Block::Text {
            text: "-----BEGIN PRIVATE KEY-----private-prefix".into(),
        },
        Block::ToolCall {
            run_by_script: false,
            call_id: None,
            tool: Some("Bash".into()),
            action: ToolAction::Shell,
            command: Some(tail.into()),
            path: None,
            input: json!({"command":tail, "notes":"independent command notes", "empty":"", "zero":0, "flag":false, "nil":null}),
        },
        Block::ToolResult {
            call_id: None,
            outcome: ToolOutcome::Ok,
            output: Some(result.into()),
            data: Some(json!({"output":result, "notes":"independent result notes"})),
            attachments: vec![],
        },
        Block::Text {
            text: "private-result-tail-----END PRIVATE KEY----- visible end".into(),
        },
    ];
    let mut masker = MessageMasker::default();
    for block in &mut blocks {
        masker.block(block);
    }
    let structured = serde_json::to_value(&blocks).unwrap();
    assert_eq!(structured[1]["command"], structured[1]["input"]["command"]);
    assert_eq!(structured[2]["output"], structured[2]["data"]["output"]);
    assert_eq!(structured[1]["input"]["notes"], "independent command notes");
    assert_eq!(structured[2]["data"]["notes"], "independent result notes");
    assert_eq!(structured[1]["input"]["empty"], "");
    assert_eq!(structured[1]["input"]["zero"], 0);
    assert_eq!(structured[1]["input"]["flag"], false);
    assert!(structured[1]["input"]["nil"].is_null());
    let text = structured.to_string();
    assert!(!text.contains("private-command"));
    assert!(!text.contains("private-result"));
    assert!(!text.contains("private-prefix"));
    assert!(text.contains("visible command"));
    assert!(text.contains("visible end"));
}

#[test]
fn argument_object_values_are_complete_values_not_a_source_chronology() {
    use crate::{Block, ToolAction};
    let mut block = Block::ToolCall {
        run_by_script: false,
        call_id: None,
        tool: Some("tool".into()),
        action: ToolAction::Other,
        command: None,
        path: None,
        input: json!({"first":"-----BEGIN PRIVATE KEY-----independent private body", "second":"independent ordinary value", "empty":""}),
    };
    MessageMasker::default().block(&mut block);
    let value = serde_json::to_value(block).unwrap();
    assert!(!value.to_string().contains("independent private body"));
    assert_eq!(value["input"]["second"], "independent ordinary value");
    assert_eq!(value["input"]["empty"], "");
}
