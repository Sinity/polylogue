//! Optional count evidence from the Python version's nullable-usage and
//! optional-numeric cases, using the supported Google number fields.

use crate::support::{messages, parse};
use serde_json::{Value, json};

fn counted(origin: &str, supplied: Option<Value>) -> Value {
    let (path, document) = match origin {
        "gemini-cli" => {
            let mut tokens = json!({"output": 3});
            if let Some(value) = supplied {
                tokens["input"] = value;
            }
            (
                ".gemini/tmp/project/chats/session-numeric.json",
                json!({
                "sessionId": "numeric", "messages": [{"id": "answer", "type": "gemini",
                    "content": "Neutral authored answer", "tokens": tokens}]}),
            )
        }
        "ai-studio" => {
            let mut chunk =
                json!({"id": "answer", "role": "model", "text": "Neutral authored answer"});
            if let Some(value) = supplied {
                chunk["tokenCount"] = value;
            }
            (
                "polylogue/drive/ai-studio/numeric.json",
                json!({"chunkedPrompt": {"chunks": [chunk]}}),
            )
        }
        _ => unreachable!(),
    };
    let parsed = parse(origin, path, &serde_json::to_vec(&document).unwrap());
    let authored = messages(&parsed);
    assert_eq!(authored.len(), 1);
    assert_eq!(authored[0]["blocks"][0]["text"], "Neutral authored answer");
    assert_eq!(authored[0]["role"], "assistant");
    authored[0]["usage"][if origin == "gemini-cli" {
        "input"
    } else {
        "output"
    }]
    .clone()
}

#[test]
fn optional_google_counts_keep_exact_integer_values_including_zero() {
    for origin in ["gemini-cli", "ai-studio"] {
        for count in [0, 7, 9_007_199_254_740_993, u64::MAX] {
            assert_eq!(
                counted(origin, Some(json!(count))),
                json!(count),
                "{origin}"
            );
        }
    }
}

#[test]
fn invalid_or_missing_google_counts_do_not_become_zero_or_remove_prose() {
    for origin in ["gemini-cli", "ai-studio"] {
        for value in [
            None,
            Some(json!(null)),
            Some(json!(true)),
            Some(json!(false)),
            Some(json!(-1)),
            Some(json!(0.5)),
            Some(json!("7")),
            Some(json!("7.0")),
            Some(json!("7e0")),
            Some(json!("NaN")),
            Some(json!({})),
            Some(json!([])),
        ] {
            assert_eq!(counted(origin, value), Value::Null, "{origin}");
        }
    }
}
