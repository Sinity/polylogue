//! Shared helpers for tests that drive one origin's parser on bytes written
//! inline: the knowledge ported from the Python predecessor's tests.
#![allow(dead_code)]

use polylogue_sources::origin::by_name;
use serde_json::Value;
use std::path::Path;

/// Parses `input` as `origin` would at `path` (relative to the home
/// directory), and returns the step's output and accounting as JSON.
pub fn parse(origin: &str, path: &str, input: &[u8]) -> Value {
    let spec = by_name(origin).unwrap_or_else(|| panic!("unknown origin {origin}"));
    let path = Path::new(path);
    let parser =
        (spec.open)(path).unwrap_or_else(|| panic!("{origin} does not open {}", path.display()));
    let (step, _carry) = parser.step(None, input, 0).expect("parse step");
    assert_eq!(
        step.consumed,
        input.len() as u64,
        "parser consumed the whole input"
    );
    serde_json::json!({"output": step.output, "accounting": step.accounting})
}

/// Joins JSON records into JSONL bytes.
pub fn jsonl(records: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in records {
        out.extend(serde_json::to_vec(r).unwrap());
        out.push(b'\n');
    }
    out
}

/// The parsed messages, in output order.
pub fn messages(parsed: &Value) -> Vec<&Value> {
    parsed["output"]["messages"]
        .as_array()
        .map(|a| a.iter().map(|m| &m["message"]).collect())
        .unwrap_or_default()
}

/// The message whose ID ends with `suffix` (IDs are `<origin>:<native>`).
pub fn message<'a>(parsed: &'a Value, suffix: &str) -> &'a Value {
    messages(parsed)
        .into_iter()
        .find(|m| m["id"].as_str().is_some_and(|id| id.ends_with(suffix)))
        .unwrap_or_else(|| panic!("no message ending {suffix} in {parsed:#}"))
}

/// The parsed sessions.
pub fn sessions(parsed: &Value) -> Vec<&Value> {
    parsed["output"]["sessions"]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default()
}

/// Events of one kind.
pub fn events<'a>(parsed: &'a Value, kind: &str) -> Vec<&'a Value> {
    parsed["output"]["events"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|e| e.get("event").unwrap_or(e))
                .filter(|e| e["kind"].as_str() == Some(kind))
                .collect()
        })
        .unwrap_or_default()
}

/// All blocks of all messages.
pub fn blocks(parsed: &Value) -> Vec<&Value> {
    messages(parsed)
        .into_iter()
        .flat_map(|m| m["blocks"].as_array().into_iter().flatten())
        .collect()
}
