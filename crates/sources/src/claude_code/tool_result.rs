//! One source-owned result reduction used by body mapping and presence recipes.
//! Inline content is the presented output; named channels remain separate data.
use polylogue_model::RawJson as Value;
use polylogue_model::{ErrorReason, ToolOutcome, ToolOutput, ToolResult, UnknownReason};
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StatusKind {
    IsError,
    Denied,
    Interrupted,
    TimedOut,
}
/// Same supplied-status interpretation for normalized mapping and raw recipe.
pub(super) fn status(kind: StatusKind, value: &Value) -> Option<ToolOutcome> {
    Some(match kind {
        StatusKind::IsError => {
            if value.as_bool()? {
                ToolOutcome::Error { reason: None }
            } else {
                ToolOutcome::Ok
            }
        }
        StatusKind::Denied => ToolOutcome::Error {
            reason: Some(ErrorReason::Denied(
                value.as_str().filter(|s| !s.is_empty())?.into(),
            )),
        },
        StatusKind::Interrupted => {
            if value.as_bool()? {
                ToolOutcome::Error {
                    reason: Some(ErrorReason::Interrupted),
                }
            } else {
                ToolOutcome::Ok
            }
        }
        StatusKind::TimedOut => {
            value.as_u64()?;
            ToolOutcome::Error {
                reason: Some(ErrorReason::TimedOut),
            }
        }
    })
}
#[derive(Clone, Debug)]
pub(super) struct Decision {
    pub result: ToolResult,
    pub outcome: Option<Vec<String>>,
    pub reason: Option<Vec<String>>,
    pub exit: Option<Vec<String>>,
    pub data: Vec<(String, Vec<String>)>,
    pub unresolved: Vec<Vec<String>>,
}
fn p(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| (*s).into()).collect()
}
/// Root-level metadata is attributable only when this record has one result.
fn sole_result_sidecar(record: &Value) -> Option<&Value> {
    let count = record
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)?
        .iter()
        .filter(|v| v.get("type").and_then(Value::as_str) == Some("tool_result"))
        .count();
    if count == 1 {
        record.get("toolUseResult")
    } else {
        None
    }
}
pub(super) fn metadata(record: &Value) -> Option<&Value> {
    sole_result_sidecar(record).filter(|v| v.is_object())
}
// These provider-owned members are opaque execution metadata, not normalized
// status/output, file authority, billing, or additional tool executions.
fn opaque_member(key: &str) -> bool {
    matches!(
        key,
        "canReadOutputFile"
            | "content"
            | "description"
            | "file"
            | "filePath"
            | "gitOperation"
            | "matches"
            | "message"
            | "newString"
            | "oldString"
            | "outputFile"
            | "pin"
            | "prompt"
            | "query"
            | "replaceAll"
            | "resolvedModel"
            | "resumedAgentId"
            | "staleReadFileStateHint"
            | "staleRecovered"
            | "success"
            | "total_deferred_tools"
            | "type"
            | "userModified"
    )
}
/// A supported reference array stays structured, including its text/order.
pub(super) fn reference_content(value: &Value) -> bool {
    let Some(items) = value.as_array() else {
        return false;
    };
    items
        .iter()
        .any(|v| v.get("type").and_then(Value::as_str) == Some("tool_reference"))
        && items.iter().all(|v| {
            let Some(m) = v.as_object() else {
                return false;
            };
            match v.get("type").and_then(Value::as_str) {
                Some("text") => m.len() == 2 && v.get("text").is_some_and(Value::is_string),
                Some("tool_reference") => {
                    m.len() == 2
                        && v.get("tool_name")
                            .is_some_and(|n| n.is_string() && !n.as_str().unwrap_or("").is_empty())
                }
                _ => false,
            }
        })
}
pub(super) fn decide(record: &Value, index: usize, item: &Value) -> Decision {
    let base = vec!["message".into(), "content".into(), index.to_string()];
    let at = |name: &str| {
        let mut path = base.clone();
        path.push(name.into());
        path
    };
    let meta = metadata(record);
    let mut outcome_source = item
        .get("is_error")
        .and_then(Value::as_bool)
        .map(|_| at("is_error"));
    let mut outcome = item
        .get("is_error")
        .and_then(|v| status(StatusKind::IsError, v))
        .unwrap_or(ToolOutcome::Ok);
    let mut reason_source = None;
    let mut status_unreadable = item.get("is_error").is_some_and(|v| v.as_bool().is_none());
    let mut unresolved = Vec::new();
    if status_unreadable {
        unresolved.push(at("is_error"));
    }
    let denial = item
        .get("toolDenialKind")
        .map(|value| (value, at("toolDenialKind")))
        .or_else(|| {
            record
                .get("toolDenialKind")
                .map(|value| (value, p(&["toolDenialKind"])))
        });
    let has_denial = denial.is_some();
    if let Some((value, path)) = denial {
        if let Some(resolved) = status(StatusKind::Denied, value) {
            outcome = resolved;
            outcome_source = Some(path.clone());
            reason_source = Some(path);
        } else {
            status_unreadable = true;
            unresolved.push(path);
        }
    } else if let Some(value) = meta.and_then(|m| m.get("interrupted")) {
        if let Some(interrupted) = value.as_bool() {
            if interrupted {
                let path = p(&["toolUseResult", "interrupted"]);
                outcome = status(StatusKind::Interrupted, value).unwrap_or(ToolOutcome::Unknown {
                    reason: UnknownReason::StatusFieldUnreadable,
                });
                outcome_source = Some(path.clone());
                reason_source = Some(path);
            }
        } else {
            status_unreadable = true;
            unresolved.push(p(&["toolUseResult", "interrupted"]));
        }
    }
    if !has_denial && reason_source.is_none() {
        if let Some(timeout) = meta.and_then(|m| m.get("timedOutAfterMs")) {
            if timeout.as_u64().is_some() {
                let path = p(&["toolUseResult", "timedOutAfterMs"]);
                outcome = status(StatusKind::TimedOut, timeout).unwrap_or(ToolOutcome::Unknown {
                    reason: UnknownReason::StatusFieldUnreadable,
                });
                outcome_source = Some(path.clone());
                reason_source = Some(path);
            } else {
                status_unreadable = true;
                unresolved.push(p(&["toolUseResult", "timedOutAfterMs"]));
            }
        }
    }
    // Launch acknowledgements are retained, but this mapper does not claim to
    // have joined their eventual completion notification.
    if item.get("backgroundTaskId").is_some()
        || meta.is_some_and(|m| {
            m.get("backgroundTaskId").is_some()
                || m.get("isAsync").and_then(Value::as_bool) == Some(true)
                || m.get("status").and_then(Value::as_str) == Some("async_launched")
        })
    {
        outcome = ToolOutcome::Unknown {
            reason: UnknownReason::BackgroundNeverReported,
        };
        outcome_source = None;
        reason_source = None;
    }
    let mut exit = None;
    let mut exit_source = None;
    if let Some(meta) = meta {
        for name in ["exitCode", "exit_code"] {
            if let Some(value) = meta.get(name) {
                if let Some(value) = value.as_i64() {
                    exit = Some(value);
                    exit_source = Some(p(&["toolUseResult", name]));
                } else {
                    unresolved.push(p(&["toolUseResult", name]));
                }
                break;
            }
        }
    }
    let references = item.get("content").filter(|v| reference_content(v));
    let output = item
        .get("content")
        .filter(|_| references.is_none())
        .and_then(super::text_of);
    if exit.is_none() {
        exit = output.as_deref().and_then(super::exit_code_in);
    }
    // Supplied boolean flags are execution metadata only. They do not infer
    // image content, outcome, or output absence. Missing flags stay absent.
    let mut data = std::collections::BTreeMap::new();
    let mut data_paths = Vec::new();
    if let Some(meta) = meta.and_then(Value::as_object) {
        for (key, value) in meta {
            if matches!(
                key.as_str(),
                "stdout"
                    | "stderr"
                    | "structuredPatch"
                    | "originalFile"
                    | "agentId"
                    | "status"
                    | "totalTokens"
                    | "totalToolUseCount"
                    | "totalDurationMs"
                    | "returnCodeInterpretation"
            ) || opaque_member(key.as_str())
                || (matches!(key.as_str(), "isImage" | "noOutputExpected")
                    && value.as_bool().is_some())
                || (key == &polylogue_model::NativeId::unicode("interrupted")
                    && value.as_bool() == Some(false))
            {
                data.insert(key.clone(), value.opaque());
                data_paths.push((key.as_str().to_owned(), p(&["toolUseResult", key.as_str()])));
            } else if !matches!(
                key.as_str(),
                "persistedOutputPath"
                    | "interrupted"
                    | "timedOutAfterMs"
                    | "exitCode"
                    | "exit_code"
                    | "backgroundTaskId"
                    | "isAsync"
            ) {
                // Retain unknown named metadata without pretending its semantic
                // reduction is complete. No opaque wrapper completeness bonus.
                data.insert(key.clone(), value.opaque());
                unresolved.push(p(&["toolUseResult", key.as_str()]));
            }
        }
    }
    if let Some(content) = references {
        data.insert(
            polylogue_model::NativeId::unicode("content_items"),
            content.opaque(),
        );
        data_paths.push(("content_items".into(), at("content")));
    }
    // A string sidecar is a distinct supplied carrier, never presented output.
    let string_sidecar = sole_result_sidecar(record).filter(|v| v.is_string());
    if let Some(value) = string_sidecar {
        data.insert(
            polylogue_model::NativeId::unicode("toolUseResult"),
            value.opaque(),
        );
        data_paths.push(("toolUseResult".into(), p(&["toolUseResult"])));
    }
    if record.get("toolUseResult").is_some() && meta.is_none() && string_sidecar.is_none() {
        unresolved.push(p(&["toolUseResult"]));
    }
    if status_unreadable && matches!(outcome, ToolOutcome::Ok) {
        outcome = ToolOutcome::Unknown {
            reason: UnknownReason::StatusFieldUnreadable,
        };
        outcome_source = None;
    }
    Decision {
        result: ToolResult {
            call_id: item.get("tool_use_id").and_then(Value::native).cloned(),
            outcome,
            exit_code: exit,
            duration_ms: None,
            output: output.map(ToolOutput::Text),
            data: (!data.is_empty()).then_some(Value::Object(data)),
        },
        outcome: outcome_source,
        reason: reason_source,
        exit: exit_source,
        data: data_paths,
        unresolved,
    }
}
