//! Typed actor requests are bounded before they enter the single writer queue.
use polylogue_engine::owner::{Error as OwnerError, Handle, ImportRequest, Observation};
use polylogue_model::types::{ErrorKind, Failure};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::mpsc};
const METADATA_BYTES: usize = 65536;
fn failed(kind: ErrorKind, message: impl Into<String>) -> Value {
    json!({"outcome":"error","error":Failure::new(kind,message)})
}
fn actor_error(error: OwnerError) -> Value {
    if let OwnerError::Store(polylogue_store::Error::Authored(ref failure)) = error {
        return json!({"outcome":"error","error":failure});
    }
    let kind = match &error {
        OwnerError::Busy | OwnerError::Store(polylogue_store::Error::Cancelled) => {
            ErrorKind::Timeout
        }
        OwnerError::Store(polylogue_store::Error::ParserAdmission { .. }) => {
            ErrorKind::NotYetAvailable
        }
        OwnerError::Stopped => ErrorKind::NotRunning,
        OwnerError::Store(polylogue_store::Error::ClientRequestConflict { .. }) => {
            ErrorKind::Conflict
        }
        OwnerError::Store(polylogue_store::Error::ExcisionCheckRequired) => ErrorKind::Refused {
            reason: "excision-aware collector is not registered".into(),
        },
        OwnerError::InvalidRequest(_) | OwnerError::RequestTooLarge { .. } => {
            ErrorKind::InvalidQuery
        }
        OwnerError::UnknownInput => ErrorKind::NotFound,
        _ => ErrorKind::Io,
    };
    let mut failure = Failure::new(kind, error.to_string());
    failure.retryable = matches!(error, OwnerError::Busy | OwnerError::Stopped);
    json!({"outcome":"error","error":failure})
}
fn receive<T>(
    pending: Result<mpsc::Receiver<Result<T, OwnerError>>, OwnerError>,
) -> Result<T, Value> {
    pending
        .map_err(actor_error)?
        .recv()
        .map_err(|_| failed(ErrorKind::NotRunning, "writer stopped before replying"))?
        .map_err(actor_error)
}
fn text<'a>(request: &'a Value, key: &str) -> Result<&'a str, Value> {
    request
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            failed(
                ErrorKind::InvalidQuery,
                format!("{key} must be a nonempty string"),
            )
        })
}
fn valid_request_id(id: &str) -> bool {
    let b = id.as_bytes();
    b.len() == 36
        && [8, 13, 18, 23].into_iter().all(|i| b[i] == b'-')
        && b[14] == b'7'
        && matches!(b[19], b'8' | b'9' | b'a' | b'b')
        && b.iter()
            .enumerate()
            .all(|(i, v)| [8, 13, 18, 23].contains(&i) || matches!(v,b'0'..=b'9'|b'a'..=b'f'))
}
pub(crate) fn handle(request: &Value, owner: &Handle) -> Value {
    let action = || -> Result<Value, Value> {
        match text(request, "command")? {
            "note" => {
                let mut request = request.clone();
                let object = request.as_object_mut().ok_or_else(|| {
                    failed(ErrorKind::InvalidQuery, "note request must be an object")
                })?;
                object.remove("command");
                let deadline = object
                    .remove("deadline_ms")
                    .map(|v| {
                        v.as_u64().filter(|n| *n > 0 && *n <= 30000).ok_or_else(|| {
                            failed(ErrorKind::InvalidQuery, "deadline_ms must be1..30000")
                        })
                    })
                    .transpose()?
                    .unwrap_or(30000);
                let request: polylogue_model::authored::WriteNote = serde_json::from_value(request)
                    .map_err(|e| {
                        failed(
                            ErrorKind::InvalidQuery,
                            format!("unsupported authored request: {e}"),
                        )
                    })?;
                request
                    .validate()
                    .map_err(|e| json!({"outcome":"error","error":e}))?;
                let until = std::time::Instant::now() + std::time::Duration::from_millis(deadline);
                for id in [request.author.as_str(), request.machine.as_str()] {
                    crate::output_json::check_identity(id, until)
                        .map_err(|e| failed(ErrorKind::NotYetAvailable, e.to_string()))?;
                }
                for reference in request
                    .targets
                    .iter()
                    .flatten()
                    .map(|t| t.reference.as_str())
                    .chain(request.relations.iter().map(|r| r.other.as_str()))
                {
                    crate::output_json::check_reference(reference, until)
                        .map_err(|e| failed(ErrorKind::NotYetAvailable, e.to_string()))?;
                }
                let committed = receive(owner.try_note(request, until))?;
                Ok(crate::authored::write_reply(
                    committed.revision,
                    committed.recovery,
                    until,
                ))
            }
            "import" => {
                let allowed = [
                    "command", "id", "path", "format", "context", "machine", "host", "seen_at",
                ];
                let object = request
                    .as_object()
                    .ok_or_else(|| failed(ErrorKind::InvalidQuery, "request must be an object"))?;
                if object.keys().any(|k| !allowed.contains(&k.as_str())) {
                    return Err(failed(ErrorKind::InvalidQuery, "unknown import field"));
                }
                let format = match text(request, "format")? {
                    "codex-rollout" => polylogue_store::InputFormat::CodexRollout,
                    "codex-history-sqlite" => polylogue_store::InputFormat::CodexHistorySqlite,
                    "claude-code-jsonl" => polylogue_store::InputFormat::ClaudeCodeJsonl,
                    _ => {
                        return Err(failed(
                            ErrorKind::NotYetAvailable,
                            "source import format is not registered",
                        ));
                    }
                };
                let context = request
                    .get("context")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        failed(
                            ErrorKind::InvalidQuery,
                            "context must be an explicit string array",
                        )
                    })?
                    .iter()
                    .map(|v| {
                        v.as_str().map(str::to_owned).ok_or_else(|| {
                            failed(
                                ErrorKind::InvalidQuery,
                                "context must be an explicit string array",
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                format
                    .validate_context(&context.iter().map(String::as_str).collect::<Vec<_>>())
                    .map_err(|e| failed(ErrorKind::InvalidQuery, e.to_string()))?;
                if !valid_request_id(text(request, "id")?) {
                    return Err(failed(
                        ErrorKind::InvalidQuery,
                        "import id must be a canonical UUIDv7",
                    ));
                }
                let path = text(request, "path")?.to_owned();
                let request = ImportRequest {
                    format,
                    context,
                    path: PathBuf::from(&path),
                    metadata_allowance: METADATA_BYTES,
                    observation: Observation {
                        id: text(request, "id")?.to_owned(),
                        machine: text(request, "machine")?.to_owned(),
                        host: text(request, "host")?.to_owned(),
                        path,
                        seen_at: request.get("seen_at").and_then(Value::as_i64).ok_or_else(
                            || failed(ErrorKind::InvalidQuery, "seen_at must be an integer"),
                        )?,
                    },
                };
                let retained = receive(owner.try_import(request))?;
                let progress = receive(owner.try_progress(&retained.id))?;
                Ok(
                    json!({"outcome":"ok","retained":{"id":retained.id,"length":retained.length,"chain":retained.chain},"source_rows":progress.source_rows,"query_coverage":"incomplete"}),
                )
            }
            "progress" => {
                if request
                    .as_object()
                    .is_none_or(|o| o.keys().any(|k| !matches!(k.as_str(), "command" | "input")))
                {
                    return Err(failed(ErrorKind::InvalidQuery, "unknown progress field"));
                }
                let progress = receive(owner.try_progress(text(request, "input")?))?;
                let parser=progress.parser.map(|h|json!({"status":h.status,"step":h.token.step,"acquired_upto":h.acquired_upto,"inspected_upto":h.inspected_upto,"parsed_upto":h.parsed_upto,"finalization_pending":h.finalization_pending,
                    "records":{"framed":h.counts.framed,"used":h.counts.used,"ignored":h.counts.ignored,"unrecognized":h.counts.unrecognized,"undecodable":h.counts.undecodable}}));
                Ok(
                    json!({"outcome":"ok","input":progress.input.id,"length":progress.input.length,"parser":parser,"failure":progress.failure,"recovery_error":progress.recovery_error,"active":progress.active,"source_rows":progress.source_rows,"query_coverage":"incomplete"}),
                )
            }
            "retry" => {
                if request
                    .as_object()
                    .is_none_or(|o| o.keys().any(|k| !matches!(k.as_str(), "command" | "input")))
                {
                    return Err(failed(ErrorKind::InvalidQuery, "unknown retry field"));
                }
                receive(owner.try_retry(text(request, "input")?))?;
                Ok(json!({"outcome":"ok","query_coverage":"incomplete"}))
            }
            _ => Err(failed(
                ErrorKind::NotYetAvailable,
                "operation is not registered; only selected message: terminal reads are available, with incomplete coverage",
            )),
        }
    };
    match action() {
        Ok(reply) => reply,
        Err(reply) => reply,
    }
}
