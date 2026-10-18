//! Note commands on the CLI. The CLI adds the author; the daemon validates the note.
use polylogue_model::{
    ErrorKind, Failure,
    authored::{NoteKind, Relation, Target, WriteNote},
};
use serde_json::{Value, json};
use std::{
    env, fs,
    io::{IsTerminal, Read},
    time::{SystemTime, UNIX_EPOCH},
};
fn invalid(s: impl Into<String>) -> Failure {
    Failure::new(ErrorKind::InvalidQuery, s)
}
fn attribution(
    mcp: bool,
    claude: Option<String>,
    codex: Option<String>,
    harness: bool,
    interactive: bool,
) -> String {
    if let Some(id) = claude.filter(|s| !s.is_empty()) {
        return format!("claude-code:{id}");
    }
    if let Some(id) = codex.filter(|s| !s.is_empty()) {
        return format!("codex:{id}");
    }
    if mcp || harness {
        "agent".into()
    } else if interactive {
        "operator".into()
    } else {
        "script".into()
    }
}
pub(crate) fn author(mcp: bool) -> String {
    attribution(
        mcp,
        env::var("CLAUDE_CODE_SESSION_ID").ok(),
        env::var("CODEX_THREAD_ID").ok(),
        ["CLAUDECODE", "CODEX_CI", "CODEX_SANDBOX"]
            .iter()
            .any(|k| env::var_os(k).is_some()),
        std::io::stdin().is_terminal() && std::io::stdout().is_terminal(),
    )
}
pub(crate) fn request_id() -> Result<String, Failure> {
    let mut random = [0; 16];
    fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut random))
        .map_err(|e| Failure::new(ErrorKind::Io, e.to_string()))?;
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| invalid(e.to_string()))?
        .as_millis()
        .try_into()
        .map_err(|_| invalid("clock outside UUIDv7 range"))?;
    crate::import_cli::uuid_v7(ms, random)
}
pub(crate) fn uncertain_reply(request: &Value, error: std::io::Error) -> Value {
    let kind = match error.kind() {
        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused => {
            ErrorKind::NotRunning
        }
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => ErrorKind::Timeout,
        _ => ErrorKind::Io,
    };
    polylogue::authored::uncertain_reply(
        request,
        kind,
        "No reply from the daemon; the note may or may not have been saved. Retry with the same --request-id.",
    )
}
pub(crate) fn request(args: &[String]) -> Result<Option<Value>, Failure> {
    if args.first().is_some_and(|s| s.starts_with("note:"))
        && (args.len() == 2 && args[1] == "read"
            || args.len() == 4 && args[1..] == ["read", "--format", "json"])
    {
        return Ok(Some(json!({"command":"model-read","reference":args[0]})));
    }
    let Some(verb) = args
        .get(1)
        .filter(|s| matches!(s.as_str(), "note" | "revise" | "withdraw"))
    else {
        return Ok(None);
    };
    let reference = &args[0];
    let mut id = None;
    let mut kind = None;
    let mut value = None;
    let mut targets_file = None;
    let mut corrects = None;
    let mut reason = None;
    let mut i = 2;
    while i < args.len() {
        let arg = &args[i];
        if arg.starts_with("--") {
            i += 1;
            let v = args
                .get(i)
                .ok_or_else(|| invalid(format!("{arg} requires a value")))?
                .clone();
            let slot = match arg.as_str() {
                "--request-id" => &mut id,
                "--kind" => &mut kind,
                "--targets" => &mut targets_file,
                "--corrects" => &mut corrects,
                "--reason" => &mut reason,
                _ => {
                    return Err(Failure::new(
                        ErrorKind::NotYetAvailable,
                        format!("unknown note option {arg}"),
                    ));
                }
            };
            if slot.replace(v).is_some() {
                return Err(invalid("duplicate note option"));
            }
        } else if value.replace(arg.clone()).is_some() {
            return Err(invalid("give the note text once"));
        }
        i += 1;
    }
    let kind = kind
        .as_deref()
        .map(|k| match k {
            "note" => Ok(NoteKind::Note),
            "correction" => Ok(NoteKind::Correction),
            _ => Err(Failure::new(
                ErrorKind::NotYetAvailable,
                "--kind must be note or correction",
            )),
        })
        .transpose()?;
    let mut relations = vec![];
    let mut targets = None;
    if let Some(path) = targets_file {
        let file = fs::File::open(path).map_err(|e| Failure::new(ErrorKind::Io, e.to_string()))?;
        let mut bytes = vec![];
        file.take(65537)
            .read_to_end(&mut bytes)
            .map_err(|e| Failure::new(ErrorKind::Io, e.to_string()))?;
        if bytes.len() > 65536 {
            return Err(invalid("--targets file is larger than 65536 bytes"));
        }
        let parsed: Vec<Target> = serde_json::from_slice(&bytes).map_err(|e| {
            invalid(format!(
                "--targets must be a JSON array of {{\"ref\", \"seen_hash\"?}}: {e}"
            ))
        })?;
        targets = Some(parsed);
    }
    let (kind, value) = match verb.as_str() {
        "note" => {
            if reason.is_some() {
                return Err(invalid("--reason is only for withdraw"));
            }
            if targets
                .as_ref()
                .is_some_and(|t| t.len() != 1 || t[0].reference != *reference)
            {
                return Err(invalid("--targets must name the note's target"));
            }
            if targets.is_none() {
                targets = Some(vec![Target {
                    reference: reference.clone(),
                    seen_hash: None,
                }]);
            }
            (
                Some(kind.unwrap_or(NoteKind::Note)),
                value.ok_or_else(|| invalid("note requires final text"))?,
            )
        }
        "revise" => {
            if kind.is_some() || reason.is_some() {
                return Err(invalid(
                    "usage: note:ID revise TEXT [--targets FILE] [--request-id UUIDv7]",
                ));
            }
            relations.push(Relation {
                kind: "revises".into(),
                other: reference.clone(),
            });
            (
                None,
                value.ok_or_else(|| invalid("usage: note:ID revise TEXT"))?,
            )
        }
        "withdraw" => {
            if value.is_some() || kind.is_some() || targets.is_some() || corrects.is_some() {
                return Err(invalid(
                    "usage: note:ID withdraw --reason TEXT [--request-id UUIDv7]",
                ));
            }
            relations.push(Relation {
                kind: "withdraws".into(),
                other: reference.clone(),
            });
            (
                Some(NoteKind::Withdrawal),
                reason.ok_or_else(|| {
                    invalid("usage: note:ID withdraw --reason TEXT [--request-id UUIDv7]")
                })?,
            )
        }
        _ => unreachable!(),
    };
    if let Some(other) = corrects {
        relations.push(Relation {
            kind: "corrects".into(),
            other,
        });
    }
    let request = WriteNote {
        request_id: id.map(Ok).unwrap_or_else(request_id)?,
        author: author(false),
        machine: crate::host_identity::machine()?,
        created_at: None,
        kind,
        value,
        targets,
        relations,
    };
    request.validate()?;
    eprintln!("to retry: --request-id {}", request.request_id);
    let mut value = serde_json::to_value(request).map_err(|e| invalid(e.to_string()))?;
    value["command"] = json!("note");
    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn author_is_operator_script_agent_or_harness_session() {
        assert_eq!(attribution(false, None, None, false, true), "operator");
        assert_eq!(attribution(false, None, None, false, false), "script");
        assert_eq!(attribution(false, None, None, true, true), "agent");
        assert_eq!(attribution(true, None, None, false, true), "agent");
        assert_eq!(
            attribution(true, None, Some("synthetic".into()), false, true),
            "codex:synthetic"
        );
        assert_eq!(
            attribution(false, Some("synthetic".into()), None, false, true),
            "claude-code:synthetic"
        );
    }
}
