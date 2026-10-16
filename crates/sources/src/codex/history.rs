//! Pure mapping for the pinned native CommandExecution conversion. SQL and
//! snapshot acquisition belong to store; unsupported item schemas stay visible.
use polylogue_model::{
    SessionId,
    tool_item::{Command, HistoryRow, Operation, ToolOrigin},
};
use serde_json::Value;
pub const PARSER_VERSION: u32 = 1;
pub const SCHEMA: &str = "codex-rust-v0.162.1/thread-history-migrations-1-through-8";

pub fn command(
    item: &Value,
    session: SessionId,
    native_session: &str,
    turn: Option<String>,
    origin: ToolOrigin,
    ui: bool,
) -> Result<Operation, String> {
    let get = |core: &str, display: &str| item.get(if ui { display } else { core });
    let id = item
        .get("id")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .ok_or("CommandExecution has no invocation ID")?;
    let status = get("status", "status")
        .and_then(Value::as_str)
        .ok_or("CommandExecution has no lifecycle status")?;
    let (class, executed) = match get("source", "source").and_then(Value::as_str) {
        Some("agent") => ("agent", true),
        Some("user_shell" | "userShell") => ("non_agent", true),
        Some(
            "unified_exec_startup"
            | "unifiedExecStartup"
            | "unified_exec_interaction"
            | "unifiedExecInteraction",
        ) => ("unknown", false),
        _ => ("unknown", false),
    };
    let terminal = matches!(status, "completed" | "failed");
    if !matches!(
        status,
        "in_progress" | "inProgress" | "completed" | "failed" | "declined"
    ) {
        return Err("unsupported CommandExecution lifecycle".into());
    }
    let output = get("aggregated_output", "aggregatedOutput")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or("CommandExecution output is not text")
        })
        .transpose()?;
    let exit = get("exit_code", "exitCode")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_i64()
                .ok_or("CommandExecution exit code is not an integer")
        })
        .transpose()?;
    let command = if ui {
        item.get("command")
            .and_then(Value::as_str)
            .map(|s| Command::Text(s.into()))
    } else {
        item.get("command")
            .and_then(Value::as_array)
            .and_then(|a| a.iter().map(Value::as_str).collect::<Option<Vec<_>>>())
            .map(|a| Command::Argv(a.into_iter().map(str::to_owned).collect()))
    }
    .ok_or("CommandExecution command is missing or has an unsupported shape")?;
    let result = executed && terminal;
    Ok(Operation {
        kind: "CommandExecution".into(),
        tool: None,
        native_session: native_session.into(),
        direct_anchors: vec![],
        session,
        turn: turn.clone(),
        invocation: id.into(),
        native: turn.is_some(),
        provisional_anchor: None,
        agent_class: class.into(),
        correspondence: if turn.is_some() {
            "established"
        } else {
            "unresolved"
        }
        .into(),
        call_present: true,
        command: Some(command),
        result_present: result,
        output: if result { output } else { None },
        outcome: result.then(|| {
            if exit.is_some_and(|c| c != 0) || status == "failed" {
                "error"
            } else {
                "ok"
            }
            .into()
        }),
        exit_code: if result { exit } else { None },
        duration_ms: if result {
            get("duration_ms", "durationMs")
                .and_then(Value::as_u64)
                .or_else(|| {
                    let d = item.get("duration")?;
                    d.get("secs")?
                        .as_u64()?
                        .checked_mul(1000)?
                        .checked_add(d.get("nanos")?.as_u64()? / 1_000_000)
                })
        } else {
            None
        },
        message: None,
        wrapper: None,
        wrapper_overlaps: vec![],
        origins: vec![origin],
    })
}
// Every consulted physical-row fact contributes to immutable version identity.
// Keep the item JSON hash separate: equal item JSON alone does not imply equal
// qualification when its started/completed timestamps differ.
fn row_evidence(row: &HistoryRow) -> Result<[u8; 32], String> {
    let bytes = serde_json::to_vec(row).map_err(|e| e.to_string())?;
    let canonical =
        polylogue_model::identity::canonical_record(&bytes, &[]).map_err(|e| e.to_string())?;
    let mut hash = blake3::Hasher::new();
    hash.update(b"polylogue.codex-history-row.v1\0");
    hash.update(&PARSER_VERSION.to_be_bytes());
    for part in [
        polylogue_model::tool_item::CODEX_REVISION.as_bytes(),
        polylogue_model::tool_item::CODEX_CONSTRUCTOR.as_bytes(),
        SCHEMA.as_bytes(),
        canonical.as_slice(),
    ] {
        hash.update(&(part.len() as u64).to_be_bytes());
        hash.update(part);
    }
    Ok(*hash.finalize().as_bytes())
}
pub fn map(row: &HistoryRow) -> Result<Option<Operation>, String> {
    let item: Value =
        serde_json::from_str(&row.item_json).map_err(|e| format!("history item_json: {e}"))?;
    if item.get("type").and_then(Value::as_str) != Some("commandExecution") {
        return Ok(None);
    }
    if row.item_type != "commandExecution"
        || item.get("id").and_then(Value::as_str) != Some(row.item_id.as_str())
        || row.thread_id.is_empty()
        || row.turn_id.is_empty()
    {
        return Err("history row and item coordinates disagree".into());
    }
    let session =
        SessionId::from_stored(&format!("codex:{}", row.thread_id)).map_err(|e| e.to_string())?;
    let origin = ToolOrigin::History {
        schema: SCHEMA.into(),
        table: "thread_items".into(),
        thread: row.thread_id.clone(),
        turn: row.turn_id.clone(),
        item: row.item_id.clone(),
        raw_hash: *blake3::hash(
            &polylogue_model::identity::canonical_record(row.item_json.as_bytes(), &[])
                .map_err(|e| e.to_string())?,
        )
        .as_bytes(),
        row_hash: row_evidence(row)?,
    };
    let mut operation = command(
        &item,
        session,
        &row.thread_id,
        Some(row.turn_id.clone()),
        origin,
        true,
    )?;
    if row
        .item_id
        .strip_prefix("item-")
        .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
    {
        operation.correspondence = "unresolved".into();
    }
    if operation.agent_class == "agent"
        && row.started_at_ms.is_none()
        && item.get("exitCode").is_none_or(Value::is_null)
        && item.get("processId").is_none_or(Value::is_null)
    {
        operation.agent_class = "unknown".into();
        operation.result_present = false;
        operation.output = None;
        operation.outcome = None;
        operation.exit_code = None;
        operation.duration_ms = None;
    }
    Ok(Some(operation))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_qualification_facts_change_version_evidence_for_identical_item_json() {
        let mut row = HistoryRow { thread_id:"S".into(),turn_id:"T".into(),item_id:"C".into(),rollout_ordinal:1,updated_at_ordinal:1,created_at_ms:1,started_at_ms:None,completed_at_ms:None,item_type:"commandExecution".into(),item_json:serde_json::json!({"type":"commandExecution","id":"C","source":"agent","command":"same","status":"completed"}).to_string() };
        let unknown = map(&row).unwrap().unwrap();
        row.started_at_ms = Some(1);
        let proved = map(&row).unwrap().unwrap();
        assert_eq!(unknown.id(), proved.id());
        assert_eq!(unknown.agent_class, "unknown");
        assert!(!unknown.result_present);
        assert_eq!(proved.agent_class, "agent");
        assert!(proved.result_present);
        match (&unknown.origins[0], &proved.origins[0]) {
            (
                ToolOrigin::History {
                    raw_hash: a,
                    row_hash: av,
                    ..
                },
                ToolOrigin::History {
                    raw_hash: b,
                    row_hash: bv,
                    ..
                },
            ) => {
                assert_eq!(a, b);
                assert_ne!(av, bv);
            }
            _ => panic!("history origins"),
        }
    }
    #[test]
    fn empty_core_and_omitted_ui_are_distinct_present_results() {
        let session = SessionId::from_stored("codex:S").unwrap();
        let origin = ToolOrigin::Rollout {
            tag: "item".into(),
            start: 0,
            end: 1,
            raw_hash: [1; 32],
        };
        let core=command(&serde_json::json!({"id":"C","source":"agent","command":[],"status":"failed","aggregated_output":"","exit_code":1}),session.clone(),"S",Some("T".into()),origin.clone(),false).unwrap();
        let ui = command(
            &serde_json::json!({"id":"C","source":"agent","command":"","status":"failed","exitCode":1}),
            session,
            "S",
            Some("T".into()),
            origin,
            true,
        )
        .unwrap();
        assert_eq!(core.id(), ui.id());
        assert_eq!(core.tool, None);
        assert_eq!(ui.tool, None);
        assert!(core.result_present && ui.result_present);
        assert_eq!(core.output, Some("".into()));
        assert_eq!(ui.output, None);
        assert_eq!(core.command, Some(Command::Argv(vec![])));
        assert_eq!(ui.command, Some(Command::Text(String::new())));
    }
}
