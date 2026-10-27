//! Trajectory SQLite rows captured by Store's ordinary row-dump reader.
//! Scan metadata first, then read ordered steps one at a time.
use super::*;
use crate::export::{ExportParser, ExportState, Input, ItemSink, ScanError, ScanStats};
use base64::Engine as _;
use polylogue_model::{Event, Link, LinkKind, SessionId, ToolOutcome, UnknownReason};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

pub const LOCATIONS: &[&str] = &[
    ".gemini/antigravity/**/*.db",
    ".gemini/antigravity/**/*.sqlite",
    ".gemini/antigravity/**/*.sqlite3",
];
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    path.file_name().is_some_and(|name| {
        let name = name.to_string_lossy();
        [".db.jsonl", ".sqlite.jsonl", ".sqlite3.jsonl"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
    }) && String::from_utf8_lossy(head).contains("trajectory_meta")
}
pub fn open(path: &Path) -> Option<Parse> {
    Some(Parse::Export(Box::new(Database {
        file: Path::new(path.file_name()?.to_str()?.strip_suffix(".jsonl")?)
            .file_stem()?
            .to_string_lossy()
            .into_owned(),
        ..Default::default()
    })))
}
#[derive(Default)]
pub struct Database {
    file: String,
    unnamed: String,
    meta: BTreeMap<String, Json>,
    aliases: BTreeMap<String, String>,
    valid: bool,
    has_steps: bool,
    superseded: u64,
    last_call: Option<(SessionId, String, String)>,
    calls: usize,
    call_session: Option<SessionId>,
    parents: BTreeMap<String, BTreeSet<String>>,
}
fn lines(
    reader: &mut dyn Input,
    mut visit: impl FnMut(Range<u64>, &Json, &[u8]) -> Result<(), String>,
) -> Result<ScanStats, ScanError> {
    let mut stats = ScanStats::default();
    let mut offset = 0;
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        let n = reader
            .read_until(b'\n', &mut bytes)
            .map_err(|e| ScanError::Input(serde_json::Error::io(e)))?;
        if n == 0 {
            return Ok(stats);
        }
        if bytes.last() != Some(&b'\n') {
            return Err(ScanError::Callback(
                "incomplete Antigravity row dump line".into(),
            ));
        }
        let value = serde_json::from_slice(&bytes).map_err(ScanError::Malformed)?;
        visit(offset..offset + n as u64, &value, &bytes).map_err(ScanError::Callback)?;
        offset += n as u64;
        stats.elements += 1;
        stats.peak_element_bytes = stats.peak_element_bytes.max(n);
    }
}
fn key(v: &Json) -> Option<String> {
    Some(format!("{}:{}", v["t"].as_str()?, v.get("k")?))
}
fn tokens(sql: &str, required: &[&str]) -> bool {
    let names: Vec<_> = sql
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .map(str::to_ascii_lowercase)
        .collect();
    required.iter().all(|name| names.iter().any(|n| n == name))
}
impl ExportParser for Database {
    fn scan(
        &mut self,
        _: &str,
        reader: &mut dyn Input,
        items: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        self.parents.clear();
        self.call_session = None;
        self.meta.clear();
        self.aliases.clear();
        self.valid = false;
        self.has_steps = false;
        self.superseded = 0;
        self.last_call = None;
        self.calls = 0;
        let mut seen = std::collections::HashSet::new();
        let mut references = BTreeMap::new();
        let mut summaries = BTreeMap::new();
        let stats = lines(reader, |_, v, _| {
            if let Some(schema) = v.get("schema") {
                self.valid = schema["trajectory_meta"]
                    .as_str()
                    .is_some_and(|s| tokens(s, &["trajectory_id", "cascade_id"]))
                    && schema["steps"].as_str().is_some_and(|s| {
                        tokens(s, &["idx", "step_type", "step_format", "step_payload"])
                    });
            }
            if v["t"] == "steps" && v["gone"] != true {
                self.has_steps = true;
            }
            if v["gone"] != true
                && v["r"].is_object()
                && let Some(key) = key(v)
            {
                if !seen.insert(key.clone()) {
                    self.superseded += 1;
                }
                if v["t"] == "trajectory_meta" {
                    self.meta.insert(key.clone(), v["r"].clone());
                }
                if v["t"] == "parent_references" {
                    references.insert(key.clone(), v["r"].clone());
                }
                if v["t"] == "conversation_summaries" {
                    summaries.insert(key, v["r"].clone());
                }
            }
            Ok(())
        })?;
        let anonymous = self
            .meta
            .values()
            .filter(|r| {
                str_of(r, "trajectory_id")
                    .or_else(|| str_of(r, "cascade_id"))
                    .is_none()
            })
            .count();
        if anonymous > 1 {
            return Err(ScanError::Callback(
                "multiple Antigravity trajectories have no stable identity".into(),
            ));
        }
        let reserved: BTreeSet<String> = self
            .meta
            .values()
            .flat_map(|row| [str_of(row, "trajectory_id"), str_of(row, "cascade_id")])
            .flatten()
            .chain(summaries.values().filter_map(summary_id))
            .map(str::to_owned)
            .collect();
        self.unnamed = self.file.clone();
        if reserved.contains(&self.unnamed) {
            self.unnamed = format!("{}:trajectory", self.file);
            let mut number = 0_u64;
            while reserved.contains(&self.unnamed) {
                number += 1;
                self.unnamed = format!("{}:trajectory:{number}", self.file);
            }
        }
        for row in self.meta.values() {
            let native = str_of(row, "trajectory_id")
                .or_else(|| str_of(row, "cascade_id"))
                .unwrap_or(&self.unnamed)
                .to_owned();
            for alias in [str_of(row, "trajectory_id"), str_of(row, "cascade_id")]
                .into_iter()
                .flatten()
            {
                if self
                    .aliases
                    .insert(alias.into(), native.clone())
                    .is_some_and(|old| old != native)
                {
                    return Err(ScanError::Callback(
                        "conflicting Antigravity trajectory identities".into(),
                    ));
                }
            }
        }
        for row in references.values() {
            if let Some(child) = self.named_session(row)
                && let Some(parent) = parent_id(row)
            {
                let parent = self
                    .aliases
                    .get(parent)
                    .map_or(parent, String::as_str)
                    .to_owned();
                self.parents.entry(child).or_default().insert(parent);
            }
        }
        reader
            .rewind()
            .map_err(|e| ScanError::Input(serde_json::Error::io(e)))?;
        lines(reader, |range, v, raw| {
            let table = v["t"].as_str().unwrap_or("");
            let mut order = vec![match table {
                "trajectory_meta" => 1,
                "conversation_summaries" => 2,
                "parent_references" => 3,
                "steps" => 4,
                _ => 0,
            }];
            if table == "steps" {
                let row = &v["r"];
                let native = self.step_session(row).unwrap_or_default();
                order.extend_from_slice(&(native.len() as u64).to_be_bytes());
                order.extend_from_slice(native.as_bytes());
                order.extend_from_slice(
                    &(row["idx"].as_i64().unwrap_or(i64::MAX) as u64 ^ (1u64 << 63)).to_be_bytes(),
                );
            }
            items.push(
                range,
                order,
                if v["gone"] == true { None } else { key(v) }.as_deref(),
                raw,
            )
        })?;
        Ok(stats)
    }
    fn begin(&mut self, length: u64, _: Option<&[u8]>) -> Result<Option<crate::Step>, String> {
        if !self.valid {
            return Err("Antigravity SQLite lacks the declared trajectory schema".into());
        }
        let mut records = Records::default();
        for _ in 0..self.superseded {
            records.record(
                "row",
                &RecordOutcome::skipped("superseded row remains stored"),
            );
        }
        let mut output = Output::default();
        if self.meta.is_empty() {
            let sid = session_id(ORIGIN, &self.unnamed).ok_or("invalid Antigravity file name")?;
            output
                .sessions
                .push(Session::new(sid.clone(), SessionKind::Main));
            if !self.has_steps {
                event(
                    &mut output,
                    Some(sid),
                    "antigravity_trajectory_empty",
                    Json::Null,
                );
            }
        }
        Ok(Some(crate::Step {
            output,
            counts: records.finish().map_err(|e| e.to_string())?,
            start: 0,
            consumed: length,
            repeats_whole_conversations: true,
        }))
    }
    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        state: &mut dyn ExportState,
    ) -> Result<(), String> {
        let v: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if v.get("schema").is_some() {
            sink.records.record("schema", &RecordOutcome::Used);
            return Ok(());
        }
        let Some(table) = v["t"].as_str() else {
            sink.records.record(
                "row",
                &RecordOutcome::unreadable("row dump line has no table or schema"),
            );
            return Ok(());
        };
        if v.get("k").is_none() || (v["gone"] != true && !v["r"].is_object()) {
            sink.records.record(
                table,
                &RecordOutcome::unreadable("row dump line needs a key and row object"),
            );
            return Ok(());
        }
        if v["gone"] == true {
            sink.records.record(
                table,
                &RecordOutcome::skipped("deleted tool row remains archived"),
            );
            return Ok(());
        }
        let row = &v["r"];
        let sid = self.named_session(row).and_then(|n| session_id(ORIGIN, &n));
        match table {
            "trajectory_meta" => {
                let native = str_of(row, "trajectory_id")
                    .or_else(|| str_of(row, "cascade_id"))
                    .unwrap_or(&self.unnamed);
                let sid = session_id(ORIGIN, native).ok_or("invalid trajectory identity")?;
                sink.out
                    .sessions
                    .push(Session::new(sid.clone(), SessionKind::Main));
                event(sink.out, Some(sid), "antigravity_trajectory", row.clone());
            }
            "conversation_summaries" => {
                let native = summary_id(row);
                let matched = native.and_then(|id| self.aliases.get(id));
                let sid = matched
                    .map(String::as_str)
                    .or(native)
                    .and_then(|native| session_id(ORIGIN, native));
                if let Some(sid) = &sid {
                    let mut session = Session::new(sid.clone(), SessionKind::Main);
                    if let Some(title) = ["title", "name", "summary"]
                        .iter()
                        .find_map(|key| str_of(row, key))
                    {
                        session.offer_title(title, TitleSource::Summary);
                    }
                    sink.out.sessions.push(session);
                }
                let at = [
                    "last_modified_time",
                    "lastModifiedTime",
                    "updated_at",
                    "updatedAt",
                    "modified_at",
                ]
                .iter()
                .find_map(|key| row.get(*key).and_then(crate::time::rfc3339));
                event_at(
                    sink.out,
                    sid,
                    if matched.is_some() {
                        "antigravity_conversation_summary"
                    } else {
                        "antigravity_unmatched_summary"
                    },
                    at,
                    row.clone(),
                );
            }
            "parent_references" => {
                if let Some(child) = &sid
                    && let Some(parents) = self
                        .parents
                        .get(child.native().as_str())
                        .filter(|p| p.len() == 1)
                    && let Some(parent) = parents.first().and_then(|p| session_id(ORIGIN, p))
                {
                    sink.out.links.push(Link {
                        kind: LinkKind::Parent,
                        parent,
                        child: child.clone(),
                        at: None,
                        call_id: None,
                        evidence: "parent_references".into(),
                    });
                }
                event(sink.out, sid, "antigravity_parent_reference", row.clone());
            }
            "steps" => {
                let Some(sid) = self
                    .step_session(row)
                    .and_then(|native| session_id(ORIGIN, &native))
                else {
                    self.last_call = None;
                    self.calls = 0;
                    event(sink.out, None, "antigravity_unattributed_step", row.clone());
                    sink.records.record(
                        table,
                        &RecordOutcome::unreadable("step has no unambiguous trajectory"),
                    );
                    return Ok(());
                };
                if self.call_session.as_ref() != Some(&sid) {
                    self.calls = 0;
                    self.last_call = None;
                }
                self.call_session = Some(sid.clone());
                let payload = step_payload(row);
                let kind = str_of(row, "step_type")
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase()
                    .replace('-', "_");
                let format = str_of(row, "step_format")
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase()
                    .replace('-', "_");
                let last = self.last_call.take();
                let supported =
                    matches!(format.as_str(), "v1" | "json" | "json_v1" | "trajectory_v1")
                        && matches!(
                            kind.as_str(),
                            "message"
                                | "user"
                                | "human"
                                | "prompt"
                                | "assistant"
                                | "planner"
                                | "planning"
                                | "plan"
                                | "plan_step"
                                | "terminal"
                                | "terminal_command"
                                | "command"
                                | "tool"
                                | "tool_call"
                                | "tool_use"
                                | "tool_result"
                                | "tool_output"
                                | "command_result"
                                | "file_edit"
                                | "edit"
                        );
                if !supported || !payload.as_ref().is_some_and(Json::is_object) {
                    self.calls = 0;
                    event(
                        sink.out,
                        Some(sid),
                        "antigravity_unsupported_step",
                        row.clone(),
                    );
                    sink.records.record(
                        table,
                        &RecordOutcome::unreadable(
                            "unsupported trajectory step type, format or payload",
                        ),
                    );
                    return Ok(());
                }
                let mut payload = payload.expect("checked payload");
                if let Some(nested) = payload.get("payload").and_then(Json::as_object).cloned() {
                    let object = payload.as_object_mut().expect("checked payload object");
                    object.remove("payload");
                    object.extend(nested);
                }
                sink.records.fields("trajectory_step", row, &STEP_FIELDS);
                sink.records
                    .fields("trajectory_step_payload", &payload, &PAYLOAD_FIELDS);
                let ordinal = row
                    .get("step_id")
                    .or_else(|| row.get("id"))
                    .or_else(|| row.get("idx"))
                    .ok_or("trajectory step without identity")?;
                let native = ordinal
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| ordinal.to_string());
                let mid = message_id(ORIGIN, &format!("{}:step:{native}", sid.native()))
                    .ok_or("invalid step ID")?;
                let answered = last.filter(|(session, _, _)| *session == sid);
                let (role, prompt_kind, blocks, call) =
                    step(&kind, &payload, row, &mid.to_string(), answered.as_ref());
                if blocks.is_empty() {
                    self.calls = 0;
                    event(
                        sink.out,
                        Some(sid),
                        "antigravity_unsupported_step",
                        row.clone(),
                    );
                    sink.records.record(
                        table,
                        &RecordOutcome::unreadable("trajectory step has no readable content"),
                    );
                    return Ok(());
                }
                if matches!(blocks.first(), Some(Block::ToolCall { .. })) {
                    self.calls += 1;
                } else {
                    self.calls = 0;
                }
                self.last_call = call
                    .filter(|_| self.calls == 1)
                    .map(|(id, name)| (sid.clone(), id, name));
                let parent = state.last_message(&sid)?;
                state.set_last_message(&sid, &mid)?;
                let at = [
                    "timestamp",
                    "occurred_at",
                    "occurred_at_ms",
                    "created_at",
                    "createdAt",
                    "updated_at",
                ]
                .iter()
                .find_map(|key| {
                    let value = payload.get(*key).or_else(|| row.get(*key))?;
                    if *key == "occurred_at_ms" {
                        crate::time::epoch_millis(value)
                    } else {
                        crate::document::time_of(value)
                    }
                });
                let position = sink.position();
                sink.out.messages.push(MessageRow {
                    offset: position,
                    end: position + 1,
                    message: Message {
                        id: mid,
                        session: sid,
                        parent,
                        role,
                        prompt_kind,
                        at,
                        blocks,
                        model: None,
                        usage: Default::default(),
                        stop_reason: None,
                        reported_cost_usd: None,
                        duration_ms: None,
                        production: Default::default(),
                    },
                });
            }
            _ => {
                event(
                    sink.out,
                    sid,
                    "antigravity_database_row",
                    serde_json::json!({"table":table,"key":v["k"],"row":row}),
                );
            }
        }
        sink.records.record(table, &RecordOutcome::Used);
        Ok(())
    }
}
impl Database {
    fn named_session(&self, row: &Json) -> Option<String> {
        let identity = str_of(row, "trajectory_id").or_else(|| str_of(row, "cascade_id"))?;
        self.aliases.get(identity).cloned()
    }
    fn step_session(&self, row: &Json) -> Option<String> {
        let identity = str_of(row, "trajectory_id").or_else(|| str_of(row, "cascade_id"));
        match identity {
            Some(identity) => self.aliases.get(identity).cloned(),
            None if self.meta.len() == 1 => {
                let r = self.meta.values().next()?;
                Some(
                    str_of(r, "trajectory_id")
                        .or_else(|| str_of(r, "cascade_id"))
                        .unwrap_or(&self.unnamed)
                        .into(),
                )
            }
            None if self.meta.is_empty() => Some(self.unnamed.clone()),
            _ => None,
        }
    }
}
static STEP_FIELDS: Fields = Fields::new(&[
    ("idx", Rule::Read),
    ("id", Rule::Read),
    ("step_id", Rule::Read),
    ("trajectory_id", Rule::Read),
    ("cascade_id", Rule::Read),
    ("step_type", Rule::Read),
    ("step_format", Rule::Read),
    ("step_payload", Rule::Read),
    ("role", Rule::Read),
    ("timestamp", Rule::Read),
    ("occurred_at", Rule::Read),
    ("occurred_at_ms", Rule::Read),
    ("created_at", Rule::Read),
    ("createdAt", Rule::Read),
    ("updated_at", Rule::Read),
    ("exit_code", Rule::Read),
    ("exitCode", Rule::Read),
    ("status", Rule::Read),
    ("error", Rule::Read),
    ("error_details", Rule::Read),
]);
static PAYLOAD_FIELDS: Fields = Fields::new(&[
    ("text", Rule::Read),
    ("content", Rule::Read),
    ("message", Rule::Read),
    ("body", Rule::Read),
    ("output", Rule::Read),
    ("result", Rule::Read),
    ("role", Rule::Read),
    ("plan", Rule::Read),
    ("steps", Rule::Read),
    ("items", Rule::Read),
    ("tool_name", Rule::Read),
    ("toolName", Rule::Read),
    ("name", Rule::Read),
    ("tool_id", Rule::Read),
    ("toolId", Rule::Read),
    ("call_id", Rule::Read),
    ("callId", Rule::Read),
    ("input", Rule::Read),
    ("arguments", Rule::Read),
    ("args", Rule::Read),
    ("parameters", Rule::Read),
    ("command", Rule::Read),
    ("cmd", Rule::Read),
    ("path", Rule::Read),
    ("file_path", Rule::Read),
    ("filePath", Rule::Read),
    ("timestamp", Rule::Read),
    ("occurred_at", Rule::Read),
    ("occurred_at_ms", Rule::Read),
    ("created_at", Rule::Read),
    ("createdAt", Rule::Read),
    ("updated_at", Rule::Read),
    ("exit_code", Rule::Read),
    ("exitCode", Rule::Read),
    ("status", Rule::Read),
    ("error", Rule::Read),
    ("error_details", Rule::Read),
    ("old_string", Rule::Kept("native edit input")),
    ("new_string", Rule::Kept("native edit input")),
    ("oldString", Rule::Kept("native edit input")),
    ("newString", Rule::Kept("native edit input")),
    ("original_file", Rule::Kept("native edit input")),
    ("originalFile", Rule::Kept("native edit input")),
    ("structured_patch", Rule::Kept("native edit input")),
    ("structuredPatch", Rule::Kept("native edit input")),
    ("replace_all", Rule::Kept("native edit input")),
    ("replaceAll", Rule::Kept("native edit input")),
    ("user_modified", Rule::Kept("native edit input")),
    ("userModified", Rule::Kept("native edit input")),
]);

fn step_payload(row: &Json) -> Option<Json> {
    match &row["step_payload"] {
        Json::String(text) => serde_json::from_str(text).ok(),
        Json::Object(fields) if fields.len() == 1 && fields.contains_key("base64") => {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(fields["base64"].as_str()?)
                .ok()?;
            serde_json::from_slice(&bytes).ok()
        }
        payload @ Json::Object(_) => Some(payload.clone()),
        _ => None,
    }
}

fn tool_id(payload: &Json) -> Option<String> {
    ["tool_id", "toolId", "call_id", "callId"]
        .iter()
        .find_map(|key| match payload.get(*key)? {
            Json::String(id) if !id.is_empty() => Some(id.clone()),
            Json::Number(id) => Some(id.to_string()),
            _ => None,
        })
}

fn summary_id(row: &Json) -> Option<&str> {
    ["cascade_id", "trajectory_id", "id"]
        .iter()
        .find_map(|key| str_of(row, key))
}

fn parent_id(row: &Json) -> Option<&str> {
    [
        "parent_trajectory_id",
        "parent_cascade_id",
        "parent_session_id",
        "parent_id",
    ]
    .iter()
    .find_map(|key| str_of(row, key))
}

fn event(out: &mut Output, session: Option<SessionId>, kind: &str, data: Json) {
    event_at(out, session, kind, None, data);
}

fn event_at(
    out: &mut Output,
    session: Option<SessionId>,
    kind: &str,
    at: Option<polylogue_model::Timestamp>,
    data: Json,
) {
    out.events.push(Event {
        session,
        kind: kind.into(),
        at,
        after: None,
        range: None,
        data,
    });
}

type StepContent = (
    Role,
    Option<PromptKind>,
    Vec<Block>,
    Option<(String, String)>,
);
fn step(
    kind: &str,
    p: &Json,
    row: &Json,
    mid: &str,
    last: Option<&(SessionId, String, String)>,
) -> StepContent {
    let text = ["text", "content", "message", "body", "output", "result"]
        .iter()
        .find_map(|key| match p.get(*key)? {
            Json::String(text) if !text.trim().is_empty() => Some(text.clone()),
            value @ (Json::Object(_) | Json::Array(_)) => Some(value.to_string()),
            _ => None,
        });
    let text = text.or_else(|| {
        if matches!(kind, "plan" | "plan_step" | "planner" | "planning") {
            ["plan", "steps", "items"]
                .iter()
                .find_map(|key| p.get(*key).map(Json::to_string))
        } else {
            None
        }
    });
    let tool = ["tool_name", "toolName", "name"]
        .iter()
        .find_map(|key| str_of(p, key))
        .unwrap_or(
            if matches!(kind, "terminal" | "terminal_command" | "command") {
                "terminal"
            } else {
                kind
            },
        );
    let source_role = str_of(p, "role")
        .or_else(|| str_of(row, "role"))
        .map(|role| role.trim().to_ascii_lowercase());
    let call = tool_id(p);
    if matches!(kind, "tool_result" | "tool_output" | "command_result")
        || (kind == "message" && source_role.as_deref() == Some("tool"))
    {
        let call_id = call.or_else(|| {
            last.filter(|(_, _, name)| {
                ["tool_name", "toolName", "name"]
                    .iter()
                    .find_map(|key| str_of(p, key))
                    .is_none_or(|tool| name == tool)
            })
            .map(|(_, id, _)| id.clone())
        });
        let code = [p, row].into_iter().find_map(|values| {
            ["exit_code", "exitCode"]
                .iter()
                .find_map(|key| values.get(*key).and_then(Json::as_i64))
        });
        let status = str_of(p, "status")
            .or_else(|| str_of(row, "status"))
            .map(|s| s.trim().to_ascii_lowercase());
        let error = p.get("error").or_else(|| row.get("error"));
        let details = p.get("error_details").or_else(|| row.get("error_details"));
        let nonempty = |value: &Json| match value {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::String(s) => !s.is_empty(),
            Json::Array(a) => !a.is_empty(),
            Json::Object(o) => !o.is_empty(),
            _ => true,
        };
        let failed = error.is_some_and(nonempty)
            || details.is_some_and(nonempty)
            || matches!(
                status.as_deref(),
                Some("error" | "failed" | "failure" | "cancelled" | "canceled" | "aborted")
            );
        let outcome = if let Some(code) = code {
            if code == 0 {
                ToolOutcome::Ok
            } else {
                ToolOutcome::Error {
                    reason: Some(polylogue_model::ErrorReason::ExitCode(code)),
                }
            }
        } else if failed {
            ToolOutcome::Error { reason: None }
        } else if error == Some(&Json::Bool(false))
            || matches!(
                status.as_deref(),
                Some("ok" | "success" | "succeeded" | "completed" | "complete" | "done")
            )
        {
            ToolOutcome::Ok
        } else {
            ToolOutcome::Unknown {
                reason: UnknownReason::StatusFieldUnreadable,
            }
        };
        return (
            Role::User,
            Some(PromptKind::Harness),
            vec![Block::ToolResult {
                call_id,
                outcome,
                output: text,
                data: Some(p.clone()),
                attachments: Vec::new(),
            }],
            None,
        );
    }
    if matches!(
        kind,
        "terminal"
            | "terminal_command"
            | "command"
            | "tool"
            | "tool_call"
            | "tool_use"
            | "file_edit"
            | "edit"
    ) {
        let input = ["input", "arguments", "args", "parameters"]
            .iter()
            .find_map(|key| p.get(*key))
            .cloned()
            .unwrap_or_else(|| p.clone());
        let command = str_of(p, "command")
            .or_else(|| str_of(p, "cmd"))
            .or_else(|| str_of(&input, "command"))
            .map(str::to_owned);
        let path = str_of(p, "path")
            .or_else(|| str_of(p, "file_path"))
            .or_else(|| str_of(p, "filePath"))
            .or_else(|| str_of(&input, "path"))
            .map(str::to_owned);
        let action = if matches!(kind, "file_edit" | "edit") {
            ToolAction::FileEdit
        } else if command.is_some() {
            ToolAction::Shell
        } else {
            ToolAction::Other
        };
        let no_id = call.is_none();
        let id = call.unwrap_or_else(|| format!("{mid}:tool"));
        return (
            Role::Assistant,
            None,
            vec![Block::ToolCall {
                call_id: Some(id.clone()),
                tool: Some(tool.into()),
                action,
                input,
                command,
                path,
                run_by_script: false,
            }],
            no_id.then(|| (id, tool.into())),
        );
    }
    let role = match source_role.as_deref() {
        Some("user" | "human") => Role::User,
        Some("system") => Role::System,
        _ if matches!(kind, "user" | "human" | "prompt") => Role::User,
        _ => Role::Assistant,
    };
    (
        role,
        (role == Role::User).then_some(PromptKind::Prose),
        text.map(|text| vec![Block::Text { text }])
            .unwrap_or_default(),
        None,
    )
}

#[cfg(test)]
mod tests;
