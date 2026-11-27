//! Hermes state rows, read from one stored row dump. Session facts
//! and record identities are metadata; message bodies are decoded one at a time.
use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 5;

parser_modules! {
    mod identity;
    mod message;
    mod observer;
    mod outcome;
    mod trajectory;
    mod verification;
}

use crate::fields::ExpectedType as T;

use crate::document::{Sink, message_id, session_id};
use crate::export::{ExportParser, ExportState, Input, ItemSink, ScanError, ScanStats};
use crate::fields::{Fields, Rule};
use crate::parse::{MessageRow, Output, Parse, Step};
use crate::{RecordOutcome, Records};
use polylogue_model::{
    Event, Json, Link, LinkKind, MessageId, Session, SessionId, SessionKind, TitleSource,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Range;
use std::path::Path;

/// Database file names under the Hermes root.
pub const LOCATIONS: &[&str] = &["state.db", "verification_evidence.db"];
pub const ORIGIN: &str = "hermes";

/// The ID of a session in the Hermes state database at `state_db`, as its
/// parsed rows name it, for the hook events Hermes sends while it runs.
pub fn session_at(state_db: &Path, native: &str) -> Option<SessionId> {
    let source = serde_json::json!({ "path": state_db });
    let profile = identity::profile(Some(&source)).ok()??;
    session_key(native, &profile)
}

fn session_key(native: &str, profile: &str) -> Option<SessionId> {
    session_id(ORIGIN, &format!("{native}@profile-{profile}"))
}

/// An event whose time is unknown, as every Hermes diagnostic and marker is.
fn unknown_time_event(
    session: SessionId,
    kind: &str,
    after: Option<MessageId>,
    data: Json,
) -> Event {
    let (at, time_accuracy) = crate::time::parts(None);
    Event {
        session: Some(session),
        kind: kind.into(),
        at,
        at_quality: time_accuracy.quality,
        at_precision: time_accuracy.precision,
        at_until: time_accuracy.until,
        after,
        range: None,
        data,
    }
}

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    if observer::recognize(path, head)
        || verification::recognize(path)
        || trajectory::recognize(path, head)
    {
        return true;
    }
    path.file_name()
        .is_some_and(|name| name == "state.db.jsonl")
        && crate::document::starts_object_lines(head)
}

pub fn is_database(path: &Path) -> bool {
    path.file_name().is_some_and(|name| {
        matches!(
            name.to_str(),
            Some(
                "state.db"
                    | "state.db.jsonl"
                    | "verification_evidence.db"
                    | "verification_evidence.db.jsonl"
            )
        )
    })
}

pub fn open(path: &Path) -> Option<Parse> {
    if path
        .file_name()
        .is_some_and(|name| name == "state.db.jsonl" || name == "state.db")
    {
        Some(Parse::Export(Box::new(Database::default())))
    } else if verification::recognize(path)
        || path
            .file_name()
            .is_some_and(|name| name == "verification_evidence.db")
    {
        Some(verification::open())
    } else if path.extension().is_some_and(|s| s == "json") {
        Some(trajectory::open())
    } else {
        observer::open(path)
    }
}

#[derive(Default)]
struct Database {
    profile: Option<String>,
    refusal: Option<String>,
    unbound: bool,
    rows_seen: bool,
    sessions: BTreeMap<String, Facts>,
    selected: HashMap<String, Selected>,
    superseded: BTreeMap<String, u64>,
    order: HashMap<String, u64>,
}

#[derive(Clone)]
struct Facts {
    model: Option<String>,
    parent: Option<String>,
    branch: Option<LinkKind>,
    end: Option<String>,
    kind: SessionKind,
    system: bool,
    messages: bool,
    leaf: Option<MessageId>,
    activity_known: bool,
}
struct Selected {
    session: Option<String>,
    number: Option<i64>,
    seed: Option<String>,
    id: Option<MessageId>,
    previous: Option<MessageId>,
    active: Option<bool>,
}

static SESSION_FIELDS: Fields = Fields::new(
    "session",
    &[
        ("id", Rule::Read(T::String)),
        ("source", Rule::Read(T::String)),
        ("model", Rule::Read(T::String)),
        ("model_config", Rule::Read(T::StringOrObject)),
        ("parent_session_id", Rule::Read(T::String)),
        ("title", Rule::Read(T::String)),
        ("cwd", Rule::Read(T::String)),
        ("git_branch", Rule::Read(T::String)),
        ("git_remote", Rule::Read(T::String)),
        ("started_at", Rule::Read(T::StringOrNumber)),
        ("ended_at", Rule::Read(T::StringOrNumber)),
        ("end_reason", Rule::Read(T::String)),
        ("system_prompt", Rule::Read(T::String)),
    ],
);

fn lines(
    reader: &mut dyn Input,
    mut visit: impl FnMut(Range<u64>, &Json, &[u8]) -> Result<(), String>,
) -> Result<ScanStats, ScanError> {
    let mut bytes = Vec::new();
    let mut offset = 0;
    let mut stats = ScanStats::default();
    loop {
        bytes.clear();
        let size = reader
            .read_until(b'\n', &mut bytes)
            .map_err(|error| ScanError::Input(serde_json::Error::io(error)))?;
        if size == 0 {
            break;
        }
        if bytes.last() != Some(&b'\n') {
            return Err(ScanError::Callback(
                "incomplete Hermes row dump line".into(),
            ));
        }
        let value = serde_json::from_slice(&bytes).map_err(ScanError::Malformed)?;
        stats.elements += 1;
        stats.peak_element_bytes = stats.peak_element_bytes.max(bytes.len());
        visit(offset..offset + size as u64, &value, &bytes).map_err(ScanError::Callback)?;
        offset += size as u64;
    }
    Ok(stats)
}
fn key(value: &Json) -> Option<String> {
    Some(format!("{}\0{}", value.str("t")?, value.get("k")?))
}
fn text(value: &Json, field: &str) -> Option<String> {
    value.str(field).map(str::to_owned)
}
fn native(value: &Json, field: &str) -> Option<String> {
    text(value, field).filter(|value| !value.is_empty())
}
fn decoded(value: &Json, field: &str) -> Option<Json> {
    serde_json::from_str(value.str(field)?).ok()
}

impl Database {
    fn sid(&self, native: &str) -> Option<SessionId> {
        session_key(native, self.profile.as_ref()?)
    }
    fn mid(&self, session: &str, native: &str) -> Option<MessageId> {
        message_id(ORIGIN, &format!("{}:{native}", self.sid(session)?.native()))
    }
    fn header(&mut self, value: &Json) {
        match identity::profile(value.get("source")) {
            Ok(Some(profile))
                if !self.unbound && self.profile.as_ref().is_none_or(|old| old == &profile) =>
            {
                self.profile = Some(profile)
            }
            Ok(None) if self.profile.is_none() => self.unbound = true,
            _ => self.refusal = Some(
                "Hermes export has conflicting, invalid or historically unbound source bindings"
                    .into(),
            ),
        }
    }
    fn order_messages(&mut self) {
        // Ancestors precede their children. A genuine cycle is kept as a
        // source relationship; the reader reports its parents as incomplete.
        for session in self.sessions.keys() {
            let mut seen = HashSet::new();
            let mut at = Some(session.clone());
            let mut depth = 0;
            while let Some(id) = at {
                if !seen.insert(id.clone()) {
                    break;
                }
                at = self
                    .sessions
                    .get(&id)
                    .and_then(|facts| facts.parent.clone());
                depth += 1;
            }
            self.order.insert(session.clone(), depth);
        }
        let mut rows = self
            .selected
            .iter()
            .filter_map(|(key, row)| Some((row.session.clone()?, row.number?, key.clone())))
            .collect::<Vec<_>>();
        rows.sort();
        let mut occurrences = HashMap::<(String, String), u64>::new();
        let mut previous = HashMap::<String, MessageId>::new();
        for (session, _, key) in rows {
            let Some(seed) = self.selected[&key].seed.clone() else {
                continue;
            };
            let n = occurrences
                .entry((session.clone(), seed.clone()))
                .or_default();
            let suffix = if seed.starts_with('~') || *n > 0 {
                if seed.starts_with('~') {
                    format!("{seed}-{n}")
                } else {
                    format!("{seed}~occurrence-{n}")
                }
            } else {
                seed
            };
            *n += 1;
            let id = self.mid(&session, &suffix);
            let row = self.selected.get_mut(&key).unwrap();
            row.id = id.clone();
            row.previous = previous.get(&session).cloned();
            if let Some(facts) = self.sessions.get_mut(&session) {
                facts.messages = true;
                facts.activity_known &= row.active.is_some();
                if row.active != Some(false) {
                    facts.leaf = id.clone();
                }
            }
            if row.active != Some(false)
                && let Some(id) = id
            {
                previous.insert(session, id);
            }
        }
    }
    fn parent(&self, session: &str) -> Option<(String, LinkKind)> {
        let facts = self.sessions.get(session)?;
        let mut parent = facts.parent.clone()?;
        let kind = facts.branch.or_else(|| {
            self.sessions
                .get(&parent)
                .and_then(|facts| facts.end.as_deref())
                .filter(|end| matches!(*end, "compression" | "compaction"))
                .map(|_| LinkKind::Continuation)
        })?;
        if kind == LinkKind::Continuation {
            let mut seen = HashSet::from([session.to_owned()]);
            while let Some(parent_facts) = self.sessions.get(&parent) {
                if !seen.insert(parent.clone()) || parent_facts.messages || parent_facts.system {
                    break;
                }
                let Some(next) = parent_facts.parent.clone() else {
                    break;
                };
                parent = next;
            }
            // A cycle through compression intermediaries leads back to the
            // session itself; a session is not its own continuation parent.
            if parent == session {
                return None;
            }
        }
        Some((parent, kind))
    }
}

impl ExportParser for Database {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn Input,
        items: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        *self = Self::default();
        let stats = lines(reader, |_, value, _| {
            self.rows_seen = true;
            if value.get("schema").is_some() {
                self.header(value);
                return Ok(());
            }
            if value.get("gone") == Some(&Json::Bool(true)) {
                return Ok(());
            }
            let Some(table) = value
                .str("t")
                .filter(|table| matches!(*table, "sessions" | "messages"))
            else {
                return Ok(());
            };
            let Some(key) = key(value) else {
                return Ok(());
            };
            let row = &value["r"];
            let selected = if table == "sessions" {
                if let Some(id) = native(row, "id") {
                    let config = decoded(row, "model_config").unwrap_or(Json::Null);
                    let source = text(row, "source");
                    self.sessions.insert(
                        id,
                        Facts {
                            model: text(row, "model"),
                            parent: native(row, "parent_session_id"),
                            branch: if config
                                .get("_branched_from")
                                .is_some_and(|value| !value.is_null())
                            {
                                Some(LinkKind::Fork)
                            } else if config
                                .get("_delegate_from")
                                .is_some_and(|value| !value.is_null())
                                || source.as_deref() == Some("tool")
                            {
                                Some(LinkKind::Spawn)
                            } else {
                                None
                            },
                            kind: if matches!(source.as_deref(), Some("subagent" | "tool")) {
                                SessionKind::Subagent
                            } else {
                                SessionKind::Main
                            },
                            end: text(row, "end_reason"),
                            system: row.get("system_prompt").is_some_and(Json::is_string),
                            messages: false,
                            leaf: None,
                            activity_known: true,
                        },
                    );
                }
                Selected {
                    session: None,
                    number: None,
                    seed: None,
                    id: None,
                    previous: None,
                    active: None,
                }
            } else {
                Selected {
                    session: native(row, "session_id"),
                    number: row.i64("id"),
                    seed: message::seed(row),
                    id: None,
                    previous: None,
                    active: message::active(row),
                }
            };
            if self.selected.insert(key, selected).is_some() {
                *self.superseded.entry(table.into()).or_default() += 1;
            }
            Ok(())
        })?;
        if self.profile.is_none() {
            self.refusal.get_or_insert(
                "Hermes row dump names no profile; original rows stay unbound".into(),
            );
        }
        self.order_messages();
        reader
            .rewind()
            .map_err(|error| ScanError::Input(serde_json::Error::io(error)))?;
        lines(reader, |range, value, raw| {
            let table = value.str("t");
            let row_key = key(value);
            let positive = value.get("gone") != Some(&Json::Bool(true));
            let known =
                matches!(table, Some("sessions" | "messages")) && positive && row_key.is_some();
            let mut order = Vec::new();
            if value.get("schema").is_some() {
                order.push(0);
            } else if table == Some("sessions") && positive {
                order.push(1);
                let id = text(&value["r"], "id").unwrap_or_default();
                order.extend_from_slice(&self.order.get(&id).copied().unwrap_or(0).to_be_bytes());
                order.extend_from_slice(id.as_bytes());
            } else if table == Some("messages") && positive {
                order.push(2);
                let sid = text(&value["r"], "session_id").unwrap_or_default();
                order.extend_from_slice(&self.order.get(&sid).copied().unwrap_or(0).to_be_bytes());
                order.extend_from_slice(&(sid.len() as u64).to_be_bytes());
                order.extend_from_slice(sid.as_bytes());
                let n = value["r"]["id"].as_i64().unwrap_or(i64::MAX);
                order.extend_from_slice(&(n as u64 ^ (1u64 << 63)).to_be_bytes());
            } else {
                order.push(3);
            }
            items.push(
                range,
                order,
                if known { row_key.as_deref() } else { None },
                raw,
            )
        })?;
        Ok(stats)
    }
    fn begin(&mut self, length: u64, _: Option<&[u8]>) -> Result<Option<Step>, String> {
        let mut records = Records::default();
        for (kind, count) in &self.superseded {
            for _ in 0..*count {
                records.record(
                    kind,
                    &RecordOutcome::skipped("superseded row stays in the stored input"),
                );
            }
        }
        Ok(Some(Step {
            output: Output::default(),
            counts: records.finish().map_err(|error| error.to_string())?,
            start: 0,
            consumed: length,
            // Rows restate their whole record: a later row replaces an earlier one.
            repeats_whole_conversations: self.refusal.is_none() && self.rows_seen,
        }))
    }
    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let value: Json = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        let table = value.str("t").unwrap_or("?");
        if let Some(reason) = &self.refusal {
            sink.records.record(table, &RecordOutcome::skipped(reason));
            return Ok(());
        }
        if value.get("schema").is_some() {
            sink.records.record("schema", &RecordOutcome::Used);
            return Ok(());
        }
        if value.get("gone") == Some(&Json::Bool(true)) {
            sink.records.record(
                table,
                &RecordOutcome::skipped(
                    "provider deleted the row; its last known positive value is kept",
                ),
            );
            return Ok(());
        }
        let row = &value["r"];
        match table {
            "sessions" => {
                let Some(raw) = native(row, "id") else {
                    sink.records.record(
                        table,
                        &RecordOutcome::unreadable("Hermes session ID is not a nonempty string"),
                    );
                    return Ok(());
                };
                let Some(id) = self.sid(&raw) else {
                    return Err("invalid Hermes session identity".into());
                };
                let facts = self
                    .sessions
                    .get(&raw)
                    .ok_or("missing Hermes session facts")?;
                sink.records.fields("sessions", row, &SESSION_FIELDS);
                let parent = self.parent(&raw);
                if !facts.messages
                    && !facts.system
                    && parent
                        .as_ref()
                        .is_some_and(|(_, kind)| *kind == LinkKind::Continuation)
                {
                    sink.records.record(
                        table,
                        &RecordOutcome::skipped(
                            "empty compression intermediary; parents use its nearest stored source",
                        ),
                    );
                    return Ok(());
                }
                let mut session = Session::new(id.clone(), facts.kind);
                if let Some(title) = text(row, "title").filter(|title| title.trim() != "New chat") {
                    session.offer_title(&title, TitleSource::Generated);
                }
                if let Some(cwd) = text(row, "cwd") {
                    session.offer_cwd(&cwd);
                }
                session.git_branch = text(row, "git_branch");
                session.repo = text(row, "git_remote");
                let (at, accuracy) =
                    crate::time::parts(row.get("started_at").and_then(crate::time::epoch_seconds));
                session
                    .set_time(at, accuracy)
                    .expect("decoded source time is valid");
                sink.out.sessions.push(session);
                sink.out.events.push(unknown_time_event(
                    id.clone(),
                    "session_metadata",
                    None,
                    row.clone(),
                ));
                let mut limits: Vec<String> = Vec::new();
                if facts.messages && (!facts.activity_known || facts.leaf.is_none()) {
                    sink.out.events.push(unknown_time_event(
                        id.clone(),
                        if facts.activity_known {
                            "active_path_empty"
                        } else {
                            "active_path_unknown"
                        },
                        None,
                        Json::Null,
                    ));
                }
                if row.get("source").is_some_and(|value| {
                    !matches!(value.as_str(), Some("cli" | "subagent" | "tool"))
                }) {
                    limits.push("unrecognized source; default session kind is main".into());
                }
                if row
                    .get("model")
                    .is_some_and(|value| !value.is_null() && !value.is_string())
                {
                    limits.push("model is not text; no model assigned".into());
                }
                for field in ["started_at", "ended_at"] {
                    if row.get(field).is_some_and(|value| {
                        !value.is_null() && crate::time::epoch_seconds(value).is_none()
                    }) {
                        limits.push(format!(
                            "{field} epoch seconds unreadable; time remains unknown"
                        ));
                    }
                }
                if !limits.is_empty() {
                    sink.out.events.push(unknown_time_event(
                        id.clone(),
                        "hermes_session_diagnostic",
                        None,
                        serde_json::json!({"reason":limits}),
                    ));
                }
                if let Some((parent, kind)) = parent
                    && let Some(parent_id) = self.sid(&parent)
                {
                    // A live fork/spawn parent can have advanced since the child
                    // began. Its present leaf is not a supplied branch point.
                    let branch_point = self
                        .sessions
                        .get(&parent)
                        .filter(|_| kind == LinkKind::Continuation)
                        .and_then(|facts| {
                            facts.leaf.clone().or_else(|| {
                                facts.system.then(|| self.mid(&parent, "system")).flatten()
                            })
                        });
                    sink.out.links.push(Link {
                        kind,
                        copies_prefix: false,
                        reason: None,
                        parent: parent_id,
                        child: id.clone(),
                        branch_point,
                        call_id: None,
                        evidence: "Hermes parent_session_id/model_config/end_reason".into(),
                    });
                }
                if let Some(prompt) = text(row, "system_prompt") {
                    let offset = sink.position();
                    sink.out.messages.push(MessageRow {
                        response: None,
                        page_text: false,
                        id_kind: polylogue_model::IdKind::Native,
                        offset,
                        end: offset + 1,
                        message: message::system(
                            self.mid(&raw, "system").ok_or("invalid system identity")?,
                            id.clone(),
                            prompt,
                        ),
                    });
                }
                if let Some(end) = text(row, "end_reason") {
                    sink.out.events.push({
                        let (at, time_accuracy) = crate::time::parts(
                            row.get("ended_at").and_then(crate::time::epoch_seconds),
                        );
                        Event {
                            session: Some(id),
                            kind: "session_end".into(),
                            at,
                            at_quality: time_accuracy.quality,
                            at_precision: time_accuracy.precision,
                            at_until: time_accuracy.until,
                            after: facts.leaf.clone(),
                            range: None,
                            data: serde_json::json!({"end_reason":end}),
                        }
                    });
                }
                sink.records.record(table, &RecordOutcome::Used);
            }
            "messages" => {
                let Some(key) = key(&value) else {
                    sink.records.record(
                        table,
                        &RecordOutcome::unreadable("Hermes message has no row key"),
                    );
                    return Ok(());
                };
                let selected = self
                    .selected
                    .get(&key)
                    .ok_or("missing Hermes message key")?;
                if selected.seed.is_none() {
                    sink.records.record(table, &RecordOutcome::skipped("unrecognized Hermes role or message shape; authorship was not assigned"));
                    return Ok(());
                }
                let (Some(raw), Some(id), Some(_)) =
                    (&selected.session, &selected.id, selected.number)
                else {
                    sink.records.record(
                        table,
                        &RecordOutcome::unreadable(
                            "Hermes message needs a textual session and integer insertion ID",
                        ),
                    );
                    return Ok(());
                };
                let Some(facts) = self.sessions.get(raw) else {
                    sink.records.record(
                        table,
                        &RecordOutcome::skipped("Hermes message has no owning session"),
                    );
                    return Ok(());
                };
                let sid = self.sid(raw).ok_or("invalid message session")?;
                sink.records.fields("messages", row, &message::FIELDS);
                match message::parse(row, id.clone(), sid.clone(), facts.model.clone()) {
                    Ok((mut message, diagnostics)) => {
                        message.parent = selected
                            .previous
                            .clone()
                            .or_else(|| facts.system.then(|| self.mid(raw, "system")).flatten());
                        for (kind, data) in diagnostics {
                            sink.out.events.push({
                                let (at, time_accuracy) = (message.at, message.time_accuracy());
                                Event {
                                    session: Some(sid.clone()),
                                    kind,
                                    at,
                                    at_quality: time_accuracy.quality,
                                    at_precision: time_accuracy.precision,
                                    at_until: time_accuracy.until,
                                    after: Some(id.clone()),
                                    range: None,
                                    data,
                                }
                            });
                        }
                        let offset = sink.position();
                        sink.out.messages.push(MessageRow {
                            response: None,
                            page_text: false,
                            id_kind: polylogue_model::IdKind::Native,
                            offset,
                            end: offset + 1,
                            message,
                        });
                        if facts.activity_known && facts.leaf.as_ref() == Some(id) {
                            sink.out.events.push(unknown_time_event(
                                sid,
                                "active_leaf",
                                Some(id.clone()),
                                Json::Null,
                            ));
                        }
                        sink.records.record(table, &RecordOutcome::Used);
                    }
                    Err(reason) => sink.records.record(table, &RecordOutcome::skipped(reason)),
                }
            }
            "schema_version" => sink.records.record(table, &RecordOutcome::Used),
            _ => {
                sink.records.unknown_table(table, table, row);
                sink.records.record(
                    table,
                    &RecordOutcome::skipped("Hermes auxiliary row stays in the stored input"),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
