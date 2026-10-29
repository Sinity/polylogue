//! Antigravity language-server Markdown exports. Activity markers are agent
//! actions even when the exporter places them inside a User Input section.
use crate::document::{Sink, message_id, session_id, str_of};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::parse::{DocumentParser, MessageRow, Output, Parse};
use crate::{RecordOutcome, Records};
use polylogue_model::{
    BlobHash, Block, Event, Json, Message, PromptKind, Role, Session, SessionKind, TitleSource,
    ToolAction,
};
use std::collections::HashMap;
use std::path::Path;

pub mod server;

pub const VERSION: u32 = 1;

pub const ORIGIN: &str = "antigravity";
pub const LOCATIONS: &[&str] = &[".gemini/antigravity/conversations/**/*.pb"];

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    (path.extension().is_some_and(|e| e == "pb")
        && path.ancestors().any(|p| {
            p.file_name().is_some_and(|name| name == "conversations")
                && p.parent()
                    .is_some_and(|p| p.ends_with(".gemini/antigravity"))
        }))
        || path.extension().is_some_and(|e| e == "json")
            && ((String::from_utf8_lossy(head).contains("\"antigravity_language_server\"")
                || String::from_utf8_lossy(head).contains("\"antigravity_conversion\""))
                || crate::document::first_key(head) == Some("cascadeId"))
}
pub fn open(path: &Path) -> Option<Parse> {
    if path.extension().is_some_and(|e| e == "pb") {
        path.file_stem()
            .and_then(|name| name.to_str())
            .and_then(|name| session_id(ORIGIN, name))
            .map(|session| Parse::Document(Box::new(ConversationFile { session })))
    } else {
        Some(Parse::Document(Box::new(Antigravity)))
    }
}

struct ConversationFile {
    session: polylogue_model::SessionId,
}
impl DocumentParser for ConversationFile {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut Records) {
        out.sessions
            .push(Session::new(self.session.clone(), SessionKind::Main));
        out.events.push({ let (at,time_evidence)=crate::time::parts(None); Event {
            session: Some(self.session.clone()),
            kind: "antigravity_conversation_bytes".into(),
 at, at_quality:time_evidence.quality, at_precision:time_evidence.precision, at_until:time_evidence.until,
            after: None,
            range: None,
            data: serde_json::json!({"hash":BlobHash::of(bytes).to_string(),"length":bytes.len()}),
        } });
        records.record("conversation_file", &RecordOutcome::Used);
    }
}
pub struct Antigravity;
static FIELDS: Fields = Fields::new(&[
    ("source", Rule::Read(T::String)),
    ("conversationFile", Rule::Read(T::String)),
    ("cascadeId", Rule::Read(T::String)),
    ("markdown", Rule::Read(T::String)),
    ("title", Rule::Read(T::String)),
    ("lastModifiedTime", Rule::Read(T::StringOrNumber)),
    (
        "workspaceName",
        Rule::Kept("workspace label, not a working directory"),
    ),
    (
        "snippet",
        Rule::Excluded("preview repeated in the transcript"),
    ),
])
.for_kind("record");

impl DocumentParser for Antigravity {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut Records) {
        let Ok(value) = serde_json::from_slice::<Json>(bytes) else {
            records.record(
                "export",
                &RecordOutcome::unreadable("malformed Antigravity JSON"),
            );
            return;
        };
        if value["source"] == "antigravity_conversion" {
            records.fields(
                "conversion",
                &value,
                &Fields::new(&[
                    ("source", Rule::Read(T::String)),
                    ("cascadeId", Rule::Read(T::String)),
                    ("conversationFile", Rule::Read(T::String)),
                    ("status", Rule::Read(T::String)),
                    ("server_pid", Rule::Read(T::Number)),
                    ("error", Rule::Read(T::String)),
                ]),
            );
            let Some(id) = str_of(&value, "cascadeId").and_then(|id| session_id(ORIGIN, id)) else {
                records.record(
                    "conversion",
                    &RecordOutcome::unreadable("invalid cascadeId"),
                );
                return;
            };
            out.sessions
                .push(Session::new(id.clone(), SessionKind::Main));
            out.events.push({
                let (at, time_evidence) = crate::time::parts(None);
                Event {
                    session: Some(id),
                    kind: "antigravity_conversion".into(),
                    at,
                    at_quality: time_evidence.quality,
                    at_precision: time_evidence.precision,
                    at_until: time_evidence.until,
                    after: None,
                    range: None,
                    data: value,
                }
            });
            records.record("conversion", &RecordOutcome::Used);
            return;
        }
        records.fields("export", &value, &FIELDS);
        let Some((native, markdown)) = str_of(&value, "cascadeId")
            .zip(value["markdown"].as_str())
            .filter(|_| value["source"] == "antigravity_language_server")
        else {
            records.record(
                "export",
                &RecordOutcome::unreadable("expected Antigravity source, cascadeId and markdown"),
            );
            return;
        };
        let Some(id) = session_id(ORIGIN, native) else {
            records.record("export", &RecordOutcome::unreadable("invalid cascadeId"));
            return;
        };
        let mut session = Session::new(id.clone(), SessionKind::Main);
        if let Some(title) = str_of(&value, "title") {
            session.offer_title(title, TitleSource::Generated);
        }
        out.sessions.push(session);
        let mut metadata = value.as_object().expect("validated export object").clone();
        metadata.remove("markdown");
        // The source reports its last update, not the time of each message.
        out.events.push({
            let (at, time_evidence) =
                crate::time::parts(value.get("lastModifiedTime").and_then(crate::time::rfc3339));
            Event {
                session: Some(id.clone()),
                kind: "antigravity_session".into(),
                at,
                at_quality: time_evidence.quality,
                at_precision: time_evidence.precision,
                at_until: time_evidence.until,
                after: None,
                range: None,
                data: Json::Object(metadata),
            }
        });
        let mut sink = Sink::new(out, records);
        let mut role = Role::Assistant;
        let mut body = String::new();
        let mut sections = Vec::new();
        let mut saw_heading = false;
        for line in markdown.lines() {
            let next_role = match line.trim_end() {
                "### User Input" => Some(Role::User),
                "### Planner Response" => Some(Role::Assistant),
                _ => None,
            };
            if let Some(next_role) = next_role {
                if saw_heading {
                    sections.push((role, std::mem::take(&mut body)));
                } else {
                    body.clear();
                }
                saw_heading = true;
                role = next_role;
            } else {
                body.push_str(line);
                body.push('\n');
            }
        }
        if !saw_heading {
            body = body
                .lines()
                .skip_while(|line| {
                    line.starts_with("# ") || line.starts_with("Note:") || line.trim().is_empty()
                })
                .collect::<Vec<_>>()
                .join("\n");
        }
        sections.push((role, body));
        let mut counts = HashMap::<String, usize>::new();
        let mut parent = None;
        for (role, body) in sections {
            for (activity, blocks) in runs(&body) {
                let message_role = if activity { Role::Assistant } else { role };
                let digest = BlobHash::of(
                    serde_json::to_vec(&(message_role, &blocks))
                        .expect("blocks serialize")
                        .as_slice(),
                )
                .to_string();
                let occurrence = counts.entry(digest.clone()).or_default();
                let mid = message_id(ORIGIN, &format!("{native}:{digest}:{}", *occurrence))
                    .expect("valid native ID");
                *occurrence += 1;
                let position = sink.position();
                sink.out.messages.push(MessageRow {
                    offset: position,
                    end: position + 1,
                    message: Message {
                        attrs: Default::default(),
                        id: mid.clone(),
                        session: id.clone(),
                        parent: parent.replace(mid),
                        role: message_role,
                        at: None,
                        model: None,
                        prompt_kind: (message_role == Role::User).then_some(PromptKind::Prose),
                        blocks,
                        usage: Default::default(),
                        stop_reason: None,
                        reported_cost_usd: None,
                        duration_ms: None,
                        production: Default::default(),
                    },
                });
                sink.records.record(
                    if activity { "activity" } else { "message" },
                    &RecordOutcome::Used,
                );
            }
        }
        sink.records.record("export", &RecordOutcome::Used);
    }
}

fn runs(body: &str) -> Vec<(bool, Vec<Block>)> {
    let mut runs = Vec::new();
    let mut prose = String::new();
    let mut lines = body.lines().peekable();
    while let Some(line) = lines.next() {
        let mut candidate = line.to_owned();
        // Only accepted commands may span lines in the vendor format.
        if candidate.starts_with("*User accepted the command `") && !candidate.ends_with("`*") {
            let mut trial = lines.clone();
            // Count each line once; command bodies can be large.
            let mut backticks = candidate.matches('`').count();
            while backticks < 2 {
                let Some(next) = trial.next() else {
                    break;
                };
                backticks += next.matches('`').count();
                candidate.push('\n');
                candidate.push_str(next);
            }
            if activity(&candidate).is_some() {
                lines = trial;
            } else {
                candidate = line.to_owned();
            }
        }
        if let Some(block) = activity(&candidate) {
            if !prose.trim().is_empty() {
                runs.push((
                    false,
                    vec![Block::Text {
                        text: prose.trim().into(),
                    }],
                ));
                prose.clear();
            }
            if let Some((true, blocks)) = runs.last_mut() {
                blocks.push(block);
            } else {
                runs.push((true, vec![block]));
            }
        } else {
            prose.push_str(&candidate);
            prose.push('\n');
        }
    }
    if !prose.trim().is_empty() {
        runs.push((
            false,
            vec![Block::Text {
                text: prose.trim().into(),
            }],
        ));
    }
    runs
}

fn activity(line: &str) -> Option<Block> {
    let line = line.strip_prefix('*')?.strip_suffix('*')?.trim_end();
    let (tool, key, value, action) = if let Some(value) = line
        .strip_prefix("User accepted the command `")
        .and_then(|s| s.strip_suffix('`'))
    {
        if value.contains('`') {
            return None;
        }
        ("accepted_command", "command", value, ToolAction::Shell)
    } else if let Some((prefix, tool, action)) = [
        ("Viewed [", "viewed_file", ToolAction::FileRead),
        ("Listed directory [", "listed_directory", ToolAction::Search),
    ]
    .into_iter()
    .find(|(prefix, _, _)| line.starts_with(prefix))
    {
        let value = line
            .strip_prefix(prefix)?
            .split_once("](")?
            .1
            .strip_suffix(')')?;
        (tool, "path", value, action)
    } else if let Some((prefix, tool, key)) = [
        ("Searched web for ", "searched_web", "query"),
        ("Read URL content from ", "read_url_content", "url"),
        ("Read resource from ", "read_resource", "resource"),
        ("Listed resources from ", "listed_resources", "server"),
    ]
    .into_iter()
    .find(|(prefix, _, _)| line.starts_with(prefix))
    {
        (tool, key, line.strip_prefix(prefix)?, ToolAction::Web)
    } else {
        let (tool, action) = match line {
            "Edited relevant file" => ("edited_file", ToolAction::FileEdit),
            "Checked command status" => ("checked_command_status", ToolAction::Shell),
            "Grep searched codebase" => ("grep_searched_codebase", ToolAction::Search),
            "Searched filesystem" => ("searched_filesystem", ToolAction::Search),
            "Viewed code item" => ("viewed_code_item", ToolAction::FileRead),
            "Viewed content chunk" => ("viewed_content_chunk", ToolAction::FileRead),
            _ => return None,
        };
        (tool, "", "", action)
    };
    let mut input = serde_json::Map::new();
    if !key.is_empty() && !value.trim().is_empty() {
        input.insert(key.into(), Json::String(value.trim().into()));
    }
    Some(Block::ToolCall {
        call_id: None,
        tool: Some(tool.into()),
        action,
        command: (key == "command").then(|| value.trim().into()),
        path: (key == "path").then(|| value.trim().into()),
        input: Json::Object(input),
        run_by_script: false,
    })
}

#[cfg(test)]
mod tests;
