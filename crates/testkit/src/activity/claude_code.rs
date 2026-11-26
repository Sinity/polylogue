//! The files Claude Code writes for an [`Activity`], written from the format
//! as `examples/sources/claude-code*` and `examples/morrow/claude-code` show
//! it.

use super::{Activity, Body, Line, Part, changed_text};
use serde_json::{Value, json};
use std::path::PathBuf;

/// A file a tool wrote: its path under the home and its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct File {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
}

impl Activity {
    /// One `.claude/projects/<project>/<session>.jsonl` per session, the
    /// project named from the session's directory as Claude Code names it
    /// (`/home/op/work` is `-home-op-work`).
    pub fn claude_code_files(&self) -> Vec<File> {
        (0..self.sessions.len())
            .map(|index| {
                let transcript = &self.sessions[index];
                let project = transcript.cwd.replace('/', "-");
                let mut bytes = Vec::new();
                for line in self.lines(index) {
                    for record in records(&line) {
                        bytes.extend(record.to_string().as_bytes());
                        bytes.push(b'\n');
                    }
                }
                File {
                    path: PathBuf::from(format!(
                        ".claude/projects/{project}/{}.jsonl",
                        transcript.session
                    )),
                    bytes,
                }
            })
            .collect()
    }
}

/// The JSON records of one entry, in order.
fn records(line: &Line<'_>) -> Vec<Value> {
    let entry = line.entry;
    let at = jiff::Timestamp::from_millisecond(entry.at)
        .expect("a time in range")
        .to_string();
    let wrapper = |uuid: &str, parent: Option<&str>| {
        json!({
            "parentUuid": parent,
            "isSidechain": entry.sidechain,
            "userType": "external",
            "cwd": line.cwd,
            "sessionId": line.owner,
            "version": "2.1.220",
            "gitBranch": "main",
            "uuid": uuid,
            "timestamp": at,
        })
    };
    let with = |mut record: Value, fields: Value| {
        let object = record.as_object_mut().expect("a record is an object");
        for (key, value) in fields.as_object().expect("fields are an object") {
            object.insert(key.clone(), value.clone());
        }
        record
    };
    let parent = entry.parent.as_deref();
    match &entry.body {
        Body::Prompt(text) => vec![with(
            wrapper(&entry.uuid, parent),
            json!({"type": "user", "message": {"role": "user", "content": text}}),
        )],
        Body::Response { key, records } => {
            let mut previous = parent;
            let last = records.len() - 1;
            records
                .iter()
                .enumerate()
                .map(|(n, record)| {
                    let mut text_seen = false;
                    let content: Vec<Value> = record
                        .parts
                        .iter()
                        .map(|part| match part {
                            Part::Text(text) => {
                                let text = match line.change {
                                    Some(how) if !text_seen => changed_text(text, how),
                                    _ => text.clone(),
                                };
                                text_seen = true;
                                json!({"type": "text", "text": text})
                            }
                            Part::Thinking(text) => {
                                json!({"type": "thinking", "thinking": text, "signature": "c2lnbmF0dXJl"})
                            }
                            Part::Call { id, command } => json!({
                                "type": "tool_use", "id": id, "name": "Bash",
                                "input": {"command": command, "description": "Run a command"}
                            }),
                        })
                        .collect();
                    // Only a response's last record carries its stop reason,
                    // and a copy caught while the response was written has none.
                    let stop = (n == last && line.change.is_none()).then(|| {
                        let calls = records
                            .iter()
                            .flat_map(|r| &r.parts)
                            .any(|p| matches!(p, Part::Call { .. }));
                        if calls { "tool_use" } else { "end_turn" }
                    });
                    let value = with(
                        wrapper(&record.uuid, previous),
                        json!({
                            "type": "assistant",
                            "requestId": format!("req_{key}"),
                            "message": {
                                "model": "claude-fable-5-1", "id": key, "type": "message",
                                "role": "assistant", "content": content,
                                "stop_reason": stop, "stop_sequence": null,
                            },
                        }),
                    );
                    previous = Some(record.uuid.as_str());
                    value
                })
                .collect()
        }
        Body::Results(results) => {
            let content: Vec<Value> = results
                .iter()
                .map(|r| {
                    json!({"type": "tool_result", "tool_use_id": r.call, "content": r.output, "is_error": r.error})
                })
                .collect();
            vec![with(
                wrapper(&entry.uuid, parent),
                json!({"type": "user", "message": {"role": "user", "content": content}}),
            )]
        }
        Body::Compaction { boundary, summary } => vec![
            with(
                wrapper(boundary, None),
                json!({
                    "logicalParentUuid": parent,
                    "type": "system", "subtype": "compact_boundary",
                    "content": "Conversation compacted", "level": "info",
                    "compactMetadata": {"trigger": "manual", "preTokens": 2000, "postTokens": 200},
                }),
            ),
            with(
                wrapper(&entry.uuid, Some(boundary)),
                json!({
                    "type": "user", "message": {"role": "user", "content": summary},
                    "isVisibleInTranscriptOnly": true, "isCompactSummary": true,
                }),
            ),
        ],
    }
}
