//! Session context kept beside transcripts. Only explicit IDs establish a
//! session link; a filename hash or a short ID prefix does not.

use crate::parse::{DocumentParser, Output, Parse};
use crate::{RecordOutcome, Records};
use polylogue_model::{Event, Json, SessionId};
use std::path::Path;

pub const COMPRESSED_LOCATIONS: &[&str] = &[
    "sessions/**/rollout-*.jsonl.zst",
    "archived_sessions/**/rollout-*.jsonl.zst",
];

pub const CLAUDE_LOCATIONS: &[&str] = &[
    "paste-cache/**/*",
    "teams/**/*",
    "usage-data/**/*",
    "history-summaries/**/*",
    "remote-agents/*.meta.json",
    "jobs/**/*",
    "shell-snapshots/**/*",
    "debug/**/*",
    "mcp-logs/**/*",
    "projects/*/memory/**/*",
];

pub const CODEX_LOCATIONS: &[&str] = &[
    "attachments/**/*",
    "memories/**/*",
    "shell_snapshots/**/*",
    "dictation-history",
    "dictation-history/**/*",
    "visualizations/**/*",
];

pub fn recognize_compressed(path: &Path, head: &[u8]) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("rollout-") && n.ends_with(".jsonl.zst"))
        && (head.starts_with(&[0x28, 0xb5, 0x2f, 0xfd])
            || head.first().is_some_and(|b| (0x50..=0x5f).contains(b))
                && head.get(1..4) == Some(&[0x2a, 0x4d, 0x18]))
}

fn relative(path: &Path, locations: &[&str]) -> Option<String> {
    let names: Vec<_> = path.components().collect();
    for (i, name) in names.iter().enumerate() {
        let Some(name) = name.as_os_str().to_str() else {
            continue;
        };
        if locations.iter().any(|l| l.split('/').next() == Some(name)) {
            return Some(
                names[i..]
                    .iter()
                    .collect::<std::path::PathBuf>()
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    None
}

pub fn recognize_claude(path: &Path, _: &[u8]) -> bool {
    relative(path, CLAUDE_LOCATIONS).is_some()
}
pub fn recognize_codex(path: &Path, _: &[u8]) -> bool {
    relative(path, CODEX_LOCATIONS).is_some()
}
pub fn open_claude(path: &Path) -> Option<Parse> {
    Some(Parse::Document(Box::new(Context {
        path: relative(path, CLAUDE_LOCATIONS)?,
        codex: false,
    })))
}
pub fn open_codex(path: &Path) -> Option<Parse> {
    Some(Parse::Document(Box::new(Context {
        path: relative(path, CODEX_LOCATIONS)?,
        codex: true,
    })))
}

struct Context {
    path: String,
    codex: bool,
}

impl Context {
    fn session(&self, value: &Json) -> Option<SessionId> {
        let explicit = ["sessionId", "session_id", "thread_id", "leadSessionId"]
            .iter()
            .find_map(|key| value.get(key).and_then(Json::as_str));
        let path = Path::new(&self.path);
        let stem = path.file_stem().and_then(|s| s.to_str());
        let directory = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str());
        let from_path = if self.codex && self.path.starts_with("attachments/") {
            directory
        } else if self.codex && self.path.starts_with("shell_snapshots/") {
            stem.and_then(|s| s.split('.').next())
        } else if !self.codex
            && (self.path.starts_with("usage-data/session-meta/")
                || self.path.starts_with("usage-data/facets/")
                || self.path.starts_with("debug/"))
        {
            stem.filter(|s| uuid_shape(s))
        } else {
            None
        };
        let native = explicit.or(from_path)?;
        if self.codex {
            crate::codex::session_id(native)
        } else {
            crate::claude_code::session_id(native)
        }
    }
}

fn uuid_shape(s: &str) -> bool {
    s.len() == 36
        && s.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
}

impl DocumentParser for Context {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut Records) {
        let value = serde_json::from_slice::<Json>(bytes).unwrap_or(Json::Null);
        let session = self.session(&value);
        let kind = if self.path.starts_with("paste-cache/") {
            "paste_file"
        } else {
            "context_file"
        };
        out.events.push(Event {
            session,
            kind: kind.into(),
            at: None,
            after: None,
            range: Some((0, bytes.len() as u64)),
            data: serde_json::json!({"path": self.path, "size": bytes.len()}),
        });
        records.record(kind, &RecordOutcome::Used);
    }
}

#[cfg(test)]
mod tests;
