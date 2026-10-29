//! Incremental identity reads from the hook spool owned by capture.
use polylogue_model::{
    SessionId,
    authorship::{CallerEvidence, HookIdentity},
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
    path::PathBuf,
    sync::Mutex,
};

pub struct Registry {
    directory: PathBuf,
    cursors: Mutex<BTreeMap<u32, Cursor>>,
}
#[derive(Default)]
struct Cursor {
    device: u64,
    inode: u64,
    offset: u64,
    calls: BTreeMap<String, HookIdentity>,
}
#[derive(Deserialize)]
struct Event {
    event: String,
    harness_pid: u32,
    provider: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    tool_use_id: Option<String>,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    payload: Payload,
}
#[derive(Default, Deserialize)]
struct Payload {
    session_id: Option<String>,
    tool_use_id: Option<String>,
    agent_id: Option<String>,
}

impl Registry {
    pub fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            cursors: Mutex::default(),
        }
    }
    pub fn identity(&self, caller: &CallerEvidence) -> Option<HookIdentity> {
        let pid = caller.harness_pid?;
        let file = File::open(self.directory.join(format!("{pid}.ndjson"))).ok()?;
        let metadata = file.metadata().ok()?;
        let mut cursors = self.cursors.lock().ok()?;
        let cursor = cursors.entry(pid).or_default();
        if cursor.device != metadata.dev()
            || cursor.inode != metadata.ino()
            || cursor.offset > metadata.len()
        {
            *cursor = Cursor {
                device: metadata.dev(),
                inode: metadata.ino(),
                ..Default::default()
            };
        }
        let mut input = BufReader::new(file);
        input.seek(SeekFrom::Start(cursor.offset)).ok()?;
        let mut line = Vec::new();
        loop {
            line.clear();
            let bytes = input.read_until(b'\n', &mut line).ok()?;
            if bytes == 0 || line.last() != Some(&b'\n') {
                break;
            }
            cursor.offset = cursor.offset.checked_add(bytes.try_into().ok()?)?;
            let Ok(event) = serde_json::from_slice::<Event>(&line) else {
                continue;
            };
            if event.harness_pid != pid {
                continue;
            }
            cursor.observe(event);
        }
        // A session association does not identify an active tool invocation.
        let selected = caller
            .tool_call_id
            .as_ref()
            .and_then(|tool| cursor.calls.get(tool))
            .cloned();
        selected.filter(|identity| {
            caller
                .harness
                .as_ref()
                .is_none_or(|harness| identity.session.origin() == harness)
        })
    }
}
impl Cursor {
    fn observe(&mut self, event: Event) {
        let tool = event.tool_use_id.or(event.payload.tool_use_id);
        let session = event
            .session_id
            .or(event.payload.session_id)
            .and_then(|native| SessionId::parse(&format!("{}:{native}", event.provider)).ok());
        let identity = session.map(|session| HookIdentity {
            harness_pid: event.harness_pid,
            tool_call_id: tool.clone(),
            session,
            agent_id: event.agent_id.or(event.payload.agent_id),
        });
        match event.event.as_str() {
            "SessionStart" | "SessionEnd" => {
                self.calls.clear();
            }
            "PreToolUse" => {
                if let (Some(tool), Some(identity)) = (tool, identity) {
                    self.calls.insert(tool, identity);
                }
            }
            "PostToolUse" | "PostToolUseFailure" | "ToolUseCancelled" => {
                if let Some(tool) = tool {
                    self.calls.remove(&tool);
                }
            }
            _ => {}
        }
    }
}
#[cfg(test)]
mod tests;
