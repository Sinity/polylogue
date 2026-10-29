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
    session: Option<HookIdentity>,
    calls: BTreeMap<String, Call>,
}
struct Call {
    identity: HookIdentity,
    polylogue_shell: bool,
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
    tool_name: Option<String>,
    tool_input: Option<ToolInput>,
}
#[derive(Deserialize)]
struct ToolInput {
    command: Option<String>,
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
        let selected = if let Some(tool) = caller.tool_call_id.as_ref() {
            cursor
                .calls
                .get(tool)
                .map(|call| call.identity.clone())
                .or_else(|| cursor.session.clone())
        } else {
            let mut matching = cursor.calls.values().filter(|call| call.polylogue_shell);
            let first = matching.next();
            if matching.next().is_none()
                && caller
                    .harness
                    .as_ref()
                    .is_some_and(|origin| origin.as_str() == "claude-code")
            {
                first
                    .map(|call| call.identity.clone())
                    .or_else(|| cursor.session.clone())
            } else {
                cursor.session.clone()
            }
        };
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
            "SessionStart" => {
                self.session = identity;
                self.calls.clear();
            }
            "SessionEnd" => {
                self.session = None;
                self.calls.clear();
            }
            "PreToolUse" => {
                if let (Some(tool), Some(identity)) = (tool, identity) {
                    let command = event.payload.tool_input.and_then(|input| input.command);
                    let polylogue_shell = event.payload.tool_name.as_deref() == Some("Bash")
                        && command.as_deref().is_some_and(runs_polylogue);
                    self.calls.insert(
                        tool,
                        Call {
                            identity,
                            polylogue_shell,
                        },
                    );
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
fn runs_polylogue(command: &str) -> bool {
    polylogue_model::command::effective_stages(command)
        .iter()
        .any(|stage| stage.program == "polylogue")
}

#[cfg(test)]
mod tests;
