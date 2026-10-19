//! Claude Code's side files: what the tool keeps beside its transcripts.
//! Each file is read whole and becomes events on the session its path or
//! fields name, citing the file's bytes, which the blob store keeps as the
//! input. Files that name no session (plans, project memory) are stored
//! and accounted with the reason; sessions name their plan by `slug`
//! (a `plan` event from the transcript).

use super::{ROOT, session_id};
use crate::account::{Accounting, Disposition};
use crate::document::Sink;
use crate::frame::Record;
use crate::origin::Root;
use crate::parse::{DocumentParser, Output, Parser, StepParser, Whole};
use crate::time;
use polylogue_model::{Event, Json, Link, LinkKind, Session, SessionKind, Timestamp, TitleSource};
use std::path::Path;

pub const SIDE_ROOT: Root = ROOT;

/// Every side file layout, relative to the tool's directory.
pub const SIDE_LOCATIONS: &[&str] = &[
    "todos/*.json",
    "tasks/*/*.json",
    "projects/*/*/subagents/agent-*.meta.json",
    "projects/*/*/subagents/workflows/*/agent-*.meta.json",
    "jobs/*/state.json",
    "jobs/*/timeline.jsonl",
    "file-history/*/*",
    "projects/*/*/tool-results/*",
    "plans/*.md",
    "projects/*/memory/*",
];

pub const HISTORY_LOCATIONS: &[&str] = &["history.jsonl"];

/// Which side file a path is, from its place in the tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Side {
    /// `todos/<session>.json` or `todos/<session>-agent-<agent>.json`.
    Todos {
        session: String,
        agent: Option<String>,
    },
    /// `tasks/<session>/<n>.json`: one task of the task list.
    Task { session: String },
    /// A subagent's `agent-<id>.meta.json`: its type, description and
    /// the call that spawned it; `session` is the directory above
    /// `subagents`, the spawning session unless `parentAgentId` names one.
    Meta {
        stem: String,
        session: Option<String>,
    },
    /// `jobs/<short>/state.json`: a background session's job.
    JobState,
    /// `jobs/<short>/timeline.jsonl`: the job's state changes.
    JobTimeline,
    /// `file-history/<session>/<hash>@v<n>`: a file's content before an edit.
    FileSnapshot { session: String, name: String },
    /// `projects/<project>/<session>/tool-results/<ref>`: a tool output too
    /// large for the transcript, named by its call or background task.
    ToolResult { session: String, name: String },
    /// `plans/<slug>.md`.
    Plan,
    /// `projects/<project>/memory/<file>`.
    Memory,
}

impl Side {
    pub fn from_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?;
        let dirs: Vec<&str> = path
            .ancestors()
            .skip(1)
            .filter_map(|p| p.file_name()?.to_str())
            .take(4)
            .collect();
        let dir = |i: usize| dirs.get(i).copied();
        if name.starts_with("agent-") && dirs.contains(&"subagents") {
            let at = dirs.iter().position(|d| *d == "subagents")?;
            return Some(Side::Meta {
                stem: name.strip_suffix(".meta.json")?.to_owned(),
                session: dir(at + 1).map(str::to_owned),
            });
        }
        match (dir(0), dir(1)) {
            (Some("todos"), _) => {
                let stem = name.strip_suffix(".json")?;
                Some(match stem.split_once("-agent-") {
                    Some((session, agent)) if agent != session => Side::Todos {
                        session: session.to_owned(),
                        agent: Some(agent.to_owned()),
                    },
                    Some((session, _)) => Side::Todos {
                        session: session.to_owned(),
                        agent: None,
                    },
                    None => Side::Todos {
                        session: stem.to_owned(),
                        agent: None,
                    },
                })
            }
            (Some(session), Some("tasks")) if name.ends_with(".json") => Some(Side::Task {
                session: session.to_owned(),
            }),
            (Some(_), Some("jobs")) if name == "state.json" => Some(Side::JobState),
            (Some(_), Some("jobs")) if name == "timeline.jsonl" => Some(Side::JobTimeline),
            (Some(session), Some("file-history")) => Some(Side::FileSnapshot {
                session: session.to_owned(),
                name: name.to_owned(),
            }),
            (Some("tool-results"), Some(session)) => Some(Side::ToolResult {
                session: session.to_owned(),
                name: name.to_owned(),
            }),
            (Some("plans"), _) if name.ends_with(".md") => Some(Side::Plan),
            (Some("memory"), Some(_)) if dir(2) == Some("projects") => Some(Side::Memory),
            _ => None,
        }
    }
}

pub fn recognize_side(path: &Path, _head: &[u8]) -> bool {
    Side::from_path(path).is_some()
}

pub fn open_side(path: &Path) -> Option<Box<dyn StepParser>> {
    Side::from_path(path).map(|side| Box::new(Whole(SideFile { side })) as Box<dyn StepParser>)
}

pub fn recognize_history(path: &Path, head: &[u8]) -> bool {
    path.file_name().is_some_and(|n| n == "history.jsonl")
        && head
            .iter()
            .find(|b| !b.is_ascii_whitespace())
            .is_none_or(|&b| b == b'{')
}

pub fn open_history(path: &Path) -> Option<Box<dyn StepParser>> {
    recognize_history(path, b"").then(|| Box::new(History) as Box<dyn StepParser>)
}

struct SideFile {
    side: Side,
}

/// One event on `session` citing the whole file.
fn whole(session: &str, kind: &str, data: Json, len: usize, out: &mut Output) -> Disposition {
    let Some(session) = session_id(session) else {
        return Disposition::refused("session ID in the path is empty");
    };
    out.events.push(Event {
        session,
        kind: kind.to_owned(),
        at: None,
        after: None,
        range: Some((0, len as u64)),
        data,
    });
    Disposition::Materialized
}

fn json(bytes: &[u8]) -> Result<Json, Disposition> {
    serde_json::from_slice(bytes).map_err(|e| Disposition::refused(format!("malformed JSON: {e}")))
}

impl DocumentParser for SideFile {
    fn document(&self, bytes: &[u8], out: &mut Output, acc: &mut Accounting) {
        let mut sink = Sink::new(out, acc);
        let len = bytes.len();
        let (kind, disposition) = match &self.side {
            Side::Todos { session, agent } => (
                "todos",
                match json(bytes) {
                    // The list belongs to the agent that kept it: the
                    // subagent's own session when one did.
                    Ok(v) => {
                        let owner = agent
                            .as_deref()
                            .map(|a| format!("agent-{a}"))
                            .unwrap_or_else(|| session.clone());
                        whole(&owner, "todo_list", v, len, sink.out)
                    }
                    Err(d) => d,
                },
            ),
            Side::Task { session } => (
                "task",
                match json(bytes) {
                    Ok(v) => whole(session, "task", v, len, sink.out),
                    Err(d) => d,
                },
            ),
            // Type and description only, as an event: the transcript owns
            // the session's kind and title, so a meta file read before or
            // after it never changes them (architecture 5.5a merge order).
            Side::Meta { stem, session } => (
                "subagent_meta",
                match json(bytes) {
                    Ok(v) => {
                        spawn_link(stem, session.as_deref(), &v, sink.out);
                        whole(stem, "subagent_meta", v, len, sink.out)
                    }
                    Err(d) => d,
                },
            ),
            Side::JobState => ("job_state", job_state(bytes, len, sink.out)),
            Side::JobTimeline => (
                "job_timeline",
                Disposition::unknown(
                    "a job's state changes; its state.json names the session, the bytes stay stored",
                ),
            ),
            Side::FileSnapshot { session, name } => {
                let (hash, version) = name.split_once("@v").unwrap_or((name, ""));
                let data = serde_json::json!({
                    "backup": name, "path_hash": hash, "version": version, "size": len,
                });
                (
                    "file_snapshot",
                    whole(session, "file_snapshot", data, len, sink.out),
                )
            }
            Side::ToolResult { session, name } => {
                // The file is named by the tool call (`toolu_…`, `call_…`)
                // or the background task (`b…`) whose output it holds.
                let reference = name.rsplit_once('.').map_or(name.as_str(), |(s, _)| s);
                let data = serde_json::json!({ "file": name, "ref": reference, "size": len });
                (
                    "tool_result_file",
                    whole(session, "tool_result_file", data, len, sink.out),
                )
            }
            Side::Plan => (
                "plan",
                Disposition::unknown(
                    "a plan names no session; sessions name their plan by slug (plan event)",
                ),
            ),
            Side::Memory => (
                "memory",
                Disposition::unknown(
                    "project memory belongs to the project directory, not a session",
                ),
            ),
        };
        sink.record(kind, &disposition);
    }
}

/// A background job: the session it ran as, how it was named, where it
/// ran and which session it resumed.
fn job_state(bytes: &[u8], len: usize, out: &mut Output) -> Disposition {
    let v = match json(bytes) {
        Ok(v) => v,
        Err(d) => return d,
    };
    let Some(sid) = v
        .get("sessionId")
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
    else {
        return Disposition::refused("job state without a sessionId");
    };
    let sid = sid.as_str();
    let Some(id) = session_id(sid) else {
        return Disposition::refused("session ID is not valid");
    };
    let mut s = Session::new(id.clone(), SessionKind::Main);
    s.mode = Some("bg".to_owned());
    if let Some(name) = v.get("name").and_then(Json::as_str) {
        let source = match v.get("nameSource").and_then(Json::as_str) {
            Some("user") => TitleSource::Custom,
            _ => TitleSource::Generated,
        };
        s.offer_title(name, source);
    }
    if let Some(cwd) = v.get("cwd").and_then(Json::as_str) {
        s.offer_cwd(cwd);
    }
    if let Some(at) = v.get("createdAt").and_then(time::rfc3339) {
        s.offer_start(at);
    }
    if let Some(version) = v.get("cliVersion").and_then(Json::as_str) {
        s.tool_version = Some(version.to_owned());
    }
    out.sessions.push(s);
    if let Some(parent) = v
        .get("resumeSessionId")
        .and_then(Json::as_str)
        .filter(|p| *p != sid)
        .and_then(session_id)
    {
        out.links.push(Link {
            kind: LinkKind::Continuation,
            parent,
            child: id,
            at: None,
            call_id: None,
            evidence: "resumeSessionId".to_owned(),
        });
    }
    whole(sid, "background_job", v, len, out)
}

/// The spawn link a subagent's meta file states: the call that started it,
/// in the parent agent it names or else the session it was filed under.
fn spawn_link(stem: &str, session: Option<&str>, v: &Json, out: &mut Output) {
    let Some(call) = v.get("toolUseId").and_then(Json::as_str) else {
        return;
    };
    let parent = match v.get("parentAgentId").and_then(Json::as_str) {
        Some(agent) => session_id(&format!("agent-{agent}")),
        None => session.and_then(session_id),
    };
    let (Some(parent), Some(child)) = (parent, session_id(stem)) else {
        return;
    };
    if parent == child {
        return;
    }
    out.links.push(Link {
        kind: LinkKind::Spawn,
        parent,
        child,
        at: None,
        call_id: Some(call.to_owned()),
        evidence: "toolUseId".to_owned(),
    });
}

/// `history.jsonl`: every prompt the operator typed, with its project and
/// time. Newer lines name their session.
struct History;

impl Parser for History {
    type Carry = ();

    fn kind(&self, _value: &Json) -> String {
        "prompt".to_owned()
    }

    fn record(
        &self,
        _carry: &mut (),
        record: Record<'_>,
        v: &Json,
        out: &mut Output,
        _acc: &mut Accounting,
    ) -> Disposition {
        let Some(sid) = v
            .get("sessionId")
            .and_then(Json::as_str)
            .filter(|s| !s.is_empty())
        else {
            return Disposition::unknown(
                "written before the tool recorded session IDs; project and time stay in the stored bytes",
            );
        };
        let Some(session) = session_id(sid) else {
            return Disposition::refused("session ID is not valid");
        };
        out.events.push(Event {
            session,
            kind: "prompt_history".to_owned(),
            at: v
                .get("timestamp")
                .and_then(Json::as_i64)
                .and_then(Timestamp::from_unix_ms),
            after: None,
            range: Some(super::range(record)),
            data: v.clone(),
        });
        Disposition::Materialized
    }

    fn flush(&self, _carry: &mut (), _out: &mut Output) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{run, run_document};

    fn side(path: &str) -> Side {
        Side::from_path(Path::new(path)).unwrap_or_else(|| panic!("{path}"))
    }

    #[test]
    fn paths_name_their_side_file() {
        assert_eq!(
            side("/h/.claude/todos/s1-agent-a2.json"),
            Side::Todos {
                session: "s1".into(),
                agent: Some("a2".into())
            }
        );
        assert_eq!(
            side("/h/.claude/todos/s1-agent-s1.json"),
            Side::Todos {
                session: "s1".into(),
                agent: None
            }
        );
        assert_eq!(
            side("/h/.claude/tasks/s1/3.json"),
            Side::Task {
                session: "s1".into()
            }
        );
        assert_eq!(
            side("/h/.claude/projects/p/s1/subagents/agent-x.meta.json"),
            Side::Meta {
                stem: "agent-x".into(),
                session: Some("s1".into())
            }
        );
        assert_eq!(
            side("/h/.claude/projects/p/s1/subagents/workflows/r/agent-y.meta.json"),
            Side::Meta {
                stem: "agent-y".into(),
                session: Some("s1".into())
            }
        );
        assert_eq!(side("/h/.claude/jobs/60eac085/state.json"), Side::JobState);
        assert_eq!(
            side("/h/.claude/jobs/60eac085/timeline.jsonl"),
            Side::JobTimeline
        );
        assert_eq!(
            side("/h/.claude/file-history/s1/abc@v2"),
            Side::FileSnapshot {
                session: "s1".into(),
                name: "abc@v2".into()
            }
        );
        assert_eq!(
            side("/h/.claude/projects/p/s1/tool-results/toolu_1.txt"),
            Side::ToolResult {
                session: "s1".into(),
                name: "toolu_1.txt".into()
            }
        );
        assert_eq!(side("/h/.claude/plans/quiet-river.md"), Side::Plan);
        assert_eq!(side("/h/.claude/projects/p/memory/MEMORY.md"), Side::Memory);
        assert!(Side::from_path(Path::new("/h/.claude/projects/p/s1.jsonl")).is_none());
    }

    fn doc(path: &str, bytes: &[u8]) -> Output {
        let step = run_document(&SideFile { side: side(path) }, bytes).unwrap();
        assert_eq!(step.accounting.framed, 1);
        step.output
    }

    #[test]
    fn a_subagent_todo_list_belongs_to_the_subagent() {
        let out = doc(
            "/h/.claude/todos/s1-agent-a2.json",
            br#"[{"content":"x","status":"pending"}]"#,
        );
        assert_eq!(out.events[0].session.to_string(), "claude-code:agent-a2");
        assert_eq!(out.events[0].kind, "todo_list");
        assert_eq!(out.events[0].range, Some((0, 36)));
    }

    #[test]
    fn meta_is_an_event_on_the_subagent() {
        let out = doc(
            "/h/.claude/projects/p/s1/subagents/agent-x.meta.json",
            br#"{"agentType":"Explore","description":"Find callers"}"#,
        );
        assert!(out.sessions.is_empty());
        assert_eq!(out.events[0].session.to_string(), "claude-code:agent-x");
        assert_eq!(out.events[0].data["agentType"], "Explore");
        assert!(out.links.is_empty(), "no toolUseId, no link");
    }

    #[test]
    fn meta_links_the_subagent_to_the_call_that_spawned_it() {
        let out = doc(
            "/h/.claude/projects/p/s1/subagents/agent-x.meta.json",
            br#"{"agentType":"Explore","toolUseId":"toolu_1"}"#,
        );
        let l = &out.links[0];
        assert_eq!(l.kind, LinkKind::Spawn);
        assert_eq!(l.parent.to_string(), "claude-code:s1");
        assert_eq!(l.child.to_string(), "claude-code:agent-x");
        assert_eq!(l.call_id.as_deref(), Some("toolu_1"));
        // A nested agent's parent is the agent that spawned it.
        let out = doc(
            "/h/.claude/projects/p/s1/subagents/workflows/r/agent-y.meta.json",
            br#"{"toolUseId":"toolu_2","parentAgentId":"x"}"#,
        );
        assert_eq!(out.links[0].parent.to_string(), "claude-code:agent-x");
    }

    #[test]
    fn a_job_names_its_session_and_the_one_it_resumed() {
        let out = doc(
            "/h/.claude/jobs/60eac085/state.json",
            br#"{"sessionId":"s2","resumeSessionId":"s1","name":"Grok","nameSource":"user","cwd":"/r","cliVersion":"2.1"}"#,
        );
        assert_eq!(out.sessions[0].mode.as_deref(), Some("bg"));
        assert_eq!(out.links[0].kind, LinkKind::Continuation);
        assert_eq!(out.links[0].parent.to_string(), "claude-code:s1");
        assert_eq!(out.events[0].kind, "background_job");
    }

    #[test]
    fn tool_results_and_snapshots_cite_their_bytes() {
        let out = doc(
            "/h/.claude/projects/p/s1/tool-results/toolu_9.txt",
            b"big output",
        );
        assert_eq!(out.events[0].data["ref"], "toolu_9");
        let out = doc("/h/.claude/file-history/s1/ab12@v3", b"old text");
        assert_eq!(out.events[0].data["path_hash"], "ab12");
        assert_eq!(out.events[0].data["version"], "3");
    }

    #[test]
    fn plans_and_memory_are_kept_with_a_reason() {
        let step = run_document(&SideFile { side: Side::Plan }, b"# plan").unwrap();
        assert!(step.output.events.is_empty());
        assert_eq!(step.accounting.framed, 1);
    }

    #[test]
    fn malformed_json_is_refused_not_dropped() {
        let step = run_document(
            &SideFile {
                side: Side::Task {
                    session: "s1".into(),
                },
            },
            b"{not json",
        )
        .unwrap();
        assert_eq!(step.accounting.framed, 1);
        assert!(step.output.events.is_empty());
    }

    #[test]
    fn history_lines_with_a_session_become_events() {
        let bytes = b"{\"display\":\"a\",\"timestamp\":1760892079968,\"project\":\"/p\"}\n{\"display\":\"b\",\"timestamp\":1760892079969,\"project\":\"/p\",\"sessionId\":\"s1\"}\n";
        let step = run(&History, &mut (), bytes, 0).unwrap();
        assert_eq!(step.accounting.framed, 2);
        assert_eq!(step.output.events.len(), 1);
        assert_eq!(step.output.events[0].kind, "prompt_history");
        assert!(step.output.events[0].at.is_some());
    }
}
