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
