use super::*;
use polylogue_model::live::{self, LiveFacts};
use serde_json::json;

const MIN: i64 = 60_000;

fn process(
    pid: i64,
    role: &str,
    argv: &[&str],
    ended: Option<i64>,
    call: Option<&str>,
) -> ProcessRow {
    ProcessRow {
        pid,
        parent_pid: Some(1),
        start: 7,
        comm: Some("x".into()),
        argv: argv.iter().map(|a| (*a).to_owned()).collect(),
        cwd: Some("/work/tide".into()),
        started_at: Some(MIN),
        ended_at: ended,
        role: role.into(),
        tool_call: call.map(str::to_owned),
    }
}

/// The live view shows the state with how long, the wait, the open call,
/// the task and each process with its command, age and tool call.
#[test]
fn the_live_card_shows_the_state_the_wait_and_the_process_tree() {
    let mut facts = LiveFacts::default();
    let hook = |extra: serde_json::Value| {
        let mut data = json!({"harness_pid": 10, "harness_start": 7, "boot_id": "b"});
        for (k, v) in extra.as_object().unwrap() {
            data[k] = v.clone();
        }
        data
    };
    live::fold(&mut facts, "hook_user_prompt_submit", 0, &hook(json!({})));
    live::fold(
        &mut facts,
        "hook_pre_tool_use",
        2 * MIN,
        &hook(json!({"tool_name": "Bash", "tool_use_id": "t1"})),
    );
    live::fold(
        &mut facts,
        "hook_permission_request",
        3 * MIN,
        &hook(json!({})),
    );
    live::fold(
        &mut facts,
        "hook_task_claimed",
        3 * MIN,
        &hook(json!({"task": "tide-12"})),
    );
    let now = 10 * MIN;
    let detail = LiveDetail {
        view: live::view(Some(&facts), None, now),
        facts,
        peers: vec!["codex:7c1e".into()],
        processes: vec![
            process(10, "harness", &["claude"], None, None),
            process(11, "child", &["cargo", "test"], None, Some("t1")),
            process(12, "child", &[], Some(4 * MIN), None),
        ],
    };
    let Card::Live {
        state,
        waits_on,
        call,
        task,
        peers,
        processes,
    } = render(&detail, now)
    else {
        panic!("a live card");
    };
    assert_eq!(state, ValueState::known("waiting for 7m".to_owned()));
    assert_eq!(waits_on, ValueState::known("permission".to_owned()));
    assert_eq!(call, ValueState::known("Bash for 8m".to_owned()));
    assert_eq!(task, ValueState::known("tide-12".to_owned()));
    assert_eq!(peers, vec!["codex:7c1e".to_owned()]);
    let lines: Vec<(String, ValueState<String>, ValueState<String>)> = processes
        .into_iter()
        .map(|p| (p.reference, p.command, p.state))
        .collect();
    assert_eq!(
        lines,
        vec![
            (
                "harness pid 10".to_owned(),
                ValueState::known("claude".to_owned()),
                ValueState::known("running for 9m · in /work/tide".to_owned())
            ),
            (
                "child pid 11".to_owned(),
                ValueState::known("cargo test".to_owned()),
                ValueState::known("running for 9m · tool call t1 · in /work/tide".to_owned())
            ),
            (
                "child pid 12".to_owned(),
                ValueState::known("x".to_owned()),
                ValueState::known("ended 6m ago · in /work/tide".to_owned())
            ),
        ]
    );
}
