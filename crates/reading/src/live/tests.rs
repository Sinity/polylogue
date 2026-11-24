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
    live::apply_live_event(&mut facts, "hook_user_prompt_submit", 0, &hook(json!({})));
    live::apply_live_event(
        &mut facts,
        "hook_pre_tool_use",
        2 * MIN,
        &hook(json!({"tool_name": "Bash", "tool_use_id": "t1"})),
    );
    live::apply_live_event(
        &mut facts,
        "hook_permission_request",
        3 * MIN,
        &hook(json!({})),
    );
    live::apply_live_event(
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
    } = render(&detail, now, &jiff::tz::TimeZone::UTC)
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

/// Absent facts are stated as unknown with their reason, a gone process and
/// the harness's own word are appended to the state, a wait outranks a stall,
/// and a process with no argv and no name says so.
#[test]
fn the_live_card_states_what_is_absent() {
    let now = 10 * MIN;
    let mut detail = LiveDetail {
        view: live::view(None, None, now),
        facts: LiveFacts::default(),
        peers: vec![],
        processes: vec![ProcessRow {
            comm: None,
            ..process(20, "child", &[], None, None)
        }],
    };
    detail.view.process_alive = Some(false);
    detail.view.stall = Some(live::Stall::RateLimit);
    detail.facts.harness_status = Some(live::HarnessStatus {
        word: "busy".into(),
        at: 0,
    });
    let Card::Live {
        state,
        waits_on,
        call,
        task,
        processes,
        ..
    } = render(&detail, now, &jiff::tz::TimeZone::UTC)
    else {
        panic!("a live card");
    };
    assert_eq!(
        state,
        ValueState::known("unknown · process gone · harness says busy".to_owned())
    );
    assert_eq!(waits_on, ValueState::known("stall: rate_limit".to_owned()));
    let unknown = |reason: &str| ValueState::Unknown {
        reason: Some(reason.into()),
    };
    assert_eq!(call, unknown("no tool call open"));
    assert_eq!(task, unknown("no stated task"));
    assert_eq!(processes[0].command, unknown("no command recorded"));

    detail.view.waiting_on = Some(live::WaitingOn::Question);
    let Card::Live { waits_on, .. } = render(&detail, now, &jiff::tz::TimeZone::UTC) else {
        panic!("a live card");
    };
    assert_eq!(waits_on, ValueState::known("question".to_owned()));
}

/// A stale session says why it is stale, with the time the process went.
#[test]
fn a_stale_card_names_when_the_process_went_and_that_no_end_came() {
    let mut facts = LiveFacts::default();
    let data = json!({"harness_pid": 10, "harness_start": 7, "boot_id": "b"});
    let at = |hour: i64, minute: i64| (hour * 60 + minute) * MIN;
    live::apply_live_event(&mut facts, "hook_user_prompt_submit", at(13, 0), &data);
    live::apply_live_event(
        &mut facts,
        "process_root_gone",
        at(13, 40),
        &json!({"pid": 10, "start": 7, "boot_id": "b"}),
    );
    let now = at(14, 0);
    let detail = LiveDetail {
        view: live::view(Some(&facts), None, now),
        facts,
        peers: vec![],
        processes: vec![],
    };
    let Card::Live { state, .. } = render(&detail, now, &jiff::tz::TimeZone::UTC) else {
        panic!("a live card");
    };
    assert_eq!(
        state,
        ValueState::known("stale · process gone at 13:40, no end event".to_owned())
    );
}
