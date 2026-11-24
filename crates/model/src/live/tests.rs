use super::*;
use serde_json::json;

const MIN: i64 = 60_000;

fn identity(pid: u32, start: u64, boot: &str) -> Json {
    json!({"harness_pid": pid, "harness_start": start, "boot_id": boot})
}

/// A hook line's data: the payload plus the process identity triple.
fn hook(payload: Json) -> Json {
    let mut data = identity(100, 5000, "boot-a");
    for (k, v) in payload.as_object().unwrap() {
        data[k] = v.clone();
    }
    data
}

fn run(lines: &[(&str, i64, Json)]) -> (LiveFacts, Vec<LiveInterval>) {
    let mut facts = LiveFacts::default();
    let mut closed = Vec::new();
    for (kind, at, data) in lines {
        closed.extend(apply_live_event(&mut facts, kind, *at, data));
    }
    (facts, closed)
}

fn state(facts: &LiveFacts, now: i64) -> LiveState {
    view(Some(facts), None, now).state
}

#[test]
fn the_view_is_total_and_constant_between_until_instants() {
    let lines = [
        ("hook_session_start", 0, hook(json!({}))),
        ("hook_user_prompt_submit", MIN, hook(json!({}))),
        (
            "hook_pre_tool_use",
            2 * MIN,
            hook(json!({"tool_name": "Bash", "tool_use_id": "t1"})),
        ),
    ];
    let (facts, _) = run(&lines);
    // Every instant gives a view; between `until` instants nothing changes.
    let mut now = 2 * MIN;
    let mut seen = Vec::new();
    while now < 3 * 60 * MIN {
        let v = view(Some(&facts), None, now);
        let next = v.until.unwrap_or(now + 3 * 60 * MIN);
        assert!(next > now, "until lies after now");
        for probe in [now, (now + next) / 2, next - 1] {
            assert_eq!(view(Some(&facts), None, probe), v);
        }
        seen.push((v.state, v.stall));
        now = next;
    }
    // A call open past the long-call threshold becomes a stall, not a wait.
    assert_eq!(
        seen,
        vec![
            (LiveState::Working, None),
            (LiveState::Working, Some(Stall::LongCall))
        ]
    );
    // No event and no time is unknown; growth alone is unknown for a
    // window and ended after it.
    assert_eq!(view(None, None, 0).state, LiveState::Unknown);
    let grew = view(None, Some(10 * MIN), 20 * MIN);
    assert_eq!(grew.state, LiveState::Unknown);
    assert_eq!(
        view(None, Some(10 * MIN), grew.until.unwrap()).state,
        LiveState::Ended
    );
}

#[test]
fn a_rate_limit_stop_is_a_stall_and_never_waiting() {
    let (facts, _) = run(&[
        ("hook_user_prompt_submit", 0, hook(json!({}))),
        (
            "hook_stop_failure",
            MIN,
            hook(json!({"error_type": "rate_limit"})),
        ),
    ]);
    let v = view(Some(&facts), None, 2 * MIN);
    assert_eq!(v.state, LiveState::Idle);
    assert_eq!(v.waiting_on, None);
    assert_eq!(v.stall, Some(Stall::RateLimit));
    // A permission request is waiting, with what it waits for.
    let (facts, _) = run(&[
        ("hook_user_prompt_submit", 0, hook(json!({}))),
        (
            "hook_permission_request",
            MIN,
            hook(json!({"tool_name": "Bash"})),
        ),
    ]);
    let v = view(Some(&facts), None, 2 * MIN);
    assert_eq!(
        (v.state, v.waiting_on),
        (LiveState::Waiting, Some(WaitingOn::Permission))
    );
    // An idle notification is not a wait.
    let (facts, _) = run(&[
        ("hook_stop", 0, hook(json!({}))),
        (
            "hook_notification",
            MIN,
            hook(json!({"notification_type": "idle_prompt"})),
        ),
    ]);
    assert_eq!(state(&facts, 2 * MIN), LiveState::Idle);
}

#[test]
fn the_registry_word_outranks_hook_inference() {
    let (facts, _) = run(&[
        ("hook_user_prompt_submit", 0, hook(json!({}))),
        ("hook_stop", MIN, hook(json!({}))),
        (
            "registry_status",
            2 * MIN,
            json!({"status": "busy", "pid": 100, "start": 5000, "boot_id": "boot-a"}),
        ),
        ("hook_stop", 3 * MIN, hook(json!({}))),
    ]);
    // Both events stay in the facts; the registry decides.
    assert_eq!(facts.hook_activity.unwrap().value, Activity::Idle);
    assert_eq!(facts.harness_status.as_ref().unwrap().word, "busy");
    assert_eq!(state(&facts, 4 * MIN), LiveState::Working);
    // `blocked` is waiting on a permission.
    let mut blocked = facts.clone();
    apply_live_event(
        &mut blocked,
        "registry_status",
        5 * MIN,
        &json!({"status": "blocked", "pid": 100, "start": 5000, "boot_id": "boot-a"}),
    );
    let v = view(Some(&blocked), None, 6 * MIN);
    assert_eq!(
        (v.state, v.waiting_on),
        (LiveState::Waiting, Some(WaitingOn::Permission))
    );
}

#[test]
fn a_line_about_another_process_never_attaches() {
    let start = [
        ("hook_session_start", 0, hook(json!({}))),
        ("hook_user_prompt_submit", MIN, hook(json!({}))),
    ];
    // Same pid, different start time: a different process.
    let (mut facts, _) = run(&start);
    apply_live_event(
        &mut facts,
        "process_root_gone",
        2 * MIN,
        &json!({"pid": 100, "start": 9999, "boot_id": "boot-a"}),
    );
    assert_eq!(facts.root_gone_at, None);
    // Same pid and start, another boot.
    apply_live_event(
        &mut facts,
        "process_root_gone",
        2 * MIN,
        &json!({"pid": 100, "start": 5000, "boot_id": "boot-b"}),
    );
    assert_eq!(facts.root_gone_at, None);
    // A registry line for a reused pid says nothing about this session.
    apply_live_event(
        &mut facts,
        "registry_status",
        2 * MIN,
        &json!({"status": "idle", "pid": 100, "start": 7, "boot_id": "boot-a"}),
    );
    assert_eq!(facts.harness_status, None);
    // The root itself.
    apply_live_event(
        &mut facts,
        "process_root_gone",
        3 * MIN,
        &json!({"pid": 100, "start": 5000, "boot_id": "boot-a"}),
    );
    assert_eq!(facts.root_gone_at, Some(3 * MIN));
    let v = view(Some(&facts), None, 4 * MIN);
    assert_eq!((v.state, v.process_alive), (LiveState::Stale, Some(false)));
}

#[test]
fn no_fact_comes_from_prose_or_command_text() {
    let (facts, _) = run(&[
        (
            "hook_post_tool_use",
            0,
            hook(
                json!({"tool_name": "Bash", "tool_input": {"command": "sed -i x file; bd update sx-1 --claim"}}),
            ),
        ),
        (
            "hook_user_prompt_submit",
            MIN,
            hook(json!({"prompt": "I am waiting on you: claim sx-2, edit file"})),
        ),
        (
            "hook_stop",
            2 * MIN,
            hook(json!({"last_assistant_message": "Shall I continue?"})),
        ),
    ]);
    assert!(facts.edited_files.is_empty());
    assert_eq!(facts.task, None);
    assert_eq!(state(&facts, 3 * MIN), LiveState::Idle);
    // A structured edit adds its path; an explicit task event sets the task.
    let (facts, _) = run(&[
        (
            "hook_post_tool_use",
            0,
            hook(json!({"tool_name": "Edit", "tool_input": {"file_path": "/home/op/src/file"}})),
        ),
        (
            "hook_post_tool_use",
            MIN,
            hook(
                json!({"tool_name": "apply_patch", "tool_input": {"input": "*** Begin Patch\n*** Update File: src/a.rs\n@@\n*** Add File: src/b.rs\n*** End Patch"}}),
            ),
        ),
        ("hook_task_claimed", 2 * MIN, hook(json!({"task": "sx-1"}))),
    ]);
    assert_eq!(
        facts
            .edited_files
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["/home/op/src/file", "src/a.rs", "src/b.rs"]
    );
    assert_eq!(facts.task.as_deref(), Some("sx-1"));
}

#[test]
fn intervals_close_on_live_events() {
    let lines = vec![
        ("hook_session_start", 0, hook(json!({}))),
        ("hook_user_prompt_submit", MIN, hook(json!({}))),
        ("hook_permission_request", 2 * MIN, hook(json!({}))),
        ("hook_post_tool_use", 9 * MIN, hook(json!({}))),
        ("hook_stop", 10 * MIN, hook(json!({}))),
    ];
    let (facts, closed) = run(&lines);
    let compact: Vec<_> = closed
        .iter()
        .map(|i| (i.aspect, i.value.as_str(), i.start / MIN, i.end / MIN))
        .collect();
    assert_eq!(
        compact,
        vec![
            (Aspect::Activity, "idle", 0, 1),
            (Aspect::Activity, "working", 1, 2),
            (Aspect::Activity, "waiting", 2, 9),
            (Aspect::WaitingOn, "permission", 2, 9),
            (Aspect::Activity, "working", 9, 10),
        ]
    );
    // The last interval stays open however late it is asked.
    assert_eq!(facts.open_intervals.len(), 1);
    assert_eq!(facts.open_intervals[0].start, 10 * MIN);
    // The root's exit closes every open interval.
    let mut gone = facts.clone();
    let last = apply_live_event(
        &mut gone,
        "process_root_gone",
        11 * MIN,
        &json!({"pid": 100, "start": 5000, "boot_id": "boot-a"}),
    );
    assert_eq!(last.len(), 1);
    assert_eq!(last[0].closed_by, "process_root_gone");
    assert!(gone.open_intervals.is_empty());
    // Gone after a stop is an end.
    assert_eq!(state(&gone, 12 * MIN), LiveState::Ended);
}

#[test]
fn applying_stored_facts_matches_applying_at_once() {
    let lines = vec![
        ("hook_session_start", 0, hook(json!({}))),
        (
            "hook_pre_tool_use",
            MIN,
            hook(
                json!({"tool_name": "TodoWrite", "tool_input": {"todos": [{"status": "in_progress", "content": "port cases"}]}}),
            ),
        ),
        (
            "registry_status",
            2 * MIN,
            json!({"status": "busy", "pid": 100, "start": 5000, "boot_id": "boot-a"}),
        ),
        ("hook_subagent_start", 3 * MIN, hook(json!({}))),
        ("hook_session_end", 4 * MIN, hook(json!({}))),
    ];
    let (whole, whole_closed) = run(&lines);
    // The store keeps facts as JSON between lines.
    let mut facts = LiveFacts::default();
    let mut closed = Vec::new();
    for (kind, at, data) in &lines {
        let stored = serde_json::to_string(&facts).unwrap();
        facts = serde_json::from_str(&stored).unwrap();
        closed.extend(apply_live_event(&mut facts, kind, *at, data));
    }
    assert_eq!((facts, closed), (whole, whole_closed));
}

/// Without process events, hook events hold for the live window only.
#[test]
fn hook_events_without_a_process_hold_for_the_window() {
    let (facts, _) = run(&[(
        "hook_notification",
        0,
        json!({"notification_type": "permission_prompt"}),
    )]);
    let v = view(Some(&facts), None, 5 * MIN);
    assert_eq!((v.state, v.process_alive), (LiveState::Waiting, None));
    assert_eq!(v.until, Some(LIVE_WINDOW_MS + 1));
    assert_eq!(state(&facts, LIVE_WINDOW_MS + 1), LiveState::Stale);
    // With a live process, no time passing changes the state.
    let (facts, _) = run(&[("hook_stop", 0, hook(json!({})))]);
    let v = view(Some(&facts), None, 300 * MIN);
    assert_eq!((v.state, v.until), (LiveState::Idle, None));
}

/// Events passing through a value at one instant leave no interval, so
/// no two intervals of one aspect share a start.
#[test]
fn an_instant_leaves_no_interval() {
    let (_, closed) = run(&[
        ("hook_user_prompt_submit", 0, hook(json!({}))),
        ("hook_permission_request", MIN, hook(json!({}))),
        ("hook_post_tool_use", MIN, hook(json!({}))),
        ("hook_stop", 2 * MIN, hook(json!({}))),
    ]);
    let compact: Vec<_> = closed
        .iter()
        .map(|i| (i.aspect, i.value.as_str(), i.start / MIN, i.end / MIN))
        .collect();
    assert_eq!(
        compact,
        vec![
            (Aspect::Activity, "working", 0, 1),
            (Aspect::Activity, "working", 1, 2),
        ]
    );
}

/// A stale session ages out: a day after its latest event it is ended,
/// whether its process was seen gone mid-turn or it never had process
/// events; the stale view says when.
#[test]
fn stale_sessions_end_a_day_after_their_latest_event() {
    let (gone, _) = run(&[
        ("hook_user_prompt_submit", 0, hook(json!({}))),
        (
            "process_root_gone",
            MIN,
            json!({"pid": 100, "start": 5000, "boot_id": "boot-a"}),
        ),
    ]);
    let v = view(Some(&gone), None, 2 * MIN);
    assert_eq!(
        (v.state, v.since, v.until),
        (LiveState::Stale, Some(MIN), Some(MIN + STALE_FOR_MS))
    );
    assert_eq!(state(&gone, MIN + STALE_FOR_MS - 1), LiveState::Stale);
    let aged = view(Some(&gone), None, MIN + STALE_FOR_MS);
    assert_eq!(
        (aged.state, aged.since, aged.until),
        (LiveState::Ended, Some(MIN), None)
    );
    let (unobserved, _) = run(&[("hook_user_prompt_submit", 0, json!({}))]);
    let v = view(Some(&unobserved), None, LIVE_WINDOW_MS + 1);
    assert_eq!((v.state, v.until), (LiveState::Stale, Some(STALE_FOR_MS)));
    assert_eq!(state(&unobserved, STALE_FOR_MS), LiveState::Ended);
}

/// A hosted session's lines come from the host that started its process:
/// they name the root, state its activity and, when the process exits after
/// its turn, end the session. A plain registry line still never chooses a
/// root.
#[test]
fn hosted_lines_name_the_root_and_end_the_session() {
    let process = json!({"pid": 300, "start": 77, "boot_id": "boot-a"});
    let status = |word: &str| {
        let mut data = process.clone();
        data["status"] = json!(word);
        data
    };
    let (facts, _) = run(&[("registry_hosted_status", 0, status("busy"))]);
    assert_eq!(
        facts.root,
        Some(ProcessIdentity {
            boot: "boot-a".into(),
            pid: 300,
            start: 77
        })
    );
    assert_eq!(state(&facts, MIN), LiveState::Working);
    let (facts, _) = run(&[
        ("registry_hosted_status", 0, status("busy")),
        ("registry_hosted_status", MIN, status("blocked")),
    ]);
    assert_eq!(state(&facts, 2 * MIN), LiveState::Waiting);
    let (facts, closed) = run(&[
        ("registry_hosted_status", 0, status("busy")),
        ("registry_hosted_status", MIN, status("idle")),
        ("registry_hosted_ended", 2 * MIN, process.clone()),
    ]);
    assert_eq!(state(&facts, 3 * MIN), LiveState::Ended);
    assert_eq!(
        closed
            .iter()
            .map(|i| (
                i.aspect,
                i.value.as_str(),
                i.start,
                i.end,
                i.closed_by.as_str()
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                Aspect::Activity,
                "working",
                0,
                MIN,
                "registry_hosted_status"
            ),
            (
                Aspect::Activity,
                "idle",
                MIN,
                2 * MIN,
                "registry_hosted_ended"
            ),
        ]
    );
    // A reader's line names no root by itself.
    let (facts, _) = run(&[("registry_status", 0, status("busy"))]);
    assert_eq!(facts.root, None);
}

/// A message waiting behind a hosted session's running turn is a queue
/// stall, shown while the turn runs; it ends when the count returns to
/// zero, the session goes idle or it ends, and it never counts as waiting.
#[test]
fn a_session_with_messages_queued_behind_its_turn_shows_the_queue_stall() {
    let process = json!({"pid": 300, "start": 77, "boot_id": "boot-a"});
    let mut busy = process.clone();
    busy["status"] = json!("busy");
    let queued = |n: u32| json!({"queued": n});
    let stall_at = |events: &[(&str, i64, Json)], now: i64| {
        let (facts, _) = run(events);
        let v = view(Some(&facts), None, now);
        (v.state, v.stall)
    };
    assert_eq!(
        stall_at(
            &[
                ("registry_hosted_status", 0, busy.clone()),
                ("registry_hosted_queued", MIN, queued(1)),
            ],
            2 * MIN
        ),
        (LiveState::Working, Some(Stall::Queue))
    );
    assert_eq!(
        stall_at(
            &[
                ("registry_hosted_status", 0, busy.clone()),
                ("registry_hosted_queued", MIN, queued(2)),
                ("registry_hosted_queued", 2 * MIN, queued(0)),
            ],
            3 * MIN
        ),
        (LiveState::Working, None)
    );
    let mut idle = process.clone();
    idle["status"] = json!("idle");
    assert_eq!(
        stall_at(
            &[
                ("registry_hosted_status", 0, busy.clone()),
                ("registry_hosted_queued", MIN, queued(1)),
                ("registry_hosted_status", 2 * MIN, idle),
            ],
            3 * MIN
        ),
        (LiveState::Idle, None)
    );
    // The queue is an interval of its own, closed when the count returns to zero.
    let (facts, closed) = run(&[
        ("registry_hosted_status", 0, busy),
        ("registry_hosted_queued", MIN, queued(1)),
        ("registry_hosted_queued", 3 * MIN, queued(0)),
    ]);
    assert_eq!(
        closed
            .iter()
            .map(|i| (i.aspect, i.value.as_str(), i.start, i.end))
            .collect::<Vec<_>>(),
        vec![(Aspect::Stall, "queue", MIN, 3 * MIN)]
    );
    assert_eq!(view(Some(&facts), None, 4 * MIN).waiting_on, None);
}
