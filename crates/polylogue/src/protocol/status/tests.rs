use super::*;
use serde_json::json;

#[test]
fn held_signal_observations_preserve_reasons_without_inventing_available_rules() {
    let failure = crate::daemon::service::error::<()>(
        "not_yet_available",
        "owner context has not been obtained",
    );
    let observed =
        json!({"held":[{"reason":"the operation clock is unavailable","detail":failure}]});
    let typed: Signals = serde_json::from_value(observed.clone()).unwrap();
    let roundtrip = serde_json::to_value(typed).unwrap();
    assert_eq!(roundtrip, observed);
    assert!(roundtrip.get("rules").is_none());
    assert_eq!(
        roundtrip["held"][0]["detail"]["error"]["kind"],
        "not_yet_available"
    );
}

#[test]
fn ordinary_signal_summaries_retain_held_rule_identity() {
    let observed = serde_json::to_value(Signals::Observed {
        rules: vec![],
        unusable: vec![],
        held: vec![HeldSignal {
            id: Some("rule-one".into()),
            reason: "the index is catching up".into(),
            detail: None,
        }],
    })
    .unwrap();
    let typed: Signals = serde_json::from_value(observed.clone()).unwrap();
    assert_eq!(serde_json::to_value(typed).unwrap(), observed);
}

/// `status --line`: how many run, how many wait and the longest wait;
/// nothing at all when nothing runs.
#[test]
fn the_status_line_counts_running_and_the_longest_wait() {
    use polylogue_model::live::{LiveState, WaitingOn};
    let session = |id: &str, state, since| polylogue_store::RunningSession {
        session: id.into(),
        state,
        since: Some(since),
        waiting_on: (state == LiveState::Waiting).then_some(WaitingOn::Permission),
        stall: None,
        open_tool: None,
        task: None,
        footprint: 0,
        cwd: None,
    };
    let now = 10 * 60_000;
    let running = Running::Measured {
        waiting: 2,
        sessions: vec![
            session("codex:a", LiveState::Waiting, 3 * 60_000),
            session("codex:b", LiveState::Waiting, 9 * 60_000),
            session("codex:c", LiveState::Working, 0),
        ],
    };
    assert_eq!(running.line(now), "3 running · 2 waiting 7m");
    let quiet = Running::Measured {
        waiting: 0,
        sessions: vec![session("codex:c", LiveState::Idle, 0)],
    };
    assert_eq!(quiet.line(now), "1 running");
    let none = Running::Measured {
        waiting: 0,
        sessions: vec![],
    };
    assert_eq!(none.line(now), "");
    assert_eq!(short_duration(3 * 3_600_000 + 12 * 60_000), "3h12m");
    assert_eq!(short_duration(45_000), "45s");
}
