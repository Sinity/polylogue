use super::*;
use polylogue_model::live::{LiveState, LiveView, WaitingOn};
use polylogue_store::{FollowedCall, FollowedMessage};

fn snapshot(
    state: LiveState,
    waiting_on: Option<WaitingOn>,
    messages: &[i64],
    calls: &[i64],
) -> FollowSnapshot {
    FollowSnapshot {
        view: LiveView {
            state,
            since: None,
            until: None,
            waiting_on,
            stall: None,
            process_alive: Some(true),
        },
        open_tool: None,
        messages: messages
            .iter()
            .map(|k| FollowedMessage {
                key: *k,
                id: format!("m{k}"),
                role: "assistant".into(),
                at: Some(*k),
            })
            .collect(),
        tool_calls: calls
            .iter()
            .map(|k| FollowedCall {
                key: *k,
                call_id: Some(format!("c{k}")),
                tool: "Bash".into(),
                at: Some(*k),
            })
            .collect(),
    }
}

fn kinds(lines: &[Line]) -> Vec<String> {
    lines
        .iter()
        .map(|l| match l {
            Line::State { state, .. } => format!(
                "state {}{}",
                state.state,
                state
                    .waiting_on
                    .as_deref()
                    .map(|w| format!(" {w}"))
                    .unwrap_or_default()
            ),
            Line::Message { message, .. } => format!("message {message}"),
            Line::ToolCall { tool, call_id, .. } => {
                format!("tool {tool} {}", call_id.as_deref().unwrap_or(""))
            }
            Line::Marks { .. } => "marks".into(),
        })
        .collect()
}

/// A session first seen shows its state only; later steps show each new
/// message and tool call once and a state line only when the state
/// changed.
#[test]
fn following_shows_each_new_row_and_state_change_once() {
    let (lines, mark) = step(
        "s",
        None,
        &snapshot(LiveState::Working, None, &[1, 2], &[5]),
        0,
    );
    assert_eq!(kinds(&lines), ["state working"]);
    assert_eq!((mark.message, mark.tool_call), (2, 5));
    // Nothing new: nothing shown.
    let (lines, same) = step(
        "s",
        Some(&mark),
        &snapshot(LiveState::Working, None, &[], &[]),
        1,
    );
    assert!(lines.is_empty());
    assert_eq!(same, mark);
    // A new message and call, then a wait.
    let (lines, mark) = step(
        "s",
        Some(&same),
        &snapshot(LiveState::Waiting, Some(WaitingOn::Permission), &[3], &[6]),
        2,
    );
    assert_eq!(
        kinds(&lines),
        ["message m3", "tool Bash c6", "state waiting permission"]
    );
    assert_eq!((mark.message, mark.tool_call), (3, 6));
    let text = lines[2].text(&jiff::tz::TimeZone::UTC).unwrap();
    assert_eq!(text, "00:00:00 s → waiting: permission");
    // Marks travel through the client and back unchanged.
    let marks = Marks {
        sessions: [("s".to_owned(), mark)].into_iter().collect(),
    };
    let line = Line::Marks {
        marks: marks.clone(),
    };
    let back: Line = serde_json::from_str(&serde_json::to_string(&line).unwrap()).unwrap();
    assert_eq!(back, Line::Marks { marks });
    assert_eq!(back.text(&jiff::tz::TimeZone::UTC), None);
}
