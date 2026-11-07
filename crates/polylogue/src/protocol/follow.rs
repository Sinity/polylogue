//! `QUERY follow`: one line per live-state change and per new message or
//! tool call of the sessions a query selects, until interrupted. The daemon
//! streams lines until the request's deadline and ends with the marks it
//! reached; the client sends them back with the next request, so a follow
//! never repeats or skips a line and the daemon keeps no follower state.

use polylogue_store::FollowSnapshot;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg(test)]
mod tests;

/// How far a follower has seen each session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Marks {
    pub sessions: BTreeMap<String, SessionMark>,
}

/// The last state shown for a session and the last message and tool call
/// row keys shown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionMark {
    pub state: State,
    pub message: i64,
    pub tool_call: i64,
}

/// What a state line shows: the state, the wait, the stall and the open
/// tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waiting_on: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stall: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_tool: Option<String>,
}

/// One followed line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Line {
    State {
        session: String,
        at: i64,
        #[serde(flatten)]
        state: State,
    },
    Message {
        session: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        at: Option<i64>,
        message: String,
        role: String,
    },
    ToolCall {
        session: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        at: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        tool: String,
    },
    /// The last line of a request: send these marks with the next one.
    Marks { marks: Marks },
}

/// The lines for one session since its mark, and its new mark. A session
/// first seen gets a state line only: following starts now, not from the
/// session's history.
pub fn step(
    session: &str,
    mark: Option<&SessionMark>,
    snapshot: &FollowSnapshot,
    now: i64,
) -> (Vec<Line>, SessionMark) {
    let state = State {
        state: snapshot.view.state.as_str().to_owned(),
        waiting_on: snapshot.view.waiting_on.map(|w| w.as_str().to_owned()),
        stall: snapshot.view.stall.map(|s| s.as_str().to_owned()),
        open_tool: snapshot.open_tool.clone(),
    };
    let message = snapshot.messages.last().map(|m| m.key);
    let tool_call = snapshot.tool_calls.last().map(|c| c.key);
    let mut lines = Vec::new();
    let Some(mark) = mark else {
        lines.push(Line::State {
            session: session.to_owned(),
            at: now,
            state: state.clone(),
        });
        return (
            lines,
            SessionMark {
                state,
                message: message.unwrap_or(0),
                tool_call: tool_call.unwrap_or(0),
            },
        );
    };
    lines.extend(snapshot.messages.iter().map(|m| Line::Message {
        session: session.to_owned(),
        at: m.at,
        message: m.id.clone(),
        role: m.role.clone(),
    }));
    lines.extend(snapshot.tool_calls.iter().map(|c| Line::ToolCall {
        session: session.to_owned(),
        at: c.at,
        call_id: c.call_id.clone(),
        tool: c.tool.clone(),
    }));
    if state != mark.state {
        lines.push(Line::State {
            session: session.to_owned(),
            at: now,
            state: state.clone(),
        });
    }
    (
        lines,
        SessionMark {
            state,
            message: message.unwrap_or(mark.message),
            tool_call: tool_call.unwrap_or(mark.tool_call),
        },
    )
}

impl Line {
    /// The terminal form: `14:02:11 claude-code:5f2e… → waiting: permission`.
    pub fn text(&self, zone: &jiff::tz::TimeZone) -> Option<String> {
        let clock = |at: Option<i64>| {
            at.and_then(|ms| jiff::Timestamp::from_millisecond(ms).ok())
                .map(|t| t.to_zoned(zone.clone()).strftime("%H:%M:%S").to_string())
                .unwrap_or_else(|| "--:--:--".into())
        };
        Some(match self {
            Self::State { session, at, state } => {
                let mut text = format!("{} {session} → {}", clock(Some(*at)), state.state);
                if let Some(on) = &state.waiting_on {
                    text.push_str(&format!(": {on}"));
                }
                if let Some(stall) = &state.stall {
                    text.push_str(&format!(" (stalled: {stall})"));
                }
                if let Some(tool) = &state.open_tool {
                    text.push_str(&format!(" · {tool}"));
                }
                text
            }
            Self::Message {
                session,
                at,
                message,
                role,
            } => format!("{} {session} {role} message:{message}", clock(*at)),
            Self::ToolCall {
                session,
                at,
                call_id,
                tool,
            } => format!(
                "{} {session} tool {tool}{}",
                clock(*at),
                call_id
                    .as_ref()
                    .map(|id| format!(" ({id})"))
                    .unwrap_or_default()
            ),
            Self::Marks { .. } => return None,
        })
    }
}
