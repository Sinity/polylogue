//! The live view of a session (`read --view live`): its state, what it
//! waits on, its open call, task and peers, and its process tree, the
//! harness first with each child's command, age and tool call.
use super::super::Failure;
use polylogue_model::read::{Card, Process, ValueState};
use polylogue_store::{LiveDetail, ProcessRow, Reader};

#[cfg(test)]
mod tests;

/// The live card of `session` at `now`.
pub(super) fn card(reader: &Reader<'_>, session: &str, now: i64) -> Result<Card, Failure> {
    let detail = reader
        .live_detail(session, now)
        .map_err(Failure::from)?
        .ok_or_else(|| Failure::new("not_found", format!("no session {session}")))?;
    Ok(render(&detail, now))
}

fn known_or(value: Option<String>, reason: &str) -> ValueState<String> {
    value.map_or_else(
        || ValueState::Unknown {
            reason: Some(reason.into()),
        },
        ValueState::known,
    )
}

/// The card for a session's live detail.
pub(crate) fn render(detail: &LiveDetail, now: i64) -> Card {
    let view = &detail.view;
    let mut state = view.state.as_str().to_owned();
    if let Some(since) = view.since {
        state.push_str(&format!(
            " for {}",
            crate::protocol::status::short_duration(now - since)
        ));
    }
    if view.process_alive == Some(false) {
        state.push_str(" · process gone");
    }
    if let Some(word) = &detail.facts.harness_status {
        state.push_str(&format!(" · harness says {}", word.word));
    }
    let waits_on = match (view.waiting_on, view.stall) {
        (Some(on), _) => ValueState::known(on.as_str().to_owned()),
        (None, Some(stall)) => ValueState::known(format!("stall: {}", stall.as_str())),
        (None, None) => ValueState::NotApplicable,
    };
    let call = known_or(
        detail.facts.open_call.as_ref().map(|c| {
            format!(
                "{} for {}",
                c.tool,
                crate::protocol::status::short_duration(now - c.at)
            )
        }),
        "no tool call open",
    );
    let task = known_or(detail.facts.task.clone(), "no explicit task evidence");
    Card::Live {
        state: ValueState::known(state),
        waits_on,
        call,
        task,
        peers: detail.peers.clone(),
        processes: detail.processes.iter().map(|p| process(p, now)).collect(),
    }
}

fn process(p: &ProcessRow, now: i64) -> Process {
    let command = if p.argv.is_empty() {
        known_or(p.comm.clone(), "no command recorded")
    } else {
        ValueState::known(p.argv.join(" "))
    };
    let mut state = match (p.started_at, p.ended_at) {
        (_, Some(ended)) => format!(
            "ended {} ago",
            crate::protocol::status::short_duration(now - ended)
        ),
        (Some(started), None) => format!(
            "running for {}",
            crate::protocol::status::short_duration(now - started)
        ),
        (None, None) => "running".to_owned(),
    };
    if let Some(call) = &p.tool_call {
        state.push_str(&format!(" · tool call {call}"));
    }
    if let Some(cwd) = &p.cwd {
        state.push_str(&format!(" · in {cwd}"));
    }
    Process {
        reference: format!("{} pid {}", p.role, p.pid),
        parent: p.parent_pid.map(|pid| pid.to_string()),
        command,
        state: ValueState::known(state),
    }
}
