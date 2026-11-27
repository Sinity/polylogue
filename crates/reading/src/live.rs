//! The live view of a session (`read --view live`): its state, what it
//! waits on, its open call, task and peers, and its process tree, the
//! harness first with each child's command, age and tool call.
use crate::Error;
use polylogue_model::live::LiveState;
use polylogue_model::read::{Card, Process, ValueState};
use polylogue_store::{LiveDetail, ProcessRow, Reader};

#[cfg(test)]
mod tests;

/// The live card of `session` at `now`.
pub(super) fn card(
    reader: &Reader<'_>,
    session: &str,
    now: i64,
    zone: &jiff::tz::TimeZone,
) -> Result<Card, Error> {
    let detail = reader
        .live_detail(session, now)?
        .ok_or_else(|| Error::not_found(format!("no session {session}")))?;
    let mut card = render(&detail, now, zone);
    if detail.view.state == LiveState::Ended
        && detail.facts.ended_at.is_some()
        && let Card::Live {
            state: ValueState::Value { value },
            ..
        } = &mut card
    {
        let session = polylogue_model::SessionId::parse(session).map_err(Error::not_found)?;
        let reason = reader
            .events(&session)?
            .into_iter()
            .filter(|event| event.kind == "registry_hosted_ended")
            .max_by_key(|event| event.at)
            .and_then(|event| event.data.get("detail")?.as_str().map(str::to_owned));
        if let Some(reason) = reason {
            value.push_str(" · ");
            value.push_str(&reason);
        }
    }
    Ok(card)
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
pub(crate) fn render(detail: &LiveDetail, now: i64, zone: &jiff::tz::TimeZone) -> Card {
    let view = &detail.view;
    let mut state = view.state.as_str().to_owned();
    if view.state == LiveState::Stale {
        // A stale session shows what made it stale, not how long.
        match detail
            .facts
            .root_gone_at
            .and_then(|at| clock_time(at, zone))
        {
            Some(time) => state.push_str(&format!(" · process gone at {time}, no end event")),
            None => state.push_str(" · no end event"),
        }
    } else {
        if let Some(since) = view.since {
            state.push_str(&format!(
                " for {}",
                polylogue_render::short_duration(now - since)
            ));
        }
        if view.process_alive == Some(false) {
            state.push_str(" · process gone");
        }
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
                polylogue_render::short_duration(now - c.at)
            )
        }),
        "no tool call open",
    );
    let task = known_or(detail.facts.task.clone(), "no stated task");
    Card::Live {
        state: ValueState::known(state),
        waits_on,
        call,
        task,
        peers: detail.peers.clone(),
        processes: detail.processes.iter().map(|p| process(p, now)).collect(),
    }
}

/// `13:40` in `zone`.
fn clock_time(at: i64, zone: &jiff::tz::TimeZone) -> Option<String> {
    let at = jiff::Timestamp::from_millisecond(at).ok()?;
    Some(at.to_zoned(zone.clone()).strftime("%H:%M").to_string())
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
            polylogue_render::short_duration(now - ended)
        ),
        (Some(started), None) => format!(
            "running for {}",
            polylogue_render::short_duration(now - started)
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
