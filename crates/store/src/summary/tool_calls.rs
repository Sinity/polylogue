//! A shell call's normalized command (a row value the writer writes with the
//! call), and the outcome reason a call without a result earns once its
//! session moved on. The call's action comes from its origin's parser.

use super::fold::{Context, Flow, Fold, SessionRow};
use super::{Kind, Output, Summary, Version};
use crate::Result;
use rusqlite::{Connection, Transaction, params};

pub(super) const COMMAND_NORM: Summary = Summary {
    name: "command_norm",
    products: &[],
    version: Version::Rules(2),
    kind: Kind::Row(|conn, pk, _, out| {
        out.command_norm = normalize(conn, pk)?;
        Ok(())
    }),
};

/// A session's shell calls. The unary `+` keeps the planner off
/// `tool_call_action`: one action can hold most of the archive's calls, and
/// a lookup by it would visit them all for every session.
pub(super) const SHELL_CALLS: &str =
    "SELECT pk, command FROM tool_call WHERE session_pk = ?1 AND +action = 'shell'";

/// Every shell call of the session, with its normalized command.
fn normalize(conn: &Connection, session_pk: i64) -> Result<Vec<(i64, Option<String>)>> {
    let calls: Vec<(i64, Option<String>)> = conn
        .prepare_cached(SHELL_CALLS)?
        .query_map([session_pk], |r| <(_, _)>::try_from(r))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(calls
        .into_iter()
        .map(|(pk, command)| {
            let norm = command
                .as_deref()
                .and_then(polylogue_model::command::normalize);
            (pk, norm)
        })
        .collect())
}

pub(super) fn apply_command_norm(
    tx: &Transaction<'_>,
    calls: &[(i64, Option<String>)],
) -> Result<()> {
    let mut set = tx.prepare_cached(
        "UPDATE tool_call SET command_norm = ?2 WHERE pk = ?1 AND command_norm IS NOT ?2",
    )?;
    for (pk, norm) in calls {
        set.execute(params![pk, norm])?;
    }
    Ok(())
}

/// The outcome reason of calls without a result. A call still without a
/// result once a later model response exists in its session never gets
/// one, since a harness answers only after every call of the earlier
/// response has its result. The reason is a fact of the records; whether a
/// `no_result` call is pending is decided at read time from live state.
pub(crate) struct Reasons;

/// The calls without a result since the last model response that can pass
/// them: those of messages that mark where their response ends.
#[derive(Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct State {
    open: Vec<i64>,
}

impl Fold for Reasons {
    type State = State;
    const NAME: &'static str = "tool_calls";
    const VERSION: i64 = 4;

    fn step(
        state: &mut State,
        row: &SessionRow,
        _: &Context<'_>,
        out: &mut Output,
    ) -> Result<Flow> {
        let SessionRow::Message(m) = row else {
            return Ok(Flow::Continue);
        };
        if m.role == "assistant" && m.written_by == "model" && !m.boundary_unknown {
            for call in state.open.drain(..) {
                out.reasons.push((call, "passed_without_result"));
            }
        }
        for call in &m.calls {
            let open = !call.paired
                && call.outcome == "unknown"
                && matches!(
                    call.outcome_reason.as_deref(),
                    Some("no_result" | "passed_without_result")
                );
            if !open {
                continue;
            }
            if m.boundary_unknown {
                out.reasons.push((call.pk, "no_result"));
            } else {
                state.open.push(call.pk);
            }
        }
        Ok(Flow::Continue)
    }

    fn end(state: &State, _: &Context<'_>, out: &mut Output) -> Result<()> {
        out.reasons
            .extend(state.open.iter().map(|call| (*call, "no_result")));
        Ok(())
    }
}

pub(super) fn apply_reasons(tx: &Transaction<'_>, calls: &[(i64, &'static str)]) -> Result<()> {
    let mut set = tx.prepare_cached(
        "UPDATE tool_call SET outcome_reason = ?2 WHERE pk = ?1 AND outcome_reason IS NOT ?2",
    )?;
    for (pk, reason) in calls {
        set.execute(params![pk, reason])?;
    }
    Ok(())
}
