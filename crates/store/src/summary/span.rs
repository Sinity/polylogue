//! A session's clocks, tool unions and time per party
//! ([`crate::measure::time`]), folded over its own messages.
//!
//! The calculation adds across a settled message, one no interval crosses,
//! so the fold holds the messages after its last settled one and settles
//! them when its state is kept. A message or call that starts before the
//! settled message cannot be added to what was settled: the run starts
//! again from the first row. Which code calls count depends on whether the
//! session holds a call run by a script, so the fold keeps the calculation
//! both with and without them and chooses at the end.
use super::Output;
use super::fold::{CallRow, Context, Flow, Fold, MessageRow, SessionRow};
use crate::Result;
use crate::measure::time::{Message, Prefix, Rows, Tool, settled};
use polylogue_model::derived::CLUSTER_GAP_MS;
use serde::{Deserialize, Serialize};

pub(crate) struct SessionSpan;

#[derive(Default, Serialize, Deserialize)]
pub(crate) struct State {
    /// Code calls counted, for a session without a script call.
    with_code: Part,
    /// Code calls left out, for a session with one.
    without_code: Part,
    script: bool,
    /// The earliest and latest event time, the clocks of a session
    /// without messages of its own.
    events: Option<(i64, i64)>,
}

/// The calculation up to its last settled message, and the rows after it.
#[derive(Default, Serialize, Deserialize)]
struct Part {
    prefix: Prefix,
    pending: Rows,
}

impl Part {
    /// Adds a message, with the calls `counted` keeps; whether its rows
    /// start before what is settled.
    fn add(&mut self, m: &MessageRow, counted: impl Fn(&CallRow) -> bool) -> Flow {
        let settled_at = self.prefix.end.as_ref().map(|(_, at)| *at);
        let starts =
            m.at.into_iter()
                .chain(m.calls.iter().filter(|c| counted(c)).filter_map(|c| c.at));
        if let Some(at) = settled_at
            && starts.into_iter().any(|start| start < at)
        {
            return Flow::Restart;
        }
        let last = m.blocks.last();
        let index = self.pending.messages.len();
        self.pending.messages.push(Message {
            id: m.id.clone(),
            at: m.at,
            author: m.written_by.clone(),
            prompt: m.prompt_kind.clone(),
            stop: m.stop_reason.clone(),
            last_block: last.map(|b| b.kind.clone()),
            open_call: last
                .is_some_and(|b| m.calls.iter().any(|c| c.block_pk == b.pk && !c.paired)),
        });
        self.pending.keys.push((m.order.clone(), m.id.clone()));
        for call in m.calls.iter().filter(|c| counted(c)) {
            let key = match call.action.as_deref() {
                Some("shell") => call
                    .command_norm
                    .clone()
                    .unwrap_or_else(|| call.tool.clone()),
                Some(action) => action.to_owned(),
                None => call.tool.clone(),
            };
            self.pending.tools.push(Tool {
                message_index: index,
                key,
                start: call.at,
                end: call.result.as_ref().and_then(|r| r.own_at),
                paired: call.paired,
                failed: call.outcome == "error",
            });
            self.pending.result_keys.push(
                call.result
                    .as_ref()
                    .map(|r| (r.order.clone(), r.id.clone())),
            );
        }
        Flow::Continue
    }

    /// Adds the pending messages up to the last settled one to the prefix.
    fn settle(&mut self) {
        let Some(c) = settled(self.prefix.last(), &self.pending) else {
            return;
        };
        let pending = std::mem::take(&mut self.pending);
        let split = pending.tools.partition_point(|t| t.message_index <= c);
        self.prefix.extend(
            &pending.messages[..=c],
            &pending.tools[..split],
            0,
            CLUSTER_GAP_MS,
        );
        let at = pending.messages[c].at.expect("a settled message is timed");
        self.prefix.end = Some((pending.keys[c].clone(), at));
        let mut tools = pending.tools;
        let mut results = pending.result_keys;
        self.pending = Rows {
            messages: pending.messages[c + 1..].to_vec(),
            keys: pending.keys[c + 1..].to_vec(),
            tools: tools
                .drain(split..)
                .map(|mut t| {
                    t.message_index -= c + 1;
                    t
                })
                .collect(),
            result_keys: results.drain(split..).collect(),
        };
    }

    fn whole(&self) -> Prefix {
        let mut whole = self.prefix.clone();
        whole.extend(
            &self.pending.messages,
            &self.pending.tools,
            0,
            CLUSTER_GAP_MS,
        );
        whole
    }
}

impl Fold for SessionSpan {
    type State = State;
    const NAME: &'static str = "session_span";
    const VERSION: i64 = 4;

    fn step(state: &mut State, row: &SessionRow, _: &Context<'_>, _: &mut Output) -> Result<Flow> {
        match row {
            SessionRow::Event(e) => {
                if let Some(at) = e.at {
                    state.events = Some(match state.events {
                        Some((a, b)) => (a.min(at), b.max(at)),
                        None => (at, at),
                    });
                }
            }
            SessionRow::Message(m) => {
                state.script |= m.calls.iter().any(|c| c.run_by_script);
                if m.own {
                    if let Flow::Restart = state.with_code.add(m, |_| true) {
                        return Ok(Flow::Restart);
                    }
                    if let Flow::Restart = state.without_code.add(m, |c| !c.is_code()) {
                        return Ok(Flow::Restart);
                    }
                }
            }
        }
        Ok(Flow::Continue)
    }

    fn rest(state: &mut State) {
        state.with_code.settle();
        state.without_code.settle();
    }

    fn end(state: &State, _: &Context<'_>, out: &mut Output) -> Result<()> {
        let part = if state.script {
            &state.without_code
        } else {
            &state.with_code
        };
        let whole = part.whole();
        let whole = match state.events {
            Some((first, last)) if whole.is_empty() => Prefix::of_events(first, last),
            _ => whole,
        };
        out.span = Some(Box::new(whole.finish()));
        Ok(())
    }
}
