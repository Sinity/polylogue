//! A title for a session whose records name none: the first line of the
//! first prompt a person typed. A subagent forked with its parent's context
//! opens with the harness's fork instructions, not a prompt, so it has no
//! display title. Kept apart from the recorded title, which always wins when present and
//! is never replaced by this display fallback in exports.

use super::Output;
use super::fold::{Context, Flow, Fold, SessionRow};
use crate::Result;
use rusqlite::{Transaction, params};

pub(crate) struct Title;

/// Characters kept of the prompt's first line.
const LENGTH: usize = 80;

/// The first text block a person typed as prose, in session order, once
/// one was folded: its title, if it has one.
#[derive(Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct State(Option<Option<String>>);

impl Fold for Title {
    type State = State;
    const NAME: &'static str = "title";
    const VERSION: i64 = 7;

    fn step(state: &mut State, row: &SessionRow, cx: &Context<'_>, _: &mut Output) -> Result<Flow> {
        if state.0.is_some() {
            return Ok(Flow::Continue);
        }
        let SessionRow::Message(m) = row else {
            return Ok(Flow::Continue);
        };
        if m.duplicate
            || m.written_by != "human"
            || m.prompt_kind.as_deref().unwrap_or("prose") != "prose"
        {
            return Ok(Flow::Continue);
        }
        let Some(block) = m.blocks.iter().find(|b| b.kind == "text") else {
            return Ok(Flow::Continue);
        };
        let prompt = crate::ordered_blocks::one(cx.conn, block.pk)?;
        state.0 = Some(match prompt.text() {
            Some(text) if !is_fork_instructions(text) => title_of(text),
            _ => None,
        });
        Ok(Flow::Continue)
    }

    fn end(state: &State, _: &Context<'_>, out: &mut Output) -> Result<()> {
        out.title = Some(state.0.clone().flatten());
        Ok(())
    }
}

pub(super) fn apply(tx: &Transaction<'_>, session_pk: i64, title: Option<&str>) -> Result<()> {
    tx.prepare_cached(
        "UPDATE session SET derived_title = ?2 WHERE pk = ?1 AND derived_title IS NOT ?2",
    )?
    .execute(params![session_pk, title])?;
    Ok(())
}

/// The harness's instructions to a context-forked subagent.
fn is_fork_instructions(text: &str) -> bool {
    text.trim_start().starts_with("<fork-boilerplate>")
}

pub(crate) fn title_of(prompt: &str) -> Option<String> {
    let line = prompt.lines().map(str::trim).find(|l| !l.is_empty())?;
    let mut t: String = line.chars().take(LENGTH).collect();
    if line.chars().count() > LENGTH {
        t.push('…');
    }
    Some(t)
}
