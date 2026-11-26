//! Where a chat action or an import job stands at the query's instant, from
//! its events in order, by the model's one rule ([`chat_status`],
//! [`import_job_status`]):
//!
//! - `pl_chat_action(part, now, needs_approval, kind, phase, by, at ORDER BY n)`
//! - `pl_import_job(part, now, kind, by, at ORDER BY n)`
//!
//! `part` is `state` (the status name) or `claimed_by` (the extension
//! instance holding an unexpired claim, NULL when none).
use super::{Connection, Context, FLAGS};
use polylogue_model::user::{
    ChatEvent, ImportJobCounts, ImportJobEvent, chat_status, import_job_status,
};
use rusqlite::functions::Aggregate;

fn function_error(message: String) -> rusqlite::Error {
    rusqlite::Error::UserFunctionError(message.into())
}

/// What a fold answers: the part asked and the instant it is asked at.
#[derive(Default)]
struct Asked {
    part: String,
    now: i64,
}

impl Asked {
    fn read(&mut self, ctx: &Context<'_>) -> rusqlite::Result<()> {
        self.part = ctx.get(0)?;
        self.now = ctx.get(1)?;
        Ok(())
    }

    fn answer(
        &self,
        state: &'static str,
        claimed_by: Option<&str>,
    ) -> rusqlite::Result<Option<String>> {
        match self.part.as_str() {
            "state" => Ok(Some(state.to_owned())),
            "claimed_by" => Ok(claimed_by.map(str::to_owned)),
            other => Err(function_error(format!(
                "no part {other:?} of an extension state"
            ))),
        }
    }
}

struct ChatAction;
impl Aggregate<(Asked, bool, Vec<ChatEvent>), Option<String>> for ChatAction {
    fn init(&self, _: &mut Context<'_>) -> rusqlite::Result<(Asked, bool, Vec<ChatEvent>)> {
        Ok((Asked::default(), false, Vec::new()))
    }

    fn step(
        &self,
        ctx: &mut Context<'_>,
        (asked, needs_approval, events): &mut (Asked, bool, Vec<ChatEvent>),
    ) -> rusqlite::Result<()> {
        asked.read(ctx)?;
        *needs_approval = ctx.get(2)?;
        events.push(ChatEvent {
            request: String::new(),
            kind: ctx.get(3)?,
            phase: ctx.get(4)?,
            by: ctx.get(5)?,
            at: ctx.get(6)?,
            detail: None,
            receipt: None,
        });
        Ok(())
    }

    fn finalize(
        &self,
        _: &mut Context<'_>,
        folded: Option<(Asked, bool, Vec<ChatEvent>)>,
    ) -> rusqlite::Result<Option<String>> {
        let Some((asked, needs_approval, events)) = folded else {
            return Ok(None);
        };
        let (status, holder) = chat_status(needs_approval, &events, asked.now);
        asked.answer(status.as_str(), holder)
    }
}

struct ImportJob;
impl Aggregate<(Asked, Vec<ImportJobEvent>), Option<String>> for ImportJob {
    fn init(&self, _: &mut Context<'_>) -> rusqlite::Result<(Asked, Vec<ImportJobEvent>)> {
        Ok((Asked::default(), Vec::new()))
    }

    fn step(
        &self,
        ctx: &mut Context<'_>,
        (asked, events): &mut (Asked, Vec<ImportJobEvent>),
    ) -> rusqlite::Result<()> {
        asked.read(ctx)?;
        events.push(ImportJobEvent {
            request: String::new(),
            kind: ctx.get(2)?,
            by: ctx.get(3)?,
            at: ctx.get(4)?,
            position: None,
            counts: ImportJobCounts::default(),
            detail: None,
        });
        Ok(())
    }

    fn finalize(
        &self,
        _: &mut Context<'_>,
        folded: Option<(Asked, Vec<ImportJobEvent>)>,
    ) -> rusqlite::Result<Option<String>> {
        let Some((asked, events)) = folded else {
            return Ok(None);
        };
        let (status, holder) = import_job_status(&events, asked.now);
        asked.answer(status.as_str(), holder)
    }
}

pub(super) fn register(conn: &Connection) -> rusqlite::Result<()> {
    conn.create_aggregate_function("pl_chat_action", 7, FLAGS, ChatAction)?;
    conn.create_aggregate_function("pl_import_job", 5, FLAGS, ImportJob)
}
