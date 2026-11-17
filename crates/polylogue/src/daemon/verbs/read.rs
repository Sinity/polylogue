//! `read`: the read operation's body over the line's rows. Its next pages
//! are `rows` calls, so a reading continues in the grammar it started in.
use super::{RowRule, Selection, StepCall, StepDone, Verb, delegated, not_built, not_built_use};
use crate::daemon::Failure;
use crate::daemon::handler::Call;
use crate::daemon::service::Daemon;
use crate::protocol::line::{Rows, Step, StepAnswer};
use crate::protocol::steps::ReadStep;
use command_surface::{Deadline, RunnableRequest};
use polylogue_model::read::{Format, ReadyCall};
use polylogue_reading::{AttachmentPlace, Compile, Error, ReadRequest, ReadServices};
use polylogue_store::{Reader, WriterSender};
use std::num::NonZeroU64;
use std::sync::Arc;

pub(crate) struct Read;
impl Verb for Read {
    type Step = ReadStep;
    const NAME: &'static str = "read";
    const ROWS: RowRule = RowRule::Any;
    fn run(
        step: ReadStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        if step.follow {
            return Err(Failure::bad_request(
                "read --follow streams; it has no single answer",
            ));
        }
        let arguments = arguments(&step, target(selection)?)?;
        let call = Call {
            cursor: step.cursor.as_deref(),
            ..call.call.copy()
        };
        let reads = LineReads {
            daemon: call.daemon,
        };
        delegated(
            crate::daemon::service::read::page(
                arguments,
                &call,
                &reads,
                selection.resolved.as_deref(),
            )?,
            |page| StepAnswer::Read(Box::new(page)),
        )
    }
    fn would(step: &ReadStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        arguments(step, String::new())?;
        Ok(format!(
            "would read {} in the {} view",
            selection.described(),
            step.view
        ))
    }
}

/// The text a read selects.
pub(crate) fn target(selection: &Selection) -> Result<String, Failure> {
    selection
        .query_text()
        .ok_or_else(|| Failure::bad_request("read needs rows: name a selection"))
}

/// The read operation's arguments for a step; an option whose body is not
/// built yet is refused.
pub(crate) fn arguments(
    step: &ReadStep,
    target: String,
) -> Result<crate::protocol::read::Arguments, Failure> {
    let ReadStep {
        view,
        show,
        hide,
        from,
        until,
        first,
        last,
        around,
        diff,
        alternate,
        beside,
        blind,
        seed,
        physical,
        neutral_names,
        follow: _,
        until_idle,
        at,
        align,
        format,
        budget,
        cursor: _,
        delivery,
    } = step;
    for (given, option) in [
        (from.is_some(), "from"),
        (until.is_some(), "until"),
        (first.is_some(), "first"),
        (last.is_some(), "last"),
        (
            alternate.as_deref().is_some_and(|a| a != "none"),
            "alternate",
        ),
        (*beside, "beside"),
        (*blind, "blind"),
        (seed.is_some(), "seed"),
        (*until_idle, "until-idle"),
        (at.is_some(), "at"),
        (align.is_some(), "align"),
    ] {
        if given {
            return Err(not_built("read", option));
        }
    }
    let format = format
        .as_deref()
        .map(|name| {
            Format::parse(name).ok_or_else(|| not_built_use("read", &format!("--format {name}")))
        })
        .transpose()?;
    let page_tokens = budget
        .map(|tokens| {
            NonZeroU64::new(tokens)
                .ok_or_else(|| Failure::bad_value("read --budget must be positive"))
        })
        .transpose()?;
    Ok(crate::protocol::read::Arguments {
        target,
        view: *view,
        format,
        delivery: *delivery,
        page_tokens,
        show: show.clone(),
        hide: hide.clone(),
        output_lines: crate::protocol::read::output_lines(),
        for_others: *neutral_names,
        variant: if alternate.is_some() {
            vec!["none".into()]
        } else {
            Vec::new()
        },
        with_blobs: false,
        at: None,
        diff: diff.clone(),
        around: *around,
        latest: false,
        physical: *physical,
    })
}

/// The step that asks for the same read again, from `cursor`; the inverse
/// of [`arguments`] on every option a step carries.
pub(crate) fn step_of(
    arguments: crate::protocol::read::Arguments,
    cursor: Option<String>,
) -> ReadStep {
    let crate::protocol::read::Arguments {
        target: _,
        view,
        format,
        delivery,
        page_tokens,
        show,
        hide,
        output_lines: _,
        for_others,
        variant,
        with_blobs: _,
        at: _,
        diff,
        around,
        latest: _,
        physical,
    } = arguments;
    ReadStep {
        view,
        show,
        hide,
        around,
        diff,
        alternate: (variant == ["none"]).then(|| "none".to_owned()),
        physical,
        neutral_names: for_others,
        format: format.map(|format| format.name().to_owned()),
        budget: page_tokens.map(NonZeroU64::get),
        cursor,
        delivery,
        ..ReadStep::default()
    }
}

/// The daemon's read services, whose next pages are `rows` calls.
pub(crate) struct LineReads<'a> {
    pub daemon: &'a Daemon,
}

impl ReadServices for LineReads<'_> {
    fn requests(&self) -> &polylogue_model::user::RequestMetadata {
        ReadServices::requests(self.daemon)
    }
    fn freshness(
        &self,
        reader: &Reader<'_>,
    ) -> polylogue_store::Result<polylogue_model::freshness::ReadFreshness> {
        ReadServices::freshness(self.daemon, reader)
    }
    fn compiled<T>(
        &self,
        reader: &Reader<'_>,
        operation: &Arc<polylogue_reading::operation::Context>,
        target: &str,
        seed: u64,
        run: impl FnOnce(Compile<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        ReadServices::compiled(self.daemon, reader, operation, target, seed, run)
    }
    fn continuation(
        &self,
        request: &ReadRequest,
        cursor: Option<String>,
        deadline: Deadline,
    ) -> Result<ReadyCall, Error> {
        let arguments =
            crate::daemon::service::read::arguments_of(request).map_err(Error::Internal)?;
        let rows = Rows {
            selection: request.references.is_none().then(|| request.target.clone()),
            references: request
                .references
                .as_ref()
                .map(|references| references.iter().map(ToString::to_string).collect())
                .unwrap_or_default(),
            steps: vec![Step::Read(Box::new(step_of(arguments, cursor)))],
            ..Rows::default()
        };
        let arguments =
            serde_json::to_value(rows).map_err(|error| Error::Internal(error.to_string()))?;
        let identity = canonical_json::canonical_hash("polylogue-rows-call", &arguments)
            .map_err(|error| Error::Internal(error.to_string()))?;
        let request = RunnableRequest {
            operation: "rows".into(),
            arguments,
            request_id: format!("rows-page-{}", identity.to_hex()),
            deadline,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        // `rows` has no HTTP route yet, so its next call is checked against
        // its own declaration rather than offered as an HTTP call.
        let row = crate::commands::command("rows").expect("rows is a row");
        row.registration()
            .expect("rows is answered by the daemon")
            .admit(
                &request.arguments,
                row,
                &crate::commands::Caller {
                    surface: crate::commands::Surface::Cli,
                    cwd: None,
                },
            )
            .map_err(|failure| Error::Operation(failure.error))?;
        Ok(ReadyCall { request })
    }
    fn attachment_call(&self, at: &AttachmentPlace) -> Result<ReadyCall, Error> {
        ReadServices::attachment_call(self.daemon, at)
    }
    fn writer(&self) -> Option<&WriterSender> {
        ReadServices::writer(self.daemon)
    }
}
