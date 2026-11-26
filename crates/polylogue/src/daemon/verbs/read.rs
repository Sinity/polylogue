//! `read`: the read operation's body over the line's rows. Its next pages
//! are `rows` calls, so a reading continues in the grammar it started in.
use super::{
    Given, RowRule, Selection, StepCall, StepDone, Verb, delegated, not_built, not_built_use,
};
use crate::daemon::Failure;
use crate::daemon::handler::Call;
use crate::daemon::service::Daemon;
use crate::protocol::line::{Rows, Step, StepAnswer};
use crate::protocol::steps::ReadStep;
use command_surface::{Deadline, RunnableRequest};
use polylogue_model::Reference;
use polylogue_model::read::{Format, ReadBudget, ReadDelivery, ReadyCall};
use polylogue_reading::{AttachmentPlace, Compile, Error, ReadRequest, ReadServices};
use polylogue_store::{Reader, WriterSender};
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
        if let Given::Reference(Reference::Attachment {
            message,
            block,
            attachment,
        }) = &selection.given
        {
            return bytes(step, message, *block, *attachment, call);
        }
        let arguments = arguments(&step, target(selection)?)?;
        let call = Call {
            cursor: step.cursor.as_deref(),
            ..call.call.copy()
        };
        delegated(
            crate::daemon::service::read::page(arguments, &call, selection.resolved.as_deref())?,
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

/// An attachment reads as one checked range of its bytes; `next` is the
/// cursor of the following range.
fn bytes(
    step: ReadStep,
    message: &polylogue_model::MessageId,
    block: u64,
    attachment: u64,
    call: &StepCall<'_>,
) -> Result<StepDone, Failure> {
    let ReadStep {
        format,
        input,
        budget,
        cursor,
        ..
    } = step.clone();
    let rest = ReadStep {
        format: None,
        input: None,
        budget: ReadBudget::Default,
        cursor: None,
        displayed: false,
        delivery: ReadDelivery::Page,
        ..step
    };
    if rest != ReadStep::default() {
        return Err(Failure::bad_request(
            "an attachment reads as its bytes; it takes only --input, --budget, --cursor and --format bytes",
        ));
    }
    if format.as_deref().is_some_and(|format| format != "bytes") {
        return Err(Failure::bad_value(
            "an attachment reads only in the bytes format",
        ));
    }
    let index = |value: u64, key: &str| {
        usize::try_from(value)
            .map_err(|_| Failure::bad_value(format!("{key} exceeds address space")))
    };
    let length = match budget {
        ReadBudget::Tokens(tokens) => {
            index(tokens.saturating_mul(BYTES_PER_TOKEN).max(1), "budget")?
        }
        ReadBudget::Unlimited | ReadBudget::Default => crate::daemon::attachment::PAGE_BYTES,
    };
    delegated(
        crate::daemon::attachment::read(
            crate::daemon::attachment::Range {
                message: message.clone(),
                block: index(block, "block")?,
                attachment: index(attachment, "attachment")?,
                input,
                length,
                cursor,
            },
            call.call.daemon,
        ),
        |page| StepAnswer::Attachment(Box::new(page)),
    )
}

/// Base64 carries three bytes in four characters, about one token.
const BYTES_PER_TOKEN: u64 = 3;

/// The text a read selects.
pub(crate) fn target(selection: &Selection) -> Result<String, Failure> {
    // The fixed empty population supplies membership; its question still
    // supplies the reading's unit and recorded query identity.
    if selection.resolved.as_ref().is_some_and(Vec::is_empty)
        && let Some(question) = &selection.question
    {
        return Ok(question.clone());
    }
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
        displayed,
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
        input,
    } = step;
    if input.is_some() {
        return Err(Failure::bad_request(
            "--input chooses the stored copy of an attachment; read attachment:MESSAGE#BLOCK",
        ));
    }
    for (given, option) in [
        (from.as_deref().is_some_and(|from| from != "seen"), "from"),
        (until.is_some(), "until"),
        (first.is_some(), "first"),
        (last.is_some(), "last"),
        (*beside, "beside"),
        (*blind, "blind"),
        (seed.is_some(), "seed"),
        (*until_idle, "until-idle"),
        (align.is_some(), "align"),
    ] {
        if given {
            return Err(not_built("read", option));
        }
    }
    let format = format
        .as_deref()
        .map(|name| {
            Format::parse(name).ok_or_else(|| match name {
                // Documented formats whose encoders are not built.
                "citation" | "bytes" => not_built_use("read", &format!("--format {name}")),
                _ => Failure::bad_value(format!(
                    "unknown format `{name}`; formats are terminal, markdown, html, json and jsonl"
                )),
            })
        })
        .transpose()?;
    Ok(crate::protocol::read::Arguments {
        target,
        from: from.clone(),
        displayed: *displayed,
        view: *view,
        format,
        delivery: *delivery,
        page_tokens: *budget,
        show: show.clone(),
        hide: hide.clone(),
        output_lines: crate::protocol::read::output_lines(),
        for_others: *neutral_names,
        alternate: alternate.iter().cloned().collect(),
        with_blobs: false,
        at: at.clone(),
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
) -> Result<ReadStep, Failure> {
    let crate::protocol::read::Arguments {
        target: _,
        from,
        displayed,
        view,
        format,
        delivery,
        page_tokens,
        show,
        hide,
        output_lines: _,
        for_others,
        alternate,
        with_blobs: _,
        at,
        diff,
        around,
        latest: _,
        physical,
    } = arguments;
    let alternate = match alternate.as_slice() {
        [] => None,
        [alternate] => Some(alternate.clone()),
        _ if cursor.is_some() => None,
        _ => {
            return Err(Failure::bad_request(
                "a fresh read names one --alternate; a continuation preserves all alternates in its cursor",
            ));
        }
    };
    Ok(ReadStep {
        from,
        displayed,
        at,
        view,
        show,
        hide,
        around,
        diff,
        alternate,
        physical,
        neutral_names: for_others,
        format: format.map(|format| format.name().to_owned()),
        budget: page_tokens,
        cursor,
        delivery,
        ..ReadStep::default()
    })
}

/// The daemon's read services, whose next pages are `rows` calls.
pub(crate) struct LineReads<'a> {
    pub daemon: &'a Daemon,
    pub accepted: Option<&'a RunnableRequest>,
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
        run: impl FnOnce(Compile<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        ReadServices::compiled(self.daemon, reader, operation, target, run)
    }
    fn continuation(
        &self,
        request: &ReadRequest,
        cursor: Option<String>,
        deadline: Deadline,
    ) -> Result<ReadyCall, Error> {
        let arguments =
            crate::daemon::service::read::arguments_of(request).map_err(Error::Internal)?;
        let original: Option<Rows> = self
            .accepted
            .filter(|accepted| accepted.operation == "rows")
            .map(|accepted| serde::Deserialize::deserialize(&accepted.arguments))
            .transpose()
            .map_err(|error: serde_json::Error| Error::Internal(error.to_string()))?;
        let rows = Rows {
            all: original.is_some_and(|rows| rows.all),
            selection: request.references.is_none().then(|| request.target.clone()),
            references: request
                .references
                .as_ref()
                .map(|references| references.iter().map(ToString::to_string).collect())
                .unwrap_or_default(),
            steps: vec![Step::Read(Box::new(
                step_of(arguments, cursor).map_err(|failure| Error::Operation(failure.error))?,
            ))],
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
            budget_tokens: self
                .accepted
                .map_or(Default::default(), |accepted| accepted.budget_tokens),
            cursor: None,
            fresh: self.accepted.is_some_and(|accepted| accepted.fresh),
            format: self
                .accepted
                .map_or(Default::default(), |accepted| accepted.format),
        };
        // A continuation obeys the same declared options as its first page.
        let row = crate::commands::command(&request.operation).expect("rows is a row");
        crate::daemon::handler::policy(
            row,
            &request.arguments,
            polylogue_model::authorship::Channel::Cli,
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

#[cfg(test)]
mod tests;
