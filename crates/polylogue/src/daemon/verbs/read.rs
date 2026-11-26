//! `read`: the rows' content, a page at a time, or one checked range of an
//! attachment's bytes. Its next pages are `rows` calls carrying the same
//! step with the next cursor, so a reading continues in the grammar it
//! started in.
use super::{Given, RowRule, Selection, StepCall, StepDone, Verb, not_built, not_built_use};
use crate::commands::ToolAccess;
use crate::daemon::Failure;
use crate::daemon::handler::Call;
use crate::daemon::service::Daemon;
use crate::protocol::line::{Rows, Step, StepAnswer};
use crate::protocol::steps::ReadStep;
use command_surface::{Answer, Deadline, RunnableRequest};
use polylogue_model::read::{Format, HeaderKind, ReadBudget, ReadComplete, ReadPage, ReadyCall};
use polylogue_model::{Reason, Reference};
use polylogue_reading::{AttachmentPlace, Compile, Error, ReadRequest, ReadServices};
use polylogue_store::{Reader, WriterSender};
use std::sync::Arc;

pub(crate) struct Read;
impl Verb for Read {
    type Step = ReadStep;
    const NAME: &'static str = "read";
    const ROWS: RowRule = RowRule::Any;
    const ACCESS: ToolAccess = ToolAccess::Read;
    fn run(
        step: ReadStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<StepDone, Failure> {
        if step.follow {
            return Err(Reason::FollowStreams {}.into());
        }
        if let Given::Reference(Reference::Attachment {
            message,
            block,
            attachment,
        }) = &selection.given
        {
            return bytes(step, message, *block, *attachment, call);
        }
        let displayed = step.displayed;
        let request = request(step, selection)?;
        Ok(page(request, displayed, &call.call)?
            .map(|page| StepAnswer::Read(Box::new(page)))
            .into())
    }
    fn would(step: &ReadStep, selection: &Selection, _: &StepCall<'_>) -> Result<String, Failure> {
        check(step)?;
        Ok(format!(
            "would read {} in the {} view",
            selection.described(),
            step.options.view
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
        delivery: Default::default(),
        ..step
    };
    if rest != ReadStep::default() {
        return Err(Reason::AttachmentReadOptions {}.into());
    }
    if format.as_deref().is_some_and(|format| format != "bytes") {
        return Err(Reason::AttachmentReadFormat {}.into());
    }
    let index = |value: u64, field: &str| {
        usize::try_from(value).map_err(|_| {
            Failure::refusal(Reason::TooLarge {
                field: field.into(),
            })
        })
    };
    let length = match budget {
        ReadBudget::Tokens(tokens) => {
            index(tokens.saturating_mul(BYTES_PER_TOKEN).max(1), "budget")?
        }
        ReadBudget::Unlimited | ReadBudget::Default => crate::daemon::attachment::PAGE_BYTES,
    };
    Ok(crate::daemon::attachment::read(
        crate::daemon::attachment::Range {
            message: message.clone(),
            block: index(block, "block")?,
            attachment: index(attachment, "attachment")?,
            input,
            length,
            cursor,
        },
        call.call.daemon,
    )
    .map(|page| StepAnswer::Attachment(Box::new(page)))
    .into())
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
        .ok_or_else(|| Reason::ReadNeedsRows {}.into())
}

/// The step's document encoding; an option whose body is not built yet is
/// refused.
fn check(step: &ReadStep) -> Result<Option<Format>, Failure> {
    let ReadStep {
        displayed: _,
        options,
        until,
        first,
        last,
        beside,
        blind,
        seed,
        follow: _,
        until_idle,
        align,
        format,
        budget: _,
        cursor: _,
        delivery: _,
        input,
    } = step;
    if input.is_some() {
        return Err(Reason::InputNeedsAttachment {}.into());
    }
    for (given, option) in [
        (
            options.from.as_deref().is_some_and(|from| from != "seen"),
            "from",
        ),
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
    options
        .validate()
        .map_err(|error| Failure::from(Error::Layout(error)))?;
    format
        .as_deref()
        .map(|name| {
            Format::parse(name).ok_or_else(|| match name {
                // Documented formats whose encoders are not built.
                "citation" | "bytes" => not_built_use("read", &format!("--format {name}")),
                _ => Failure::refusal(Reason::UnknownReadFormat {
                    format: name.into(),
                }),
            })
        })
        .transpose()
}

/// The reading a step asks for over the line's rows.
pub(crate) fn request(step: ReadStep, selection: &Selection) -> Result<ReadRequest, Failure> {
    let mut request = request_for(step, target(selection)?)?;
    request.references = selection.resolved.clone();
    Ok(request)
}

/// The reading a step asks for of `target`.
pub(crate) fn request_for(step: ReadStep, target: String) -> Result<ReadRequest, Failure> {
    let format = check(&step)?;
    let mut request = ReadRequest::new(target, step.options);
    request.delivery = step.delivery;
    request.format = format.unwrap_or(Format::Json);
    request.budget = step.budget;
    request.cursor = step.cursor;
    Ok(request)
}

/// The step that reads `request` again from `cursor`: the step it came
/// from, with the alternates and the time it resolved kept.
pub(crate) fn step_from(request: &ReadRequest, cursor: Option<String>) -> ReadStep {
    ReadStep {
        options: request.options.clone(),
        format: Some(request.format.name().to_owned()),
        budget: request.budget,
        cursor,
        delivery: request.delivery,
        ..ReadStep::default()
    }
}

/// One page of a read, with ready calls bound to its accepted request; a
/// page the caller displays advances its seen marks.
pub(crate) fn page(
    mut request: ReadRequest,
    displayed: bool,
    call: &Call<'_>,
) -> Result<Answer<ReadPage>, Failure> {
    let displayed = match call.caller.channel() {
        polylogue_model::authorship::Channel::Mcp => true,
        channel => crate::operation::command_line(channel) && displayed,
    };
    let history = if request.cursor.is_none() && !polylogue_reading::is_object_read(&request) {
        call.daemon
            .writes
            .as_ref()
            .map(|writer| writer.archive().reserve_history())
            .transpose()?
    } else {
        None
    };
    let reader = call.reader()?;
    let context = call.context()?;
    polylogue_reading::resolve_start(&reader, &context, &mut request)
        .map_err(|error| failure(error, &request))?;
    let services = &LineReads {
        daemon: call.daemon,
        accepted: context.accepted_request(),
    };
    let request = polylogue_reading::with_alternates(&reader, services, request.clone())
        .map_err(|error| failure(error, &request))?;
    let (page, marks) = polylogue_reading::page(&reader, services, &request, &context, history)
        .map_err(|error| failure(error, &request))?;
    if displayed {
        crate::daemon::seen::record(call.daemon, call.invocation, marks);
    }
    let freshness = page.snapshot.freshness.clone();
    let missing = page.snapshot.incomplete.clone();
    let empty = page.records.is_empty()
        && page.document.footer.shown == 0
        && page.document.footer.coverage.is_empty()
        && matches!(&page.document.header.kind,
            HeaderKind::Record { record_type } if record_type == "query");
    let mut answer = crate::protocol::read(page, &freshness);
    if !missing.is_empty() {
        answer.outcome = match answer.outcome {
            command_surface::Outcome::Ok(value) | command_surface::Outcome::Empty(value) => {
                command_surface::Outcome::Degraded { value, missing }
            }
            command_surface::Outcome::Degraded {
                value,
                missing: mut notes,
            } => {
                notes.extend(missing);
                command_surface::Outcome::Degraded {
                    value,
                    missing: notes,
                }
            }
            error => error,
        };
    } else if empty {
        if let command_surface::Outcome::Ok(value) = answer.outcome {
            answer.outcome = command_surface::Outcome::Empty(value);
        }
        answer.envelope.notes.push("no rows matched".into());
    }
    Ok(answer)
}

/// A read as a stream of events; `services` make its ready calls.
pub(crate) fn stream(
    mut request: ReadRequest,
    call: &Call<'_>,
    context: &Arc<polylogue_reading::operation::Context>,
    output: &mut dyn std::io::Write,
    services: &impl ReadServices,
) -> Result<Answer<ReadComplete>, Failure> {
    crate::daemon::service::accepting(call.daemon)?;
    let reader = call.reader()?;
    polylogue_reading::resolve_start(&reader, context, &mut request)
        .map_err(|error| failure(error, &request))?;
    let request = polylogue_reading::with_alternates(&reader, services, request.clone())
        .map_err(|error| failure(error, &request))?;
    let freshness = services.freshness(&reader)?;
    let complete = polylogue_reading::stream(&reader, services, &request, context, output)
        .map_err(|error| failure(error, &request))?;
    if complete.counts == polylogue_model::bundle::Counts::default()
        && complete.footer.coverage.is_empty()
        && freshness.up_to_date()
    {
        Ok(Answer::empty(complete))
    } else {
        Ok(crate::protocol::read(complete, &freshness))
    }
}

/// A stale cursor names the reading to start again, from its beginning,
/// as a sentence and as a ready call.
fn failure(error: Error, request: &ReadRequest) -> Failure {
    let stale = error.code() == polylogue_model::Code::StaleCursor;
    let mut failure = Failure::from(error);
    if !stale {
        return failure;
    }
    failure.error.hint = None;
    if let Some(changed) = failure
        .error
        .message
        .strip_suffix("; run the query again without its cursor")
    {
        failure.error.message = changed.to_owned();
    }
    let mut target = request.target.clone();
    if let polylogue_reading::ReadStart::Bound(bound) = &request.start
        && bound.from_seen
    {
        target = if bound.logical {
            format!("root:{}", bound.session)
        } else {
            Reference::Session(bound.session.clone()).to_string()
        };
    }
    let view = match request.options.view {
        polylogue_model::read::View::Transcript => String::new(),
        view => format!(" --view {view}"),
    };
    failure.error.message.push_str(&format!(
        "; run `polylogue {} read{view}` again to start over",
        polylogue_render::shell_quote(&target)
    ));
    // "Start over" means the beginning, as the sentence says, not the seen
    // mark the stale cursor read from.
    let mut again = step_from(request, None);
    again.options.from = None;
    let rows = Rows {
        selection: Some(target),
        steps: vec![Step::Read(Box::new(again))],
        ..Rows::default()
    };
    match serde_json::to_value(rows) {
        Ok(arguments) => {
            failure.error = failure
                .error
                .with_suggestion(command_surface::Suggestion::Request(
                    crate::client::request(
                        "rows",
                        arguments,
                        polylogue_store::user_writes::new_request_id(),
                    ),
                ));
        }
        Err(error) => failure
            .error
            .message
            .push_str(&format!("; fresh request options: {error}")),
    }
    failure
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
            steps: vec![Step::Read(Box::new(step_from(request, cursor)))],
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
