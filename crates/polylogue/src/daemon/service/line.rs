//! The `rows` operation: one line. Its selection is resolved once, before
//! the first step, and every step acts on those rows in order, each its own
//! request (`<line request ID>-<n>`, so a retried line repeats each write
//! once). A failing step stops the line; earlier steps keep their effects.
//! A selection with no steps is the listing; a line ending in a streamed
//! read streams.
use super::{Failure, answers, handlers, query_targets};
use crate::commands::{Caller, Command, Surface, ToolAccess};
use crate::daemon::handler::{Call, CountReplyMap, Handler, Streams};
use crate::daemon::verbs::{
    self, Given, RowRule, Selection, StepCall, read::LineReads, refused_pick, rows_rule,
};
use crate::protocol::line::{
    LineAnswer, LineExplanation, Resolved, Rows, SelectionForm, Step, StepExplanation, StepOutcome,
    StepReport, StepsAnswer,
};
use command_surface::{Access, Answer, Envelope, Outcome, TextStyle};
use polylogue_model::read::ReadDelivery;
use polylogue_model::user::UserWrite;
use polylogue_model::{Code, Reference};
use polylogue_reading::operation::Context;
use polylogue_store::read::limit::ReadLimit;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) struct Line;

impl Handler for Line {
    type Arguments = Rows;
    type Answer = LineAnswer;
    const PAGES: bool = true;
    const BUDGETED: bool = true;
    const READS_FIRST: bool = true;
    fn count_reply(rows: &Rows) -> Option<CountReplyMap<Self::Answer>> {
        if rows.steps.is_empty() && !rows.explain {
            Some(listing_answer)
        } else {
            None
        }
    }
    const PRINT: fn(&Self::Answer, TextStyle) -> String = crate::cli::print::line;
    fn read_only(rows: &Rows, _: &Command, caller: &Caller) -> bool {
        rows.access() == ToolAccess::Read
            && !(caller.sender().is_some()
                && rows.steps.iter().any(|s| matches!(s, Step::Brief(_))))
    }
    fn default_deadline(rows: &Rows) -> bool {
        !streams(rows)
    }
    fn tool_access(rows: &Rows, _: &Command) -> ToolAccess {
        rows.access()
    }
    fn access(rows: &Rows, _: Access) -> Access {
        if rows.access() > ToolAccess::Read {
            Access::Write
        } else {
            Access::Read
        }
    }
    fn query_text(rows: &Rows) -> Option<&str> {
        query_selection(rows)
    }
    fn ranking_query(rows: &Rows, _: &Command) -> Option<String> {
        if rows.explain {
            None
        } else {
            query_selection(rows).map(str::to_owned)
        }
    }
    fn run(rows: Rows, call: &Call<'_>) -> Result<Answer<LineAnswer>, Failure> {
        if let Some(failure) = refused_surface(&rows, call)? {
            let next = match failure.suggestion() {
                Some(command_surface::Suggestion::Request(request)) => Some(request.clone()),
                _ => None,
            };
            let answer = failure.answer();
            return Ok(match next {
                Some(request) => answer.with_next("read", request),
                None => answer,
            });
        }
        let given = given(&rows)?;
        if rows.explain {
            return explain(&rows, given, call);
        }
        check(&rows, &given)?;
        if rows.steps.is_empty() {
            return listing(rows, given, call);
        }
        let limit = ReadLimit::current();
        let selection = prepare(&rows, given, call)?;
        let resolved = summary(&selection);
        if rows.dry_run {
            let steps = rows
                .steps
                .iter()
                .enumerate()
                .map(|(index, step)| {
                    let request_id = step_request(call.request_id, index);
                    let step_call = StepCall::new(call, &request_id);
                    let outcome = match verbs::would(step, &selection, &step_call) {
                        Ok(would) => StepOutcome::WouldDo {
                            would,
                            refusal: None,
                        },
                        Err(failure) => StepOutcome::WouldDo {
                            would: format!("would {} {}", step.verb(), selection.described()),
                            refusal: Some(failure.error),
                        },
                    };
                    StepReport {
                        verb: step.verb().into(),
                        request_id: Some(request_id),
                        outcome,
                    }
                })
                .collect();
            return Ok(Answer::ok(LineAnswer::Steps(StepsAnswer {
                resolved,
                steps,
            })));
        }
        let mut answer = execute_steps(rows.steps, &selection, call, limit.as_ref(), None)?;
        if rows.return_selection {
            attach_population(&mut answer, &selection, call);
        }
        Ok(answer.map(LineAnswer::Steps))
    }
}

impl Streams for Line {
    fn streaming(rows: &Rows) -> bool {
        streams(rows)
    }
    fn produce(
        rows: Rows,
        call: &Call<'_>,
        context: &Arc<Context>,
        output: &mut dyn Write,
    ) -> Result<(), Failure> {
        // The stream's call carries no request ID; the accepted request does.
        let request_id = context
            .accepted_request()
            .map(|request| request.request_id.clone())
            .unwrap_or_default();
        let call = Call {
            request_id: &request_id,
            ..call.copy()
        };
        if let Some(failure) = refused_surface(&rows, &call)? {
            return Err(failure);
        }
        let given = given(&rows)?;
        check(&rows, &given)?;
        let limit = ReadLimit::current();
        let selection = prepare(&rows, given, &call)?;
        let mut answer = execute_steps(
            rows.steps,
            &selection,
            &call,
            limit.as_ref(),
            Some(Stream { context, output }),
        )?;
        if rows.return_selection {
            attach_population(&mut answer, &selection, &call);
        }
        crate::protocol::stream::event(output, "answer", &answer.map(LineAnswer::Steps))
            .map_err(Failure::io)
    }
}

/// One streamed read makes the line an ordered stream; a follow remains last.
fn streams(rows: &Rows) -> bool {
    !rows.explain
        && !rows.dry_run
        && rows.steps.iter().any(|step| {
            matches!(step,
        Step::Read(read) if read.follow || read.delivery == ReadDelivery::Stream)
        })
}

/// The selection, when it is a query rather than a reference.
fn query_selection(rows: &Rows) -> Option<&str> {
    match (&rows.selection, rows.references.is_empty()) {
        (Some(text), true) if named(text).is_none() => Some(text),
        _ => None,
    }
}

/// A reference that names rows one by one: a session, message, block,
/// annotation, alternate, attachment, page, role, tool call or kept result,
/// or a provider conversation URL, which names its web-chat session whether
/// or not the archive holds it yet. Any other text, a saved query's name
/// included, is a query.
fn named(text: &str) -> Option<Reference> {
    if let Some(session) = polylogue_sources::origin::conversation_session(text) {
        return Some(Reference::Session(session));
    }
    Reference::parse(text.trim()).ok().filter(|reference| {
        polylogue_query::catalog::reference_unit(reference).is_some()
            || matches!(
                reference,
                Reference::Result { .. }
                    | Reference::Cwd(_)
                    | Reference::Repo(_)
                    | Reference::File { .. }
                    | Reference::Page(_)
                    | Reference::Role(_)
                    | Reference::Alternate { .. }
                    | Reference::ToolCall(..)
                    | Reference::Attachment { .. }
            )
    })
}

/// What the line names.
fn given(rows: &Rows) -> Result<Given, Failure> {
    for text in rows.selection.iter().chain(&rows.references) {
        let text = text.trim();
        if text.is_empty() {
            return Err(Failure::bad_request("selection must not be empty"));
        }
        if text == "@all"
            || text
                .strip_prefix('@')
                .is_some_and(|tail| tail.starts_with(|c: char| c.is_ascii_digit()))
        {
            return Err(Failure::new(
                Code::RowNumbersOffTerminal,
                "row numbers are resolved by the command line; use the references printed above",
            ));
        }
    }
    match (&rows.selection, rows.references.as_slice()) {
        (Some(_), [_, ..]) => Err(Failure::bad_request(
            "a line takes a selection or references, not both",
        )),
        (None, []) => Ok(Given::None),
        (None, references) => references
            .iter()
            .map(|text| Reference::parse(text.trim()).map_err(Failure::bad_value))
            .collect::<Result<_, _>>()
            .map(Given::References),
        (Some(text), []) => Ok(match named(text) {
            Some(reference) => Given::Reference(reference),
            None => Given::Query(text.clone()),
        }),
    }
}

/// What the rows are, in the plural: the kind their references share
/// (`sessions`, `messages`, `annotations`), else `rows`.
fn plural_kind(references: &[Reference]) -> String {
    let kind = |reference: &Reference| {
        let text = reference.to_string();
        text.split_once(':')
            .map(|(kind, _)| kind.to_owned())
            .unwrap_or_default()
    };
    let first = references.first().map(kind).unwrap_or_default();
    if first.is_empty() || references.iter().any(|reference| kind(reference) != first) {
        return "rows".into();
    }
    match first.strip_suffix('y') {
        Some(stem) => format!("{stem}ies"),
        None => format!("{first}s"),
    }
}

/// The request to continue this line, retaining its envelope options.
fn repeated(rows: &Rows, call: &Call<'_>) -> Result<command_surface::RunnableRequest, Failure> {
    let context = call.context()?;
    let mut request =
        context
            .accepted_request()
            .cloned()
            .unwrap_or_else(|| command_surface::RunnableRequest {
                operation: "rows".into(),
                arguments: serde_json::Value::Null,
                request_id: call.request_id.into(),
                deadline: Default::default(),
                budget_tokens: call.budget,
                cursor: call.cursor.map(str::to_owned),
                fresh: false,
                format: Default::default(),
            });
    if request.operation == "rows"
        && serde_json::from_value::<Rows>(request.arguments.clone())
            .is_ok_and(|original| original == *rows)
    {
        return Ok(request);
    }
    request.arguments = serde_json::to_value(rows).map_err(Failure::internal)?;
    Ok(request)
}

/// Refusals that offer a finite read of the requested rows.
fn refused_surface(rows: &Rows, call: &Call<'_>) -> Result<Option<Failure>, Failure> {
    if let Some(id) = rows
        .selection
        .as_deref()
        .and_then(|text| text.trim().strip_prefix("answer:"))
    {
        let mut read = Rows {
            selection: Some(format!("run:{id}")),
            ..Rows::default()
        };
        read.steps.push(Step::Read(Box::default()));
        let failure = Failure::bad_request(format!(
            "answer references are query runs; use polylogue run:{id} read"
        ));
        return Ok(Some(
            failure
                .error
                .with_suggestion(command_surface::Suggestion::Request(repeated(&read, call)?))
                .into(),
        ));
    }
    if let Some(reference) = rows
        .selection
        .iter()
        .chain(&rows.references)
        .filter_map(|text| Reference::parse(text.trim()).ok())
        .find(|reference| matches!(reference, Reference::Run(_)))
        && rows.steps.iter().any(|step| !matches!(step, Step::Read(_)))
    {
        let mut read = Rows {
            selection: Some(reference.to_string()),
            ..Rows::default()
        };
        read.steps.push(Step::Read(Box::default()));
        return Ok(Some(
            Failure::bad_request(format!(
                "a run holds no rows; use polylogue {reference} read"
            ))
            .error
            .with_suggestion(command_surface::Suggestion::Request(repeated(&read, call)?))
            .into(),
        ));
    }
    if call.caller.surface == Surface::Mcp
        && rows
            .steps
            .iter()
            .any(|step| matches!(step, Step::Read(read) if read.follow))
    {
        let mut read = rows.clone();
        for step in &mut read.steps {
            if let Step::Read(read) = step {
                read.follow = false;
            }
        }
        let failure = Failure::bad_request("a model cannot hold a stream; read again to continue");
        return Ok(Some(
            failure
                .error
                .with_suggestion(command_surface::Suggestion::Request(repeated(&read, call)?))
                .into(),
        ));
    }
    Ok(None)
}

/// The refusals of a line's shape, before anything resolves.
fn check(rows: &Rows, given: &Given) -> Result<(), Failure> {
    if !rows.dry_run && rows.steps.iter().any(|step| matches!(step, Step::Pick(_))) {
        return Err(refused_pick());
    }
    if let Some(position) = rows
        .steps
        .iter()
        .position(|step| matches!(step, Step::Read(read) if read.follow))
        && position + 1 != rows.steps.len()
    {
        return Err(Failure::bad_request("read --follow must be the last step"));
    }
    if rows.wait.is_some() && !rows.steps.is_empty() {
        return Err(Failure::bad_request("--wait needs a listing without verbs"));
    }
    if *given == Given::None {
        match rows.steps.as_slice() {
            [] => return Err(Failure::bad_request("a listing needs a selection")),
            [Step::Brief(_) | Step::Resume(_)] => {}
            [Step::Read(_)] => {
                return Err(Failure::bad_request(
                    "read needs rows: name a selection; on a terminal the command line opens the picker",
                ));
            }
            [first, ..] => {
                return Err(Failure::bad_request(format!(
                    "{} needs rows first: name a selection",
                    first.verb()
                )));
            }
        }
    }
    Ok(())
}

/// The selection resolved once, refused when a write or act would reach
/// several rows of a query without `all`.
fn prepare(rows: &Rows, given: Given, call: &Call<'_>) -> Result<Selection, Failure> {
    let durable = !rows.dry_run && rows.access() > ToolAccess::Read;
    let intended = UserWrite::ResolveLine {
        request: serde_json::to_value(repeated(rows, call)?).map_err(Failure::internal)?,
        resolved: None,
        retained: None,
    };
    let original = if durable {
        answers::accepted_write(&intended, call)?
    } else {
        None
    };
    let selection = match original {
        Some(UserWrite::ResolveLine {
            resolved, retained, ..
        }) => Selection {
            question: match &given {
                Given::Query(text) => Some(text.clone()),
                _ => None,
            },
            given,
            resolved,
            retained,
        },
        Some(_) => return Err(Failure::internal("accepted line has another request kind")),
        None => resolve(given, &rows.steps, rows.return_selection, call)?,
    };
    if let (Given::Query(query), Some(resolved)) = (&selection.given, &selection.resolved)
        && resolved.len() > 1
        && !rows.all
        && rows.access() > ToolAccess::Read
    {
        // The refusal carries this same line with `all` set, ready to run.
        let mut retry = repeated(rows, call)?;
        retry.arguments["all"] = serde_json::Value::Bool(true);
        return Err(crate::protocol::failure(
            Code::NeedsAll,
            format!(
                "{} {} of {query:?} match; a write or act on all of them needs --all",
                resolved.len(),
                plural_kind(resolved)
            ),
        )
        .with_candidates(resolved.iter().take(5).map(ToString::to_string).collect())
        .with_suggestion(command_surface::Suggestion::Request(retry))
        .into());
    }
    if durable {
        let UserWrite::ResolveLine { request, .. } = intended else {
            unreachable!()
        };
        let accepted = answers::write(
            call.daemon,
            call.invocation,
            call.request_id.into(),
            UserWrite::ResolveLine {
                request,
                resolved: selection.resolved.clone(),
                retained: selection.retained.clone(),
            },
        );
        if let Outcome::Error(error) = accepted.outcome {
            return Err(error.into());
        }
        // A concurrent equal request may have accepted another snapshot first.
        // Read back the journal's body rather than keeping our speculative rows.
        if let Some(UserWrite::ResolveLine {
            resolved, retained, ..
        }) = answers::accepted_write(
            &UserWrite::ResolveLine {
                request: serde_json::to_value(repeated(rows, call)?).map_err(Failure::internal)?,
                resolved: None,
                retained: None,
            },
            call,
        )? {
            return Ok(Selection {
                resolved,
                retained,
                ..selection
            });
        }
        return Err(Failure::internal(
            "accepted line is missing from the request journal",
        ));
    }
    Ok(selection)
}

/// Resolves the rows when the line has two or more steps, a step that
/// writes or acts, or a one-row step. A single read-only step is not
/// resolved: it reads the selection as it streams.
fn resolve(
    given: Given,
    steps: &[Step],
    return_selection: bool,
    call: &Call<'_>,
) -> Result<Selection, Failure> {
    // An exact saved name with withdrawal denotes its authored record. Query
    // expressions (including parameters, revisions and pipelines) denote results.
    if steps.iter().any(|step| matches!(step, Step::Withdraw(_)))
        && let Given::Query(text) = &given
        && let Ok(query) = polylogue_query::parse::parse(text)
        && query.attach.is_empty()
        && query.transforms.is_empty()
        && let polylogue_query::ast::Source::Saved {
            name,
            arguments,
            condition,
            ..
        } = &query.source
        && arguments.is_empty()
        && condition.is_none()
        && !name.contains('@')
        && (Reference::Query {
            key: name.clone(),
            parameters: vec![],
        })
        .question_id()
        .is_none()
    {
        let target = call.reader()?.saved_query_target(name)?.ok_or_else(|| {
            Failure::not_found(format!("no current authored saved query query:{name}"))
        })?;
        let mut selection = resolve(Given::Reference(target), steps, return_selection, call)?;
        selection.question = Some(text.clone());
        selection.given = given;
        return Ok(selection);
    }
    let question = match &given {
        Given::Query(text) => Some(text.clone()),
        _ => None,
    };
    let result_rows = matches!(&given, Given::Reference(Reference::Result { .. }))
        && (steps.len() > 1 || steps.iter().any(|step| step.access() > ToolAccess::Read));
    if return_selection || result_rows || steps.iter().any(|step| matches!(step, Step::Save(_))) {
        let text = match &given {
            Given::Query(text) => Some(text.clone()),
            Given::Reference(reference) => Some(reference.to_string()),
            Given::References(references) => Some(
                references
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" OR "),
            ),
            Given::None => None,
        };
        if let Some(text) = text {
            let (retained, query_rows) = retain_selection(&text, call)?;
            if return_selection && query_rows.is_none() {
                return Err(Failure::bad_request(
                    "pick needs archive rows with native identities",
                ));
            }
            if query_rows.is_none()
                && let Some(step) = steps.iter().find(|step| rows_rule(step) != RowRule::Any)
            {
                return Err(Failure::bad_request(format!(
                    "{} needs archive rows; the selection is grouped or measured",
                    step.verb()
                )));
            }
            let resolved = match &given {
                Given::Reference(Reference::Result { .. }) => query_rows,
                Given::Reference(reference) => Some(vec![reference.clone()]),
                Given::References(references) => Some(references.clone()),
                _ => query_rows,
            };
            return Ok(Selection {
                given,
                question,
                resolved,
                retained: Some(Box::new(retained)),
            });
        }
    }
    let needed = steps.len() > 1
        || steps
            .iter()
            .any(|step| step.access() > ToolAccess::Read || rows_rule(step) == RowRule::One);
    let resolved = match &given {
        Given::None => None,
        Given::Reference(reference) => Some(vec![reference.clone()]),
        Given::References(references) => Some(references.clone()),
        Given::Query(_) if !needed => None,
        Given::Query(text) => {
            let reader = call.reader()?;
            if selects_rows(call, &reader, text)? {
                Some(query_targets(call.daemon, &reader, text, call.invocation)?)
            } else if let Some(step) = steps.iter().find(|step| rows_rule(step) != RowRule::Any) {
                return Err(Failure::bad_request(format!(
                    "{} needs archive rows; the selection is grouped or measured",
                    step.verb()
                )));
            } else {
                None
            }
        }
    };
    Ok(Selection {
        given,
        resolved,
        question,
        retained: None,
    })
}

/// Capture Save's inputs during the same resolution that establishes the
/// line population. Only derived relation values remain in the journal.
fn retain_selection(
    text: &str,
    call: &Call<'_>,
) -> Result<(polylogue_model::user::LineSelection, Option<Vec<Reference>>), Failure> {
    let reader = call.reader()?;
    let daemon = call.daemon;
    crate::daemon::query::with_ranking(
        &reader,
        &daemon.warm,
        &daemon.config,
        |embeddings, ranking_identity, near| {
            crate::daemon::query::capture_selection(
                &reader,
                &crate::daemon::query::Ask {
                    text,
                    cursor: None,
                    page_size: None,
                    why: false,
                    session: None,
                    context: call.context()?,
                    embeddings,
                    ranking_identity,
                    near: Some(near),
                    fields: &[],
                },
            )
        },
    )
}

/// Whether a query's rows are archive rows rather than groups, a summary
/// or a measurement.
fn selects_rows(
    call: &Call<'_>,
    reader: &polylogue_store::Reader<'_>,
    text: &str,
) -> Result<bool, Failure> {
    let ask = crate::daemon::query::Ask {
        text,
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        context: call.context()?,
        embeddings: false,
        ranking_identity: None,
        near: None,
        fields: &[],
    };
    crate::daemon::query::selects_rows(reader, &ask)
}

/// The resolved rows as the answer states them.
fn summary(selection: &Selection) -> Option<Resolved> {
    let resolved = selection.resolved.as_ref()?;
    Some(Resolved {
        question: selection.question.clone(),
        count: resolved.len() as u64,
        named: !matches!(selection.given, Given::Query(_)),
        first: resolved.iter().take(20).map(ToString::to_string).collect(),
        population: None,
    })
}

/// A requested full population is auxiliary data on the same line response.
/// Its failure preserves all committed step reports and their undo lines.
fn attach_population(answer: &mut Answer<StepsAnswer>, selection: &Selection, call: &Call<'_>) {
    let population = (|| {
        let reader = call.reader()?;
        let text = selection
            .question
            .clone()
            .or_else(|| selection.query_text())
            .ok_or_else(|| Failure::bad_request("pick needs a selection"))?;
        let daemon = call.daemon;
        crate::daemon::query::with_ranking(
            &reader,
            &daemon.warm,
            &daemon.config,
            |embeddings, ranking_identity, near| {
                crate::daemon::query::accepted_population(
                    &reader,
                    crate::daemon::query::Ask {
                        text: &text,
                        cursor: None,
                        page_size: None,
                        why: false,
                        session: None,
                        context: call.context()?,
                        embeddings,
                        ranking_identity,
                        near: Some(near),
                        fields: &[],
                    },
                    selection,
                )
            },
        )
    })()
    .unwrap_or_else(|error: Failure| error.answer());
    let steps = match &mut answer.outcome {
        Outcome::Ok(value) | Outcome::Empty(value) | Outcome::Degraded { value, .. } => value,
        Outcome::Error(_) => return,
    };
    if let Some(resolved) = &mut steps.resolved {
        resolved.population = Some(Box::new(population));
    }
}

fn step_request(line: &str, index: usize) -> String {
    format!("{line}-{}", index + 1)
}

/// The stream carries each batch of reports once, before its next read.
struct Stream<'a> {
    context: &'a Arc<Context>,
    output: &'a mut dyn Write,
}
impl Stream<'_> {
    fn reports(
        &mut self,
        selection: &Selection,
        reports: &mut Vec<StepReport>,
        envelope: &mut Envelope,
    ) -> Result<(), Failure> {
        if reports.is_empty() {
            return Ok(());
        }
        let answer = Answer {
            outcome: Outcome::Ok(LineAnswer::Steps(StepsAnswer {
                resolved: summary(selection),
                steps: std::mem::take(reports),
            })),
            envelope: std::mem::take(envelope),
        };
        crate::protocol::stream::event(self.output, "steps", &answer).map_err(Failure::io)
    }
    fn read(
        &mut self,
        read: &crate::protocol::steps::ReadStep,
        selection: &Selection,
        call: &StepCall<'_>,
    ) -> Result<verbs::StepDone, Failure> {
        let target = verbs::read::target(selection)?;
        if read.follow {
            if read.until_idle {
                return Err(verbs::not_built("read", "until-idle"));
            }
            let marks =
                super::follow::run(call.call.daemon, self.context, &target, None, self.output)?;
            return verbs::delegated(Answer::ok(marks), |marks| {
                crate::protocol::line::StepAnswer::Followed(Box::new(marks))
            });
        }
        let call = Call {
            cursor: read.cursor.as_deref(),
            ..call.call.copy()
        };
        let answer = super::read::stream(
            verbs::read::arguments(read, target)?,
            &call,
            self.context,
            self.output,
            &LineReads {
                daemon: call.daemon,
                accepted: self.context.accepted_request(),
            },
            selection.resolved.as_deref(),
        )?;
        verbs::delegated(answer, |complete| {
            crate::protocol::line::StepAnswer::Streamed(Box::new(complete))
        })
    }
}

/// Selection, step identities, failures and outcomes have one executor in
/// finite and streamed delivery. A stream only changes where read records go.
fn execute_steps(
    steps: Vec<Step>,
    selection: &Selection,
    call: &Call<'_>,
    limit: Option<&ReadLimit>,
    mut stream: Option<Stream<'_>>,
) -> Result<Answer<StepsAnswer>, Failure> {
    let mut reports = Vec::with_capacity(steps.len());
    let mut envelope = Envelope::default();
    let mut outcome = Outcome::Empty(());
    let mut failed = false;
    for (index, step) in steps.into_iter().enumerate() {
        let verb = step.verb();
        if failed {
            reports.push(StepReport {
                verb: verb.into(),
                request_id: None,
                outcome: StepOutcome::NotRun,
            });
            continue;
        }
        let request_id = step_request(call.request_id, index);
        let step_call = StepCall::new(call, &request_id);
        let work = || {
            if let Some(limit) = limit {
                limit.check()?;
            }
            match (&mut stream, &step) {
                (Some(stream), Step::Read(read))
                    if read.follow || read.delivery == ReadDelivery::Stream =>
                {
                    stream.reports(selection, &mut reports, &mut envelope)?;
                    stream.read(read, selection, &step_call)
                }
                _ => verbs::run_step(step, selection, &step_call),
            }
        };
        // Each step inherits the original line deadline. A preceding writer
        // handoff must not remove that bound from later required reads.
        let result = match limit {
            Some(limit) => limit.scope(work),
            None => work(),
        };
        let reported = match result {
            Ok(done) => {
                verbs::merge_outcome(&mut outcome, done.outcome);
                envelope.notes.extend(done.envelope.notes);
                if !done.envelope.next.is_empty() {
                    envelope.next = done.envelope.next;
                }
                StepOutcome::Done {
                    reply: done.reply,
                    undo: done.undo,
                }
            }
            Err(failure) => {
                failed = true;
                if matches!(outcome, Outcome::Empty(())) {
                    outcome = Outcome::Ok(());
                }
                StepOutcome::Failed {
                    error: failure.error,
                }
            }
        };
        reports.push(StepReport {
            verb: verb.into(),
            request_id: Some(request_id),
            outcome: reported,
        });
        if let Some(stream) = &mut stream {
            stream.reports(selection, &mut reports, &mut envelope)?;
        }
    }
    Ok(Answer {
        outcome: outcome.map(|()| StepsAnswer {
            resolved: summary(selection),
            steps: reports,
        }),
        envelope,
    })
}

fn listing_answer(answer: crate::daemon::query::CountReply) -> Answer<LineAnswer> {
    answer.map(|found| LineAnswer::Listing(Box::new(found)))
}

/// How long a `--wait` waits: over MCP at most 50 seconds, which a model
/// can hold a call open for.
pub(super) fn listing_wait(wait: &str, surface: Surface) -> Result<Duration, Failure> {
    let millis = crate::signals::duration(wait)
        .ok_or_else(|| Failure::bad_value("--wait needs a duration such as 30s or 30m"))?;
    let duration = Duration::from_millis(millis as u64);
    Ok(if surface == Surface::Mcp {
        duration.min(Duration::from_secs(50))
    } else {
        duration
    })
}

/// The selection alone: its rows, a page at a time.
fn listing(rows: Rows, given: Given, call: &Call<'_>) -> Result<Answer<LineAnswer>, Failure> {
    let query = Selection {
        given,
        resolved: None,
        question: None,
        retained: None,
    }
    .query_text()
    .ok_or_else(|| Failure::bad_request("a listing needs a selection"))?;
    let arguments = crate::protocol::query::Arguments {
        query,
        limit: rows.page_size.map(std::num::NonZeroU64::get),
        why: rows.why,
        session: None,
        fields: rows.fields.clone(),
    };
    let Some(wait) = rows.wait.as_deref() else {
        return handlers::Query::run(arguments, call).map(listing_answer);
    };
    let mut duration = listing_wait(wait, call.caller.surface)?;
    let limit = ReadLimit::current();
    if let Some(left) = limit.as_ref().and_then(ReadLimit::remaining) {
        duration = duration.min(left.saturating_sub(super::follow::REPLY_MARGIN));
    }
    let until = Instant::now() + duration;
    let commits = call
        .daemon
        .writes
        .as_ref()
        .map(|writer| writer.completion_commits());
    let _wake = match (&commits, &limit) {
        (Some(commits), Some(limit)) => {
            let wake = commits.wake_handle();
            Some(limit.cancellation.on_cancel(move || {
                let _ = wake.try_send(());
            }))
        }
        _ => None,
    };
    loop {
        super::accepting(call.daemon)?;
        if let Some(limit) = &limit {
            limit.check()?;
        }
        let answer = listing_answer(handlers::waiting_query(&arguments, call)?);
        if !matches!(answer.outcome, command_surface::Outcome::Empty(_)) {
            return Ok(answer);
        }
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Ok(answer.with_next("wait again", repeated(&rows, call)?));
        }
        let left = limit
            .as_ref()
            .and_then(ReadLimit::remaining)
            .map_or(left, |remaining| left.min(remaining));
        ReadLimit::set_stage("listing wait");
        crate::daemon::executor::outside(|| match &commits {
            Some(commits) => {
                let _ = commits.recv_timeout(left);
            }
            None => std::thread::sleep(left),
        });
    }
}

/// How the line is read and what each step would do; nothing resolves.
fn explain(rows: &Rows, given: Given, call: &Call<'_>) -> Result<Answer<LineAnswer>, Failure> {
    let form = match &given {
        Given::None => SelectionForm::None,
        Given::Query(_) => SelectionForm::Query,
        Given::Reference(Reference::Result { .. }) => SelectionForm::KeptResult,
        Given::Reference(_) => SelectionForm::Reference,
        Given::References(_) => SelectionForm::References,
    };
    let plan = match &given {
        Given::Query(text) => Some(plan(call, text)?),
        Given::Reference(reference)
            if polylogue_query::catalog::reference_unit(reference).is_some()
                || matches!(reference, Reference::Result { .. }) =>
        {
            Some(plan(call, &reference.to_string())?)
        }
        _ => None,
    };
    let selection = Selection {
        question: match &given {
            Given::Query(text) => Some(text.clone()),
            _ => None,
        },
        given,
        resolved: None,
        retained: None,
    };
    let steps = rows
        .steps
        .iter()
        .enumerate()
        .map(|(index, step)| {
            let request_id = step_request(call.request_id, index);
            let step_call = StepCall::new(call, &request_id);
            let plain = || format!("would {} {}", step.verb(), selection.described());
            // A step that acts on rows one by one cannot look at a query's
            // rows here, which are not resolved.
            let would = if selection.question.is_some() && rows_rule(step) != RowRule::Any {
                plain()
            } else {
                verbs::would(step, &selection, &step_call)
                    .unwrap_or_else(|failure| format!("{}; refused: {}", plain(), failure.message))
            };
            StepExplanation {
                verb: step.verb().into(),
                would,
                access: step.access(),
            }
        })
        .collect();
    Ok(Answer::ok(LineAnswer::Explained(LineExplanation {
        form,
        plan,
        steps,
    })))
}

/// The selection query's plan.
fn plan(call: &Call<'_>, text: &str) -> Result<serde_json::Value, Failure> {
    let reader = call.reader()?;
    let context = call.context()?;
    let d = call.daemon;
    crate::daemon::query::with_ranking(&reader, &d.warm, &d.config, |embeddings, ranking, _| {
        crate::daemon::query::explain(
            &reader,
            crate::daemon::query::Ask {
                text,
                cursor: None,
                page_size: None,
                why: false,
                session: None,
                context,
                embeddings,
                ranking_identity: ranking,
                near: None,
                fields: &[],
            },
        )
    })
}

#[cfg(test)]
mod tests;
