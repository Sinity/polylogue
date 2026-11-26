//! Runs the same rows request as the other surfaces and delivers each reply in order.
use super::{line::*, picker};
use crate::protocol::line::{LineAnswer, StepAnswer, StepOutcome};
use command_surface::{Answer, Outcome, RunnableRequest};
use polylogue_model::Code;
use polylogue_store::Archive;
use serde_json::{Value, json};
use std::{
    io::{self, IsTerminal, Read, Write},
    process::ExitCode,
    sync::Arc,
};

/// Rendering state belongs to one logical CLI line, including a picker boundary.
pub(crate) struct OutputState {
    pub offset: usize,
    undo: Vec<String>,
    outcome: Outcome<()>,
    failed: Option<u8>,
    population: Option<Box<Answer<crate::protocol::query::Answer>>>,
}
impl Default for OutputState {
    fn default() -> Self {
        Self {
            offset: 0,
            undo: vec![],
            outcome: Outcome::Empty(()),
            failed: None,
            population: None,
        }
    }
}
impl OutputState {
    fn flush_undo(&mut self) {
        for line in self.undo.drain(..) {
            eprintln!("undo: {}", polylogue_render::terminal::escape(&line));
        }
    }
    pub(crate) fn accept<T: Clone>(&mut self, answer: &Answer<T>) {
        crate::daemon::verbs::merge_outcome(&mut self.outcome, answer.outcome.clone().map(|_| ()));
    }
    pub(crate) fn exit_code(&self) -> u8 {
        Answer {
            outcome: self.outcome.clone(),
            envelope: Default::default(),
        }
        .exit_code()
    }
}
impl Drop for OutputState {
    fn drop(&mut self) {
        self.flush_undo();
    }
}

pub(crate) fn terminal() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}
pub(crate) fn request(parsed: &ParsedLine) -> Result<RunnableRequest, Refusal> {
    let (selection, references) = match &parsed.selection {
        Some(SelectionWord::Stdin) => {
            return Err(Refusal::new(
                Code::BadRequest,
                "standard input has not been read",
            ));
        }
        Some(SelectionWord::Text(text)) => (Some(text.clone()), vec![]),
        Some(SelectionWord::References(references)) => (None, references.clone()),
        None => (None, vec![]),
    };
    let arguments = json!({"selection": selection, "references": references,
        "steps": parsed.steps.iter().map(|step| step.step.clone()).collect::<Vec<_>>(),
        "all": parsed.line.all, "dry_run": parsed.line.dry_run, "explain": parsed.line.explain,
        "page_size": parsed.listing.page_size, "fields": parsed.listing.fields,
        "why": parsed.listing.why, "wait": parsed.line.wait });
    let mut words = vec!["rows".into()];
    if parsed.line.json {
        words.push("--json".into());
    }
    if parsed.line.fresh {
        words.push("--fresh".into());
    }
    for (name, value) in [
        ("--deadline", &parsed.line.deadline),
        ("--request-id", &parsed.line.request_id),
        ("--cursor", &parsed.listing.cursor),
        ("--budget", &parsed.listing.budget),
    ] {
        if let Some(value) = value {
            words.extend([name.into(), value.clone()]);
        }
    }
    let mut call = crate::commands::declarations()
        .parse_cli(&words, polylogue_store::user_writes::new_request_id())
        .map_err(|e| Refusal::new(Code::BadRequest, e.message))?
        .request;
    call.arguments = arguments;
    if let Some(wait) = &parsed.line.wait {
        let wait = crate::signals::duration(wait).ok_or_else(|| {
            Refusal::new(Code::BadValue, "--wait needs a duration such as 30s or 30m")
        })?;
        let wait = u64::try_from(wait).unwrap_or(u64::MAX);
        call.deadline = match call.deadline {
            command_surface::Deadline::None => command_surface::Deadline::None,
            command_surface::Deadline::Default => command_surface::Deadline::Milliseconds(wait),
            command_surface::Deadline::Milliseconds(ms) => {
                command_surface::Deadline::Milliseconds(ms.max(wait))
            }
        };
    }
    Ok(call)
}
/// Consume only the input the parser marked, and clear that intent after reading.
pub(crate) fn read_stdin(parsed: &mut ParsedLine, input: &mut impl Read) -> io::Result<()> {
    if !matches!(parsed.selection, Some(SelectionWord::Stdin))
        && parsed.steps.iter().all(|step| step.stdin.is_none())
    {
        return Ok(());
    }
    let mut text = String::new();
    input.read_to_string(&mut text)?;
    if matches!(parsed.selection, Some(SelectionWord::Stdin)) {
        parsed.selection = Some(SelectionWord::References(
            text.lines()
                .filter(|line| !line.is_empty())
                .map(str::to_owned)
                .collect(),
        ));
    } else {
        for step in &mut parsed.steps {
            if let Some(key) = step.stdin.take() {
                step.step[&key] = Value::String(text.clone());
            }
        }
    }
    Ok(())
}
pub(crate) fn refusal(error: Refusal, startup: &crate::config::Startup, json: bool) -> ExitCode {
    let mut failure = crate::protocol::failure(error.code, error.message);
    failure.position = error.position.map(|start| command_surface::InputSpan {
        offset: start,
        length: 1,
    });
    failure.hint = error.try_line.map(|line| format!("try: {line}"));
    if !error.candidates.is_empty() {
        failure
            .hint
            .get_or_insert_default()
            .push_str(&format!("\n{}", error.candidates.join("\n")));
    }
    super::fail_with(startup.output, json, &failure, &[])
}
fn call(
    archive: &Archive,
    request: &RunnableRequest,
) -> Result<Answer<Value>, command_surface::Failure> {
    crate::client::call_ready(archive, request).map_err(client_failure)
}
fn client_failure(error: io::Error) -> command_surface::Failure {
    crate::protocol::failure(
        if super::is_down(&error) {
            Code::DaemonRequired
        } else {
            Code::Io
        },
        error.to_string(),
    )
}
async fn picker_call(
    archive: &Archive,
    request: &RunnableRequest,
) -> picker::Result<Answer<Value>> {
    crate::client::call_ready_as(
        archive,
        request,
        &tokio_util::sync::CancellationToken::new(),
        crate::client::this_process(),
    )
    .await
    .map_err(client_failure)
    .map_err(crate::daemon::Failure::from)
}

fn decoded(answer: &Answer<Value>) -> Result<LineAnswer, command_surface::Failure> {
    if let Outcome::Error(error) = &answer.outcome {
        return Err(error.clone());
    }
    serde_json::from_value(answer.value().cloned().unwrap_or(Value::Null))
        .map_err(|error| crate::protocol::protocol_mismatch("rows", error.to_string()))
}
pub(crate) fn run(mut parsed: ParsedLine, startup: &crate::config::Startup) -> ExitCode {
    let json = parsed.line.json;
    if parsed.first == FirstWord::Nothing {
        let words = vec!["polylogue", "status"];
        let matches = super::command()
            .try_get_matches_from(words)
            .expect("status command");
        return super::run(&matches, startup);
    }
    if parsed.line.explain && !json {
        println!(
            "first word: {:?}, rule {}\nselection: {:?}",
            parsed.first,
            parsed.first.rule(),
            parsed.selection
        );
        for step in &parsed.steps {
            println!("{}: {}", step.verb, step.step);
        }
        println!("line options: {:?}", parsed.line);
    }
    if parsed.first == FirstWord::RowNumbers {
        let shown = match crate::row_numbers::last_listing(
            &startup.config.paths,
            &startup.config.archive,
        ) {
            Ok(shown) => shown,
            Err(error) => return refusal(Refusal::new(Code::BadRequest, error), startup, json),
        };
        let Some(SelectionWord::Text(selection)) = &parsed.selection else {
            unreachable!()
        };
        let references = match crate::row_numbers::select(selection, &shown) {
            Ok(references) => references,
            Err(error) => return refusal(Refusal::new(Code::BadRequest, error), startup, json),
        };
        if parsed
            .steps
            .iter()
            .any(|step| !matches!(step.verb, "read" | "brief" | "pick"))
            && !parsed.line.explain
        {
            let minutes = (startup.clock.now().as_millisecond() - shown.shown_at).max(0) / 60_000;
            eprintln!(
                "rows of '{}' shown {minutes} min ago",
                polylogue_render::terminal::escape(&shown.question)
            );
        }
        parsed.selection = Some(SelectionWord::References(references));
    }
    if let Err(error) = read_stdin(&mut parsed, &mut io::stdin().lock()) {
        return super::fail(startup.output, json, Code::Io, &error.to_string());
    }
    for step in &mut parsed.steps {
        if step.verb == "read" {
            let displayed = if json {
                io::stdout().is_terminal()
            } else {
                match step.to.as_ref() {
                    None | Some(Destination::Stdout) => io::stdout().is_terminal(),
                    Some(Destination::Pager) => true,
                    _ => false,
                }
            };
            if displayed {
                step.step["displayed"] = serde_json::Value::Bool(true);
            } else {
                step.step
                    .as_object_mut()
                    .expect("a parsed step is an object")
                    .remove("displayed");
            }
        }
    }
    let mut request = match request(&parsed) {
        Ok(request) => request,
        Err(error) => return refusal(error, startup, json),
    };
    let archive = Archive::at(&startup.config.archive);
    let mut output_state = OutputState::default();
    let needs_choice = parsed.selection.is_none()
        && matches!(
            parsed.steps.first().map(|step| step.verb),
            Some("read" | "resume")
        );
    let pick = parsed.steps.iter().position(|step| step.verb == "pick");
    if (needs_choice || pick.is_some()) && !parsed.line.explain && !parsed.line.dry_run {
        let split = pick.unwrap_or(0);
        let terminal_picker = terminal()
            || (pick.is_some() && io::stdin().is_terminal() && io::stderr().is_terminal());
        let frozen = pick.is_some() && split > 0 && terminal_picker && !json;
        let mut before = request.clone();
        before.request_id = format!(
            "{}-pick-{}",
            request.request_id,
            if frozen { "prefix" } else { "listing" }
        );
        before.arguments["steps"] = if frozen {
            json!(
                parsed.steps[..split]
                    .iter()
                    .map(|step| &step.step)
                    .collect::<Vec<_>>()
            )
        } else {
            json!([])
        };
        before.arguments["return_selection"] = json!(frozen);
        if needs_choice {
            if parsed.steps[0].verb == "resume" {
                before.arguments["steps"] = json!([parsed.steps[0].step]);
            } else {
                before.arguments["selection"] =
                    json!(crate::protocol::steps::usual_read_selection(
                        parsed.steps[0].step["follow"] == true
                    ));
            }
        }
        let question = before.arguments["selection"]
            .as_str()
            .unwrap_or(if needs_choice {
                "unfinished"
            } else {
                "references"
            })
            .to_owned();
        let mut choices = if frozen {
            let streamed = parsed.steps[..split].iter().any(|step| {
                step.verb == "read"
                    && (matches!(step.to, Some(Destination::File(_) | Destination::Dir(_)))
                        || step.step["delivery"] == "stream")
            });
            let code = if streamed {
                super::write_out::line_stream(startup, &before, &parsed, &mut output_state)
            } else {
                let answer = match call(&archive, &before) {
                    Ok(answer) => answer,
                    Err(error) => return super::fail_with(startup.output, json, &error, &[]),
                };
                let line = match decoded(&answer) {
                    Ok(line) => line,
                    Err(error) => return super::fail_with(startup.output, json, &error, &[]),
                };
                output(&parsed, &line, &answer, startup, &mut output_state)
            };
            if output_state.failed.is_some()
                || (code != ExitCode::SUCCESS
                    && code != ExitCode::from(1)
                    && code != ExitCode::from(5))
            {
                for step in &parsed.steps[split..] {
                    eprintln!("{} not run", step.verb);
                }
                return code;
            }
            let Some(population) = output_state.population.take() else {
                return super::fail(
                    startup.output,
                    json,
                    Code::DaemonProtocolMismatch,
                    "the line omitted its requested population",
                );
            };
            if let Outcome::Error(error) = &population.outcome {
                return super::fail_with(startup.output, json, error, &[]);
            }
            match population
                .value()
                .ok_or_else(|| crate::protocol::protocol_mismatch("rows", "missing population"))
                .and_then(query_candidates)
            {
                Ok(page) => page,
                Err(error) => return super::fail_with(startup.output, json, &error, &[]),
            }
        } else {
            let first = match call(&archive, &before) {
                Ok(answer) => answer,
                Err(error) => return super::fail_with(startup.output, json, &error, &[]),
            };
            match candidates(&first) {
                Ok(page) => page,
                Err(error) => return super::fail_with(startup.output, json, &error, &[]),
            }
        };
        let one = choices.rows.len() == 1 && choices.next.is_none();
        if (!terminal_picker || json) && !(needs_choice && one) {
            let mut refusal = Refusal::new(
                Code::Ambiguous,
                "choose references from the listing; this action needs a terminal picker",
            );
            refusal.candidates = choices
                .rows
                .iter()
                .take(5)
                .map(|row| {
                    shell_line(&[
                        row.reference.clone(),
                        parsed
                            .steps
                            .get(split + usize::from(pick.is_some()))
                            .map_or("read", |s| s.verb)
                            .into(),
                    ])
                })
                .collect();
            if needs_choice {
                refusal.candidates.extend([
                    "polylogue latest read".into(),
                    "polylogue SESSION read".into(),
                ]);
            }
            return self::refusal(refusal, startup, json);
        }
        let chosen = if one {
            vec![choices.rows[0].reference.clone()]
        } else {
            let source = Arc::new(Source {
                archive: startup.config.archive.clone(),
                paths: startup.config.paths.clone(),
                shown_at: startup.clock.now().as_millisecond(),
                question: question.clone(),
                request: before.clone(),
                frozen: frozen.then(|| choices.rows.clone()),
            });
            let selection = if frozen || (needs_choice && parsed.steps[0].verb == "resume") {
                picker::Selection::Ambiguous {
                    question,
                    candidates: std::mem::take(&mut choices.rows),
                }
            } else {
                picker::Selection::Query(question)
            };
            let options = picker::Options {
                selection,
                enter: parsed.steps[split + usize::from(pick.is_some())..]
                    .iter()
                    .map(|s| s.verb)
                    .collect::<Vec<_>>()
                    .join(" "),
                multiple: parsed
                    .steps
                    .first()
                    .is_none_or(|step| step.verb != "resume"),
            };
            match picker::open(source, options) {
                Ok(picker::Choice::Selected(references)) => references,
                Ok(picker::Choice::Cancelled) => {
                    return ExitCode::from(if output_state.exit_code() == 5 { 5 } else { 0 });
                }
                Err(error) => return super::fail_with(startup.output, json, &error.error, &[]),
            }
        };
        let after = split + usize::from(pick.is_some());
        if after == parsed.steps.len() {
            for reference in chosen {
                println!("{reference}");
            }
            return ExitCode::SUCCESS;
        }
        request.arguments["selection"] = Value::Null;
        request.arguments["references"] = json!(chosen);
        request.arguments["steps"] = json!(
            parsed.steps[after..]
                .iter()
                .map(|step| &step.step)
                .collect::<Vec<_>>()
        );
        parsed.steps.drain(..after);
        output_state.offset = 0;
        request.request_id = format!("{}-pick-suffix", request.request_id);
    }
    if !parsed.line.dry_run
        && !parsed.line.explain
        && super::attachment::reads_attachment(&request)
        && let Some(destination) = parsed
            .steps
            .first()
            .and_then(|step| step.to.as_ref())
            .filter(|destination| **destination != Destination::Stdout)
    {
        return match destination {
            Destination::File(path) => super::attachment::write_to(startup, &request, path, json),
            _ => super::fail(
                startup.output,
                json,
                Code::BadRequest,
                "an attachment's bytes are written to a file: read --to FILE",
            ),
        };
    }
    if parsed.steps.iter().any(|step| {
        step.verb == "read"
            && (matches!(step.to, Some(Destination::File(_) | Destination::Dir(_)))
                || step.step["delivery"] == "stream"
                || step.step["follow"] == true)
    }) && !parsed.line.dry_run
        && !parsed.line.explain
    {
        return super::write_out::line_stream(startup, &request, &parsed, &mut output_state);
    }
    let answer = match call(&archive, &request) {
        Ok(answer) => answer,
        Err(error) => return super::fail_with(startup.output, json, &error, &[]),
    };
    if json {
        println!(
            "{}",
            polylogue_render::terminal::json(
                &serde_json::to_value(&answer).expect("answer serializes"),
                false
            )
        );
        return ExitCode::from(failed_code(&answer).unwrap_or_else(|| answer.exit_code()));
    }
    let line = match decoded(&answer) {
        Ok(line) => line,
        Err(error) => return super::fail_with(startup.output, json, &error, &[]),
    };
    output(&parsed, &line, &answer, startup, &mut output_state)
}
fn failed_code(answer: &Answer<Value>) -> Option<u8> {
    let line = decoded(answer).ok()?;
    Some(line.failed_step()?.kind.exit_code())
}
fn output(
    parsed: &ParsedLine,
    line: &LineAnswer,
    answer: &Answer<Value>,
    startup: &crate::config::Startup,
    state: &mut OutputState,
) -> ExitCode {
    state.accept(answer);
    let offset = state.offset;
    match line {
        LineAnswer::Listing(found) => {
            if let crate::protocol::query::Answer::Rows(rows) = found.as_ref() {
                remember(
                    startup,
                    parsed
                        .selection
                        .as_ref()
                        .and_then(|selection| match selection {
                            SelectionWord::Text(text) => Some(text.as_str()),
                            _ => None,
                        })
                        .unwrap_or("references"),
                    rows,
                );
            }
            if let Err(error) = text_output(
                &super::print::query(found, super::style(startup.output)),
                parsed.listing.to.as_ref(),
                startup,
            ) {
                state.failed.get_or_insert(error.kind.exit_code());
                return super::fail_with(startup.output, false, &error, &[]);
            }
        }
        LineAnswer::Explained(explained) => println!(
            "{}",
            super::print::json(explained, super::style(startup.output))
        ),
        LineAnswer::Steps(steps) => {
            if let Some(population) = steps
                .resolved
                .as_ref()
                .and_then(|resolved| resolved.population.as_ref())
            {
                state.population = Some(population.clone());
            }
            let mut failed = None;
            let mut resume_plans = Vec::new();
            for (relative, report) in steps.steps.iter().enumerate() {
                let index = offset + relative;
                match &report.outcome {
                    StepOutcome::Failed { error } => {
                        state.failed.get_or_insert(error.kind.exit_code());
                        failed = Some(super::fail_with(startup.output, false, error, &[]));
                    }
                    StepOutcome::NotRun => eprintln!("{} not run", report.verb),
                    StepOutcome::WouldDo { would, refusal } => {
                        println!("{}", polylogue_render::terminal::escape(would));
                        if let Some(error) = refusal {
                            eprintln!("{}", polylogue_render::terminal::escape(&error.message));
                        }
                    }
                    StepOutcome::Done { reply, undo: lines } => {
                        state.undo.extend(lines.iter().cloned());
                        let to = parsed.steps.get(index).and_then(|step| step.to.as_ref());
                        let result = match reply {
                            StepAnswer::Read(page) => super::write_out::line_page(
                                startup,
                                page,
                                index + 1 == parsed.steps.len(),
                                to,
                                parsed
                                    .steps
                                    .get(index)
                                    .and_then(|step| step.step["format"].as_str()),
                            ),
                            StepAnswer::Streamed(_) | StepAnswer::Followed(_) => Ok(()),
                            StepAnswer::Attachment(page) => text_output(
                                &super::print::json(page, super::style(startup.output)),
                                to,
                                startup,
                            ),
                            StepAnswer::Brief(brief) => text_output(
                                &super::print::context(brief, super::style(startup.output)),
                                to,
                                startup,
                            ),
                            StepAnswer::Resume(resume) => {
                                if terminal()
                                    && !parsed.line.dry_run
                                    && let crate::protocol::resume::Answer::Plan(plan) =
                                        resume.as_ref()
                                {
                                    resume_plans.push(plan);
                                }
                                print!(
                                    "{}",
                                    super::print::resume(resume, super::style(startup.output))
                                );
                                Ok(())
                            }
                            StepAnswer::Saved(report) => {
                                eprint!(
                                    "{}",
                                    super::print::saved(report, super::style(startup.output))
                                );
                                Ok(())
                            }
                            other => {
                                eprintln!(
                                    "{}",
                                    super::print::json(other, super::style(startup.output))
                                );
                                Ok(())
                            }
                        };
                        if let Err(error) = result {
                            state.failed.get_or_insert(error.kind.exit_code());
                            failed = Some(super::fail_with(startup.output, false, &error, &[]));
                        }
                    }
                }
            }
            state.offset += steps.steps.len();
            state.flush_undo();
            if let Some(code) = failed {
                return code;
            }
            let count = resume_plans.len();
            for (index, plan) in resume_plans.into_iter().enumerate() {
                let code = super::run_plan(plan, startup.output, index + 1 == count);
                if code != ExitCode::SUCCESS {
                    return code;
                }
            }
        }
    }
    super::notes(
        &crate::protocol::notes(answer).cloned().collect::<Vec<_>>(),
        startup.output,
    );
    ExitCode::from(state.exit_code())
}
fn text_output(
    text: &str,
    to: Option<&Destination>,
    startup: &crate::config::Startup,
) -> Result<(), command_surface::Failure> {
    match to {
        Some(Destination::Clipboard | Destination::Browser) => {
            return Err(crate::protocol::failure(
                Code::NotBuilt,
                "this destination is not built yet",
            ));
        }
        Some(Destination::File(path)) => {
            let mut file = super::staged_file::StagedFile::new(path)
                .map_err(|error| crate::protocol::failure(Code::Io, error.to_string()))?;
            file.write_all(text.as_bytes())
                .and_then(|()| file.commit().map(|_| ()))
        }
        Some(Destination::Dir(_)) => {
            return Err(crate::protocol::failure(
                Code::BadRequest,
                "a text reply needs a file destination",
            ));
        }
        Some(Destination::Pager) => {
            let mut child = std::process::Command::new("sh")
                .arg("-c")
                .arg(&startup.pager)
                .stdin(std::process::Stdio::piped())
                .spawn()
                .map_err(|error| crate::protocol::failure(Code::Io, error.to_string()))?;
            child
                .stdin
                .take()
                .expect("piped stdin")
                .write_all(text.as_bytes())
                .and_then(|()| child.wait().map(|_| ()))
        }
        _ => io::stdout().write_all(text.as_bytes()),
    }
    .map_err(|error| crate::protocol::failure(Code::Io, error.to_string()))
}
fn remember(startup: &crate::config::Startup, question: &str, rows: &crate::protocol::query::Rows) {
    let native = rows.references.as_deref().unwrap_or_default();
    let shown = crate::row_numbers::ShownRows {
        question: question.into(),
        shown_at: startup.clock.now().as_millisecond(),
        rows: rows
            .rows
            .iter()
            .zip(native)
            .map(|(value, reference)| {
                let hits = value
                    .get("snippet")
                    .and_then(|snippet| snippet["message"].as_str())
                    .and_then(|message| polylogue_model::MessageId::parse(message).ok())
                    .map(|message| vec![polylogue_model::Reference::Message(message).to_string()])
                    .unwrap_or_default();
                crate::row_numbers::Row {
                    reference: reference.to_string(),
                    hits,
                }
            })
            .collect(),
    };
    crate::row_numbers::remember_listing(&startup.config.paths, &startup.config.archive, &shown);
}
fn candidates(answer: &Answer<Value>) -> Result<picker::Page, command_surface::Failure> {
    let line = decoded(answer)?;
    match line {
        LineAnswer::Listing(found) => query_candidates(&found),
        LineAnswer::Steps(steps) => {
            for report in steps.steps {
                if let StepOutcome::Done {
                    reply: StepAnswer::Resume(resume),
                    ..
                } = report.outcome
                    && let crate::protocol::resume::Answer::Sessions(sessions) = *resume
                {
                    return Ok(picker::Page {
                        count: Some(sessions.len() as u64),
                        next: None,
                        rows: sessions
                            .iter()
                            .map(|value| picker::Candidate {
                                reference: format!("session:{}", value.session.id),
                                title: value.session.title.clone().unwrap_or_default(),
                                first_prompt: String::new(),
                                repository: String::new(),
                                description: value.follow_ups.join("; "),
                            })
                            .collect(),
                    });
                }
            }
            Err(crate::protocol::failure(
                Code::BadRequest,
                "picker needs a listing",
            ))
        }
        _ => Err(crate::protocol::failure(
            Code::BadRequest,
            "picker needs a listing",
        )),
    }
}
fn query_candidates(
    found: &crate::protocol::query::Answer,
) -> Result<picker::Page, command_surface::Failure> {
    let crate::protocol::query::Answer::Rows(rows) = found else {
        return Err(crate::protocol::failure(
            Code::BadRequest,
            "pick needs native rows",
        ));
    };
    let references = rows.references.as_ref().ok_or_else(|| {
        crate::protocol::failure(Code::BadRequest, "pick needs rows with native identities")
    })?;
    if references.len() != rows.rows.len() {
        return Err(crate::protocol::protocol_mismatch(
            "rows",
            "row identities differ from the table",
        ));
    }
    Ok(picker::Page {
        count: rows.count,
        next: rows.next.clone(),
        rows: rows
            .rows
            .iter()
            .zip(references)
            .map(|(value, reference)| {
                let text = |name: &str| {
                    value
                        .get(name)
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned()
                };
                picker::Candidate {
                    reference: reference.to_string(),
                    title: text("title"),
                    first_prompt: text("first_prompt"),
                    repository: text("repo"),
                    description: text("description"),
                }
            })
            .collect(),
    })
}

struct Source {
    archive: std::path::PathBuf,
    paths: crate::config::Paths,
    shown_at: i64,
    question: String,
    request: RunnableRequest,
    frozen: Option<Vec<picker::Candidate>>,
}
impl picker::Source for Source {
    async fn page(&self, request: picker::PageRequest) -> picker::Result<picker::Page> {
        if let Some(rows) = &self.frozen {
            if request.selection != self.question {
                return Err(crate::daemon::Failure::bad_request(
                    "earlier steps fixed this population; use fuzzy mode to choose within it",
                ));
            }
            let mut fuzzy = request.clone();
            fuzzy.fuzzy.get_or_insert_default();
            return picker::fuzzy_page(rows.iter().cloned(), &fuzzy, picker::CANDIDATE_PAGE);
        }
        let mut call = self.request.clone();
        call.request_id = polylogue_store::user_writes::new_request_id();
        call.arguments["steps"] = json!([]);
        if request.selection != self.question {
            call.arguments["selection"] = json!(request.selection);
            call.arguments["references"] = json!([]);
        }
        let archive = Archive::at(&self.archive);
        if request.fuzzy.is_none() {
            call.cursor = request.cursor;
            let answer = picker_call(&archive, &call).await?;
            return candidates(&answer).map_err(crate::daemon::Failure::from);
        }
        let mut matched = picker::FuzzyRows::new(&request, picker::CANDIDATE_PAGE)?;
        loop {
            let answer = picker_call(&archive, &call).await?;
            let page = candidates(&answer).map_err(crate::daemon::Failure::from)?;
            matched.extend(page.rows);
            call.cursor = page.next;
            if call.cursor.is_none() {
                break;
            }
        }
        matched.finish()
    }
    async fn preview(&self, request: picker::PreviewRequest) -> picker::Result<String> {
        let mut call = crate::client::request(
            "rows",
            json!({"selection": request.reference, "steps": [{"verb":"read", "view":"overview", "budget":u64::from(request.width) * 4}]}),
            polylogue_store::user_writes::new_request_id(),
        );
        call.deadline = command_surface::Deadline::Milliseconds(
            u64::try_from(request.deadline.as_millis()).unwrap_or(u64::MAX),
        );
        let answer = picker_call(&Archive::at(&self.archive), &call).await?;
        match decoded(&answer).map_err(crate::daemon::Failure::from)? {
            LineAnswer::Steps(steps) => Ok(super::print::line(
                &LineAnswer::Steps(steps),
                command_surface::TextStyle::Plain,
            )),
            _ => Err(crate::daemon::Failure::from(
                crate::protocol::protocol_mismatch("rows", "preview returned no step reports"),
            )),
        }
    }
    fn act(&self, action: picker::Action, references: &[String]) -> picker::Result<()> {
        if action == picker::Action::Enter {
            return Ok(());
        }
        if action != picker::Action::Read {
            return Err(crate::daemon::Failure::from(crate::protocol::failure(
                Code::NotBuilt,
                "this destination is not built yet",
            )));
        }
        let call = crate::client::request(
            "rows",
            json!({"references":references,"steps":[{"verb":"read"}]}),
            polylogue_store::user_writes::new_request_id(),
        );
        let answer =
            self::call(&Archive::at(&self.archive), &call).map_err(crate::daemon::Failure::from)?;
        print!(
            "{}",
            super::print::line(
                &decoded(&answer).map_err(crate::daemon::Failure::from)?,
                command_surface::TextStyle::Plain
            )
        );
        Ok(())
    }
    fn shown(&self, question: &str, rows: &[picker::Candidate]) -> picker::Result<()> {
        crate::row_numbers::remember_listing(
            &self.paths,
            &self.archive,
            &crate::row_numbers::ShownRows {
                question: question.into(),
                shown_at: self.shown_at,
                rows: rows
                    .iter()
                    .map(|row| crate::row_numbers::Row {
                        reference: row.reference.clone(),
                        hits: vec![],
                    })
                    .collect(),
            },
        );
        Ok(())
    }
}

/// Deployment options are captured once before either line or command parsing.
pub(crate) fn deployment_arguments(
    arguments: &[std::ffi::OsString],
) -> Result<(crate::config::Overrides, Vec<String>), String> {
    let mut overrides = crate::config::Overrides::default();
    let mut words = vec![];
    let mut index = 0;
    while index < arguments.len() {
        let word = arguments[index]
            .to_str()
            .ok_or("command arguments must be UTF-8")?;
        let target = match word {
            "--archive" => Some(&mut overrides.archive),
            "--config" => Some(&mut overrides.config_file),
            "--auth-token-file" => Some(&mut overrides.auth_token_file),
            _ => None,
        };
        if let Some(target) = target {
            index += 1;
            *target = Some(
                arguments
                    .get(index)
                    .ok_or_else(|| format!("{word} needs a path"))?
                    .into(),
            );
        } else if word == "--listen" {
            index += 1;
            overrides.listen = Some(
                arguments
                    .get(index)
                    .and_then(|value| value.to_str())
                    .ok_or("--listen needs an address")?
                    .parse()
                    .map_err(|error| format!("--listen: {error}"))?,
            );
        } else {
            words.push(word.to_owned());
        }
        index += 1;
    }
    Ok((overrides, words))
}
/// Commands and verb signatures are available while the daemon is down.
pub(crate) fn help(words: &[String]) -> bool {
    let declarations = crate::protocol::steps::declarations();
    if words.is_empty() {
        println!(
            "polylogue SELECTION [VERB [WORD] [OPTIONS]]…\nA query is one quoted argument; a single word searches it.\n"
        );
        for command in crate::commands::COMMANDS
            .iter()
            .filter(|command| command.cli == crate::commands::Cli::Subcommand)
        {
            println!("  {:24} {}", command.name, command.help);
        }
        println!();
        for verb in declarations.verbs() {
            println!("  {}", verb.signature());
        }
        return true;
    }
    if words.len() == 1
        && let Some(verb) = declarations
            .verbs()
            .iter()
            .find(|verb| verb.verb() == words[0])
    {
        println!("{}", verb.signature());
        for field in verb.arguments().iter().filter(|field| field.listed) {
            println!("  {}  {}", field.name.replace('_', "-"), field.help);
        }
        return true;
    }
    false
}

/// Chrome invokes the executable with one extension-origin argument.
pub(crate) fn native_host(words: &[String]) -> bool {
    words.len() == 1 && words[0].starts_with("chrome-extension://")
}

/// Render one ordered stream batch at its declared step positions.
pub(crate) fn stream_steps(
    data: &str,
    parsed: &ParsedLine,
    startup: &crate::config::Startup,
    state: &mut OutputState,
) -> io::Result<std::ops::ControlFlow<u8, u8>> {
    let answer: Answer<Value> = serde_json::from_str(data).map_err(io::Error::other)?;
    let line = decoded(&answer).map_err(|error| io::Error::other(error.message))?;
    if !matches!(line, LineAnswer::Steps(_)) {
        return Err(io::Error::other("a steps event must contain step reports"));
    }
    let code = failed_code(&answer).unwrap_or_else(|| answer.exit_code());
    if parsed.line.json {
        println!(
            "{}",
            polylogue_render::terminal::json(
                &serde_json::to_value(&answer).map_err(io::Error::other)?,
                false
            )
        );
    } else {
        // A failed delivery is recorded in `state.failed`; the printed exit
        // status is not needed here.
        output(parsed, &line, &answer, startup, state);
    }
    if parsed.line.json {
        state.accept(&answer);
        if let LineAnswer::Steps(steps) = &line {
            state.offset += steps.steps.len();
        }
    }
    Ok(if let Some(code) = failed_code(&answer).or(state.failed) {
        std::ops::ControlFlow::Break(code)
    } else {
        std::ops::ControlFlow::Continue(if code == 5 { 5 } else { 0 })
    })
}
