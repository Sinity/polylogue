//! Client output of the same finite reading. Files and directories publish
//! only after a validated completion; staged failures keep their diagnosis.
pub(crate) mod bundle;

#[cfg(test)]
mod tests;

use super::Archive;
use base64::Engine as _;
use polylogue_model::{bundle::Counts, read::*};
use polylogue_store::export_items::ExportItems;
use std::io::{self, Read, Write};

#[derive(Debug)]
enum OutputError {
    Text(String),
    Answer(Box<command_surface::Answer<serde_json::Value>>),
}
impl From<String> for OutputError {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}
impl From<&str> for OutputError {
    fn from(value: &str) -> Self {
        Self::Text(value.into())
    }
}

fn output_failure(error: OutputError, client: &crate::config::Client, json: bool) -> u8 {
    let answer = match error {
        OutputError::Text(error) => command_surface::Answer::error(crate::protocol::failure(
            polylogue_model::Code::ReadOutput,
            error,
        )),
        OutputError::Answer(answer) => *answer,
    };
    super::fail_answer(client.output, json, &answer);
    answer.exit_code()
}

fn page_of(answer: &command_surface::Answer<serde_json::Value>) -> Result<ReadPage, OutputError> {
    let Some(value) = answer.value() else {
        return Err(OutputError::Answer(Box::new(answer.clone())));
    };
    let line: crate::protocol::line::LineAnswer =
        serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    match line {
        crate::protocol::line::LineAnswer::Steps(steps) => steps
            .steps
            .into_iter()
            .find_map(|step| match step.outcome {
                crate::protocol::line::StepOutcome::Done {
                    reply: crate::protocol::line::StepAnswer::Read(page),
                    ..
                } => Some(*page),
                _ => None,
            })
            .ok_or_else(|| OutputError::Text("the line returned no reading".into())),
        _ => Err("the line returned no reading".into()),
    }
}
/// Writes one page of a line; `last_step` is whether it is the line's final step,
/// which sends terminal output through the pager.
pub(crate) fn line_page(
    client: &crate::config::Client,
    page: &ReadPage,
    last_step: bool,
    destination: Option<&super::line::Destination>,
    format: Option<&str>,
) -> Result<(), command_surface::Failure> {
    use super::line::Destination;
    use crate::protocol::line::{LineAnswer, StepAnswer, StepOutcome, StepReport, StepsAnswer};
    let line = LineAnswer::Steps(StepsAnswer {
        resolved: None,
        steps: vec![StepReport {
            verb: "read".into(),
            request_id: None,
            outcome: StepOutcome::Done {
                reply: StepAnswer::Read(Box::new(page.clone())),
                undo: vec![],
            },
        }],
    });
    let answer = command_surface::Answer::ok(
        serde_json::to_value(line).expect("a typed reading serializes"),
    );
    let archive = Archive::at(&client.archive);
    let format = format
        .map(polylogue_render::Format::parse)
        .unwrap_or(Some(polylogue_render::Format::Terminal))
        .ok_or_else(|| {
            crate::protocol::failure(polylogue_model::Code::BadValue, "unknown read format")
        })?;
    let result = match destination {
        Some(Destination::Clipboard | Destination::Browser) => {
            return Err(crate::protocol::failure(
                polylogue_model::Code::NotBuilt,
                "this destination is not built yet",
            ));
        }
        Some(Destination::File(_) | Destination::Dir(_)) => {
            return Err(crate::protocol::failure(
                polylogue_model::Code::BadRequest,
                "file destination requires a streamed reading",
            ));
        }
        Some(Destination::Pager) => through_pager(&archive, client, answer, format),
        None if last_step && client.output.stdout_terminal => {
            through_pager(&archive, client, answer, format)
        }
        _ => one_page(client, answer, format),
    };
    result.map(|_| ()).map_err(|error| {
        crate::protocol::failure(
            polylogue_model::Code::ReadOutput,
            match error {
                OutputError::Text(text) => text,
                OutputError::Answer(answer) => serde_json::to_string(&answer).unwrap_or_default(),
            },
        )
    })
}
pub(crate) fn line_stream(
    client: &crate::config::Client,
    accepted: &command_surface::RunnableRequest,
    parsed: &super::line::ParsedLine,
    state: &mut super::run_line::OutputState,
) -> u8 {
    let mut call = accepted.clone();
    for (wire, step) in call.arguments["steps"]
        .as_array_mut()
        .expect("line steps")
        .iter_mut()
        .zip(&parsed.steps)
    {
        if step.step["follow"] == true {
            wire.as_object_mut()
                .expect("read step")
                .remove("until_idle");
        }
        if step.verb == "read"
            && matches!(
                step.to,
                Some(super::line::Destination::File(_) | super::line::Destination::Dir(_))
            )
        {
            wire["delivery"] = serde_json::json!("stream");
            // Bundles preserve inherited prefixes unless explicitly hidden.
            let directory = match &step.to {
                Some(super::line::Destination::Dir(_)) => true,
                Some(super::line::Destination::File(path)) => path.is_dir(),
                _ => false,
            };
            if directory {
                let hidden = wire["hide"]
                    .as_array()
                    .is_some_and(|words| words.iter().any(|word| word == "inherited"));
                if !hidden {
                    let show = wire
                        .as_object_mut()
                        .expect("read step")
                        .entry("show")
                        .or_insert_with(|| serde_json::json!([]))
                        .as_array_mut()
                        .expect("typed read show words");
                    if !show.iter().any(|word| word == "inherited") {
                        show.push(serde_json::json!("inherited"));
                    }
                }
            }
            if wire["format"].is_null() {
                wire["format"] = serde_json::json!("markdown");
            }
        }
    }
    let archive = Archive::at(&client.archive);
    let result = write_read_stream(client, parsed, state, |callback| {
        crate::client::call_streaming(
            &archive,
            &call,
            crate::client::connection_deadline(&call, None),
            callback,
        )
    });
    match result {
        Ok(code) => code,
        Err(error) => output_failure(error, client, parsed.line.json),
    }
}

/// One page as `format`, the first line bold on a colour terminal.
fn encode_page(
    client: &crate::config::Client,
    answer: &command_surface::Answer<serde_json::Value>,
    format: polylogue_render::Format,
) -> Result<(Vec<u8>, Option<ReadyCall>), OutputError> {
    let page = page_of(answer)?;
    let mut encoded = Vec::new();
    polylogue_render::encode_document(
        &page.document,
        format,
        polylogue_render::EncodeOptions {
            commands: Some(crate::commands::declarations()),
            ..Default::default()
        },
        &mut encoded,
    )
    .map_err(|e| e.to_string())?;
    if format == polylogue_render::Format::Terminal {
        let text = String::from_utf8(encoded).map_err(|e| e.to_string())?;
        encoded = polylogue_render::terminal::bold_first_line(&text, client.output.stdout_color())
            .into_bytes();
    }
    Ok((encoded, page.document.footer.next))
}

/// Notes and degraded parts of an answer, on standard error.
fn page_notes(client: &crate::config::Client, answer: &command_surface::Answer<serde_json::Value>) {
    for note in crate::protocol::notes(answer) {
        eprint!(
            "{}",
            super::print::diagnostic("note", note, None, client.output.stderr_color())
        );
    }
}

/// Not a terminal: the page on standard output, then its notes and the
/// command that continues it on standard error.
fn one_page(
    client: &crate::config::Client,
    answer: command_surface::Answer<serde_json::Value>,
    format: polylogue_render::Format,
) -> Result<u8, OutputError> {
    let (encoded, next) = encode_page(client, &answer, format)?;
    io::stdout()
        .lock()
        .write_all(&encoded)
        .map_err(|e| e.to_string())?;
    page_notes(client, &answer);
    if let Some(next) = next {
        eprintln!(
            "continue: {}",
            polylogue_render::terminal::escape(
                &crate::commands::declarations()
                    .cli_command(&next.request)
                    .map_err(|e| e.to_string())?
            )
        );
    }
    Ok(answer.exit_code())
}

/// On a terminal: every page of the reading, following each page's `next`,
/// encoded as one document into the pager as the pages arrive, so the
/// whole read is shown once, under one header.
fn through_pager(
    archive: &Archive,
    client: &crate::config::Client,
    mut answer: command_surface::Answer<serde_json::Value>,
    format: polylogue_render::Format,
) -> Result<u8, OutputError> {
    let first = page_of(&answer)?;
    let mut pager = std::process::Command::new("sh")
        .arg("-c")
        .arg(&client.pager)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("starting the pager `{}`: {e}", client.pager))?;
    let mut input = BoldFirstLine {
        out: pager.stdin.take().expect("piped pager input"),
        state: if format == polylogue_render::Format::Terminal && client.output.stdout_color() {
            Bold::Before
        } else {
            Bold::Done
        },
    };
    let mut code = answer.exit_code();
    let shown = (|| -> Result<(), OutputError> {
        // A pager closed before the end has shown what its reader wanted.
        let closed = |error: &io::Error| error.kind() == io::ErrorKind::BrokenPipe;
        page_notes(client, &answer);
        let mut encoder = match polylogue_render::Encoder::new(
            &first.document.header,
            format,
            polylogue_render::EncodeOptions {
                commands: Some(crate::commands::declarations()),
                ..Default::default()
            },
            &mut input,
        ) {
            Ok(encoder) => encoder,
            Err(error) if closed(&error) => return Ok(()),
            Err(error) => return Err(error.to_string().into()),
        };
        let mut page = first;
        loop {
            for item in &page.document.items {
                match encoder.push(item) {
                    Ok(()) => {}
                    Err(error) if closed(&error) => return Ok(()),
                    Err(error) => return Err(error.to_string().into()),
                }
            }
            let Some(next) = page.document.footer.next.clone() else {
                return match encoder.finish(&page.document.footer) {
                    Ok(()) => Ok(()),
                    Err(error) if closed(&error) => Ok(()),
                    Err(error) => Err(error.to_string().into()),
                };
            };
            let mut call = next.request;
            let mut rows: crate::protocol::line::Rows =
                serde_json::from_value(call.arguments).map_err(|e| e.to_string())?;
            for step in &mut rows.steps {
                if let crate::protocol::line::Step::Read(read) = step {
                    read.displayed = true;
                }
            }
            call.arguments = serde_json::to_value(rows).map_err(|e| e.to_string())?;
            answer = crate::client::call_ready(archive, &call).map_err(|e| e.to_string())?;
            code = code.max(answer.exit_code());
            page_notes(client, &answer);
            page = page_of(&answer)?;
        }
    })();
    drop(input);
    pager.wait().map_err(|e| e.to_string())?;
    shown.map(|()| code)
}

/// Where a stream is in bolding its first line.
enum Bold {
    Before,
    Inside,
    Done,
}

/// Writes through, with the first line in bold, as
/// [`polylogue_render::terminal::bold_first_line`] does for a whole text.
struct BoldFirstLine<W> {
    out: W,
    state: Bold,
}

impl<W: Write> Write for BoldFirstLine<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        if let Bold::Before = self.state {
            self.out.write_all(b"\x1b[1m")?;
            self.state = Bold::Inside;
        }
        if let Bold::Inside = self.state
            && let Some(end) = bytes.iter().position(|&b| b == b'\n')
        {
            self.out.write_all(&bytes[..end])?;
            self.out.write_all(b"\x1b[0m")?;
            self.out.write_all(&bytes[end..])?;
            self.state = Bold::Done;
            return Ok(bytes.len());
        }
        self.out.write_all(bytes)?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

enum Sink {
    Directory(Box<bundle::Bundle>),
    Encoded {
        begin: Box<ReadBegin>,
        items: Box<ExportItems>,
        received: Counts,
        ordinal: u64,
    },
}

fn write_read_stream(
    client: &crate::config::Client,
    parsed: &super::line::ParsedLine,
    state: &mut super::run_line::OutputState,
    run: impl FnOnce(
        &mut dyn FnMut(command_surface::SseEvent) -> io::Result<bool>,
    ) -> io::Result<Option<command_surface::Answer<serde_json::Value>>>,
) -> Result<u8, OutputError> {
    use super::line::Destination;
    let json = parsed.line.json;
    let mut destination = None;
    let mut format = polylogue_render::Format::Terminal;
    let mut sink = None;
    let mut completed = false;
    let mut code = 0_u8;
    let mut failed = None;
    let mut on_frame = |frame: command_surface::SseEvent| {
        let follow_record = frame.event == "record"
            && parsed
                .steps
                .get(state.offset)
                .is_some_and(|step| step.step["follow"] == true);
        if json && !follow_record {
            let data: serde_json::Value =
                serde_json::from_str(&frame.data).map_err(io::Error::other)?;
            writeln!(
                io::stdout().lock(),
                "{}",
                polylogue_render::terminal::json(
                    &serde_json::json!({"event":frame.event,"data":data}),
                    false
                )
            )?;
        }
        if matches!(frame.event.as_str(), "steps" | "answer") {
            let answer: command_surface::Answer<crate::protocol::line::LineAnswer> =
                serde_json::from_str(&frame.data).map_err(io::Error::other)?;
            if let Some(crate::protocol::line::LineAnswer::Steps(steps)) = answer.value() {
                for step in &steps.steps {
                    if let crate::protocol::line::StepOutcome::Failed { error } = &step.outcome {
                        failed.get_or_insert(error.kind.exit_code());
                    }
                }
            }
            if !json {
                let reported = super::run_line::stream_steps(&frame.data, parsed, client, state)?;
                match reported {
                    std::ops::ControlFlow::Continue(5) => code = 5,
                    std::ops::ControlFlow::Break(failure) => {
                        failed.get_or_insert(failure);
                    }
                    _ => {}
                }
            } else if let Some(crate::protocol::line::LineAnswer::Steps(steps)) = answer.value() {
                state.accept(&answer);
                state.offset += steps.steps.len();
            }
            if frame.event == "answer" {
                if sink.is_some() {
                    return Err(io::Error::other("line ended inside a read"));
                }
                completed = true;
                // A read this stream delivered degraded stays degraded; an
                // empty read defers to the line, which other steps may
                // have answered.
                code = failed.unwrap_or_else(|| if code == 5 { 5 } else { state.exit_code() });
                return Ok(true);
            }
            return Ok(false);
        }
        if follow_record {
            let step = &parsed.steps[state.offset].step;
            if super::follow::record(client, step, &frame.data, json)? {
                completed = true;
                return Ok(true);
            }
            return Ok(false);
        }
        match frame.event.as_str() {
            "begin" => {
                let step = parsed
                    .steps
                    .get(state.offset)
                    .ok_or_else(|| io::Error::other("read stream has no declared step"))?;
                destination = match &step.to {
                    Some(Destination::File(path) | Destination::Dir(path)) => Some(path.as_path()),
                    _ => None,
                };
                let directory = matches!(&step.to, Some(Destination::Dir(_)))
                    || destination.is_some_and(|path| path.is_dir());
                format = polylogue_render::Format::parse(step.step["format"].as_str().unwrap_or(
                    if destination.is_some() {
                        "markdown"
                    } else {
                        "terminal"
                    },
                ))
                .ok_or_else(|| io::Error::other("unknown read format"))?;
                let package_format = match format {
                    polylogue_render::Format::Json => Some(polylogue_model::bundle::Format::Json),
                    polylogue_render::Format::Jsonl => Some(polylogue_model::bundle::Format::Jsonl),
                    polylogue_render::Format::Markdown => {
                        Some(polylogue_model::bundle::Format::Markdown)
                    }
                    polylogue_render::Format::Html => Some(polylogue_model::bundle::Format::Html),
                    polylogue_render::Format::Terminal => None,
                };
                completed = false;
                if sink.is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "the read repeated its header".to_string(),
                    ));
                }
                let begin: ReadBegin = serde_json::from_str(&frame.data).map_err(|error| {
                    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                })?;
                begin.options.validate().map_err(|error| {
                    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                })?;
                if !begin.snapshot.freshness.up_to_date() || !begin.snapshot.incomplete.is_empty() {
                    code = 5;
                }
                sink = Some(if directory {
                    Sink::Directory(Box::new(
                        bundle::Bundle::new(
                            destination.ok_or_else(|| {
                                io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "directory destination missing".to_string(),
                                )
                            })?,
                            begin,
                            package_format.ok_or_else(|| {
                                io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "directory format missing".to_string(),
                                )
                            })?,
                            true,
                        )
                        .map_err(|error| {
                            io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                        })?,
                    ))
                } else {
                    Sink::Encoded {
                        begin: Box::new(begin),
                        items: Box::new(ExportItems::new().map_err(|error| {
                            io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                        })?),
                        received: Counts::default(),
                        ordinal: 0,
                    }
                });
            }
            "record" => {
                let record: ReadRecord = serde_json::from_str(&frame.data).map_err(|error| {
                    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                })?;
                match sink.as_mut().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "the read omitted its header".to_string(),
                    )
                })? {
                    Sink::Directory(bundle) => bundle.push(record).map_err(|error| {
                        io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                    })?,
                    Sink::Encoded {
                        items,
                        received,
                        ordinal,
                        ..
                    } => {
                        if let Some(record) = record.record() {
                            record.validate().map_err(|error| {
                                io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                            })?;
                            count_record(received, record)?;
                        }
                        items
                            .push(
                                0..frame.data.len() as u64,
                                ordinal.to_be_bytes().to_vec(),
                                None,
                                frame.data.as_bytes(),
                            )
                            .map_err(|error| {
                                io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                            })?;
                        *ordinal = ordinal.checked_add(1).ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "read transfer count overflow".to_string(),
                            )
                        })?;
                    }
                }
            }
            "blob" => {
                let chunk: ReadBlobChunk = serde_json::from_str(&frame.data).map_err(|error| {
                    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                })?;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(chunk.base64)
                    .map_err(|error| {
                        io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                    })?;
                match sink.as_mut().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "blob preceded the read header".to_string(),
                    )
                })? {
                    Sink::Directory(bundle) => bundle
                        .blob_chunk(chunk.hash, chunk.offset, &bytes)
                        .map_err(|error| {
                            io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                        })?,
                    Sink::Encoded { .. } => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "blob transfer requires directory delivery".to_string(),
                        ));
                    }
                }
            }
            "complete" => {
                let complete: ReadComplete =
                    serde_json::from_str(&frame.data).map_err(|error| {
                        io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                    })?;
                if complete
                    .footer
                    .coverage
                    .iter()
                    .any(|coverage| coverage.omitted > 0)
                {
                    code = 5;
                }
                if code == 0 && complete.footer.shown == 0 && complete.counts == Counts::default() {
                    code = 1;
                }
                let finished_sink = sink.take().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "the read completed before its header".to_string(),
                    )
                })?;
                match finished_sink {
                    Sink::Directory(bundle) => {
                        let summary = bundle.finish(complete).map_err(|error| {
                            io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                        })?;
                        if json {
                            written(
                                serde_json::json!({"written":summary.destination,"counts":summary.counts,"previous":summary.previous}),
                                code,
                            )?;
                        }
                        eprintln!("polylogue: wrote {}", summary.destination.display());
                        if let Some(previous) = summary.previous {
                            eprintln!("polylogue: previous bundle kept at {}", previous.display());
                        }
                    }
                    Sink::Encoded {
                        begin,
                        mut items,
                        received,
                        ..
                    } => {
                        if complete.counts != received {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "read completion counts disagree with its records".to_string(),
                            ));
                        }
                        if let Some(path) = destination {
                            let mut output = super::staged_file::StagedFile::new(path)?;
                            encode(&begin, &mut items, &complete, format, &mut output)?;
                            let path = output.commit()?;
                            if json {
                                written(
                                    serde_json::json!({"written":path,"counts":complete.counts}),
                                    code,
                                )?;
                            }
                            eprintln!("polylogue: wrote {}", path.display());
                        } else if !json {
                            encode(
                                &begin,
                                &mut items,
                                &complete,
                                format,
                                &mut io::stdout().lock(),
                            )?;
                        }
                    }
                }
                return Ok(false);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown read stream event".to_string(),
                ));
            }
        }
        Ok(false)
    };
    // The final answer ends the line like its `steps` events; a failure
    // after events have arrived is that answer's error.
    let result = match run(&mut on_frame) {
        Ok(Some(answer)) if !answer.is_error() => serde_json::to_string(&answer)
            .map_err(io::Error::other)
            .and_then(|data| {
                on_frame(command_surface::SseEvent {
                    id: None,
                    event: "answer".into(),
                    data,
                    retry: None,
                })
            })
            .map(drop)
            .map_err(Refusal::Io),
        Ok(Some(answer)) => Err(Refusal::Answer(Box::new(answer))),
        Ok(None) => Ok(()),
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            ) =>
        {
            let mut lifted = super::run_line::request(parsed).map_err(|error| error.message)?;
            lifted.deadline = command_surface::Deadline::None;
            let answer = command_surface::Answer::error(
                crate::protocol::store_failure(&polylogue_store::StoreError::ReadDeadlineExceeded)
                    .with_hint("rerun with `--deadline none`, or narrow to one session"),
            )
            .with_next("complete without a deadline", lifted);
            Err(Refusal::Answer(Box::new(answer)))
        }
        Err(error) => Err(Refusal::Io(error)),
    };
    if let Err(error) = result {
        let kept = if let Some(Sink::Directory(bundle)) = sink.as_mut() {
            Some(bundle.error(error.to_string()).to_string())
        } else {
            None
        };
        return Err(match error {
            Refusal::Answer(mut answer) => {
                if let (
                    Some(note),
                    command_surface::Answer {
                        outcome: command_surface::Outcome::Error(_),
                        envelope: command_surface::Envelope { notes, .. },
                        ..
                    },
                ) = (kept, answer.as_mut())
                {
                    notes.push(note);
                }
                OutputError::Answer(answer)
            }
            Refusal::Io(other) => OutputError::Text(kept.unwrap_or_else(|| other.to_string())),
        });
    }
    if !completed {
        return Err("the read stream stopped before validated completion".into());
    }
    Ok(code)
}
/// How a streamed line ended without completing.
enum Refusal {
    Answer(Box<command_surface::Answer<serde_json::Value>>),
    Io(io::Error),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Answer(answer) => match answer.failure() {
                Some(failure) => formatter.write_str(&failure.message),
                None => formatter.write_str("the line failed"),
            },
            Self::Io(error) => error.fmt(formatter),
        }
    }
}

fn written(value: serde_json::Value, code: u8) -> io::Result<()> {
    let answer = if code == 5 {
        command_surface::Answer {
            outcome: command_surface::Outcome::Degraded {
                value,
                missing: vec!["the reading has incomplete coverage or a stale index".into()],
            },
            envelope: command_surface::Envelope {
                next: vec![],
                ..Default::default()
            },
        }
    } else if code == 1 {
        command_surface::Answer {
            outcome: command_surface::Outcome::Empty(value),
            envelope: command_surface::Envelope {
                notes: vec![],
                next: vec![],
                ..Default::default()
            },
        }
    } else {
        command_surface::Answer::ok(value)
    };
    let event = serde_json::json!({"event":"written", "data":answer});
    writeln!(
        io::stdout().lock(),
        "{}",
        polylogue_render::terminal::json(&event, false)
    )
}

fn encode(
    begin: &ReadBegin,
    items: &mut ExportItems,
    complete: &ReadComplete,
    format: polylogue_render::Format,
    output: &mut dyn Write,
) -> io::Result<()> {
    let mut encoder = polylogue_render::Encoder::new(
        &begin.header,
        format,
        polylogue_render::EncodeOptions {
            commands: Some(crate::commands::declarations()),
            ..Default::default()
        },
        output,
    )?;
    while let Some(mut item) = items
        .next_item()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?
    {
        let mut raw = Vec::new();
        item.reader.read_to_end(&mut raw)?;
        let record: ReadRecord = serde_json::from_slice(&raw)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        for part in record.parts() {
            if let ReadPart::Item(item) = part {
                encoder.push(item)?;
            }
        }
    }
    encoder.finish(&complete.footer)
}

fn count_record(counts: &mut Counts, record: &polylogue_model::bundle::Record) -> io::Result<()> {
    use polylogue_model::bundle::Record;
    let value = match record {
        Record::Session(_) => &mut counts.sessions,
        Record::Message(_) => &mut counts.messages,
        Record::Document(_) => &mut counts.documents,
        Record::Event(_) => &mut counts.events,
        Record::Link(_) => &mut counts.links,
        Record::Authored(_) => &mut counts.authored,
        Record::Annotation(_) => &mut counts.annotations,
        Record::Alternate(_) => &mut counts.alternates,
        Record::AlternatePart(_) => &mut counts.alternate_parts,
        Record::TextFragment(_) => &mut counts.text_fragments,
        Record::Blob(_) => &mut counts.blobs,
        Record::Manifest(_) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "a read cannot contain an embedded manifest".to_string(),
            ));
        }
    };
    *value = value.checked_add(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "read count overflow".to_string(),
        )
    })?;
    Ok(())
}
