//! Client delivery of the same finite reading. Files and directories publish
//! only after a validated completion; staged failures retain their diagnosis.
pub(crate) mod bundle;

use super::{Archive, Request, Values, fail};
use base64::Engine as _;
use polylogue_model::{Reference, bundle::Counts, read::*};
use polylogue_store::export_items::ExportItems;
use std::{
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

#[derive(Debug)]
enum DeliveryError {
    Text(String),
    Answer(Box<command_surface::Answer<serde_json::Value>>),
}
impl From<String> for DeliveryError {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}
impl From<&str> for DeliveryError {
    fn from(value: &str) -> Self {
        Self::Text(value.into())
    }
}

pub(crate) fn run(
    values: &Values,
    request: &Request,
    startup: &crate::config::Startup,
    accepted: &command_surface::RunnableRequest,
) -> ExitCode {
    match deliver(values, request, startup, accepted) {
        Ok(code) => ExitCode::from(code),
        Err(DeliveryError::Text(error)) => fail(
            startup.output,
            values.flag("json"),
            "read_output",
            &error,
            1,
        ),
        Err(DeliveryError::Answer(answer)) => {
            let code = answer.exit_code();
            if values.flag("json") {
                println!(
                    "{}",
                    polylogue_render::terminal::json(
                        &serde_json::to_value(&answer).expect("answer serializes"),
                        false
                    )
                );
            } else if let command_surface::Answer {
                outcome: command_surface::Outcome::Error(error),
                envelope: command_surface::Envelope { notes, .. },
                ..
            } = answer.as_ref()
            {
                eprint!(
                    "{}",
                    super::print::diagnostic(
                        error.code.as_deref().unwrap_or("read_output"),
                        &error.message,
                        crate::protocol::suggestion_text(error).as_deref(),
                        startup.output.stderr_color()
                    )
                );
                if let Some(hint) = &error.hint {
                    eprintln!("{}", polylogue_render::terminal::escape(hint));
                }
                for note in notes {
                    eprintln!("{}", polylogue_render::terminal::escape(note));
                }
            }
            ExitCode::from(code)
        }
    }
}
fn invalid(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

fn deliver(
    values: &Values,
    request: &Request,
    startup: &crate::config::Startup,
    accepted: &command_surface::RunnableRequest,
) -> Result<u8, DeliveryError> {
    let archive = Archive::at(&startup.config.archive);
    let delivery = match request {
        Request::Read { delivery, .. } => *delivery,
        _ => return Err("a read request is required".into()),
    };
    let destination = values.text("to").map(PathBuf::from);
    if destination.is_none() && delivery == ReadDelivery::Page {
        let answer = crate::client::call_ready(&archive, accepted).map_err(|e| e.to_string())?;
        let code = answer.exit_code();
        if values.flag("json") {
            println!(
                "{}",
                polylogue_render::terminal::json(
                    &serde_json::to_value(&answer).expect("answer serializes"),
                    false
                )
            );
            return Ok(code);
        }
        let Some(value) = answer.value() else {
            return Err(DeliveryError::Answer(Box::new(answer)));
        };
        let page: ReadPage = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
        let Request::Read { format, .. } = request else {
            unreachable!("a read request was checked above")
        };
        let format = polylogue_render::Format::parse(format).ok_or("unknown read format")?;
        let mut encoded = Vec::new();
        polylogue_render::encode_document(
            &page.document,
            format,
            polylogue_render::EncodeOptions::default(),
            &mut encoded,
        )
        .map_err(|e| e.to_string())?;
        let mut stdout = io::stdout().lock();
        if format == polylogue_render::Format::Terminal {
            let text = String::from_utf8(encoded).map_err(|e| e.to_string())?;
            let text =
                polylogue_render::terminal::bold_first_line(&text, startup.output.stdout_color());
            stdout.write_all(text.as_bytes())
        } else {
            stdout.write_all(&encoded)
        }
        .map_err(|e| e.to_string())?;
        return Ok(code);
    }
    let Request::Read {
        target,
        format,
        options,
        with_blobs,
        ..
    } = request
    else {
        return Err("stream delivery requires a full read request".into());
    };
    let format = polylogue_render::Format::parse(format).ok_or("unknown read format")?;
    let directory = destination
        .as_ref()
        .is_some_and(|path| path.is_dir() || target.parse::<Reference>().is_err());
    let package_format = match format {
        polylogue_render::Format::Json => Some(polylogue_model::bundle::Format::Json),
        polylogue_render::Format::Jsonl => Some(polylogue_model::bundle::Format::Jsonl),
        polylogue_render::Format::Markdown => Some(polylogue_model::bundle::Format::Markdown),
        polylogue_render::Format::Html => Some(polylogue_model::bundle::Format::Html),
        polylogue_render::Format::Terminal => None,
    };
    if directory && package_format.is_none() {
        return Err("directory delivery requires json, jsonl, markdown or html".into());
    }
    let with_blobs = *with_blobs;
    // Writing to a destination selects streaming delivery. Keep the caller's
    // arguments and admission fields rather than serializing execution defaults.
    let mut call = accepted.clone();
    if destination.is_some() {
        call.arguments["delivery"] = serde_json::json!("stream");
        if options.inherited {
            let mut show = values.list("show");
            if !show
                .iter()
                .any(|word| word.split(':').next() == Some("inherited"))
            {
                show.push("inherited".into());
            }
            call.arguments["show"] = serde_json::json!(show);
        }
    }
    consume(
        destination.as_deref(),
        directory,
        format,
        package_format,
        with_blobs,
        values.flag("json"),
        |callback| crate::client::stream::run_call(&archive, &call, callback),
    )
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

fn consume(
    destination: Option<&Path>,
    directory: bool,
    format: polylogue_render::Format,
    package_format: Option<polylogue_model::bundle::Format>,
    with_blobs: bool,
    json: bool,
    run: impl FnOnce(
        &mut dyn FnMut(command_surface::SseEvent) -> io::Result<bool>,
    ) -> Result<(), crate::client::stream::Error>,
) -> Result<u8, DeliveryError> {
    let mut sink = None;
    let mut completed = false;
    let mut code = 0_u8;
    let result = run(&mut |frame| {
        match frame.event.as_str() {
            "begin" => {
                if sink.is_some() {
                    return Err(invalid("the read repeated its header"));
                }
                let begin: ReadBegin = serde_json::from_str(&frame.data).map_err(invalid)?;
                begin.options.validate().map_err(invalid)?;
                if begin.snapshot.currency.degraded()
                    || !begin.snapshot.caught_up
                    || !begin.snapshot.incomplete.is_empty()
                {
                    code = 5;
                }
                sink = Some(if directory {
                    Sink::Directory(Box::new(
                        bundle::Bundle::new(
                            destination.ok_or_else(|| invalid("directory destination missing"))?,
                            begin,
                            package_format.ok_or_else(|| invalid("directory format missing"))?,
                            with_blobs,
                        )
                        .map_err(invalid)?,
                    ))
                } else {
                    Sink::Encoded {
                        begin: Box::new(begin),
                        items: Box::new(ExportItems::new().map_err(invalid)?),
                        received: Counts::default(),
                        ordinal: 0,
                    }
                });
            }
            "record" => {
                let record: ReadRecord = serde_json::from_str(&frame.data).map_err(invalid)?;
                match sink
                    .as_mut()
                    .ok_or_else(|| invalid("the read omitted its header"))?
                {
                    Sink::Directory(bundle) => bundle.push(record).map_err(invalid)?,
                    Sink::Encoded {
                        items,
                        received,
                        ordinal,
                        ..
                    } => {
                        if let Some(record) = record.record() {
                            record.validate().map_err(invalid)?;
                            count(received, record)?;
                        }
                        items
                            .push(
                                0..frame.data.len() as u64,
                                ordinal.to_be_bytes().to_vec(),
                                None,
                                frame.data.as_bytes(),
                            )
                            .map_err(invalid)?;
                        *ordinal = ordinal
                            .checked_add(1)
                            .ok_or_else(|| invalid("read transfer count overflow"))?;
                    }
                }
            }
            "blob" => {
                let chunk: ReadBlobChunk = serde_json::from_str(&frame.data).map_err(invalid)?;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(chunk.base64)
                    .map_err(invalid)?;
                match sink
                    .as_mut()
                    .ok_or_else(|| invalid("blob preceded the read header"))?
                {
                    Sink::Directory(bundle) => bundle
                        .blob_chunk(chunk.hash, chunk.offset, &bytes)
                        .map_err(invalid)?,
                    Sink::Encoded { .. } => {
                        return Err(invalid("blob transfer requires directory delivery"));
                    }
                }
            }
            "complete" => {
                let complete: ReadComplete = serde_json::from_str(&frame.data).map_err(invalid)?;
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
                let finished_sink = sink
                    .take()
                    .ok_or_else(|| invalid("the read completed before its header"))?;
                match finished_sink {
                    Sink::Directory(bundle) => {
                        let summary = bundle.finish(complete).map_err(invalid)?;
                        if json {
                            written(
                                serde_json::json!({"written":summary.destination,"counts":summary.counts,"previous":summary.previous}),
                                code,
                            )?;
                        }
                        eprintln!("polylogue: wrote {}", summary.destination.display());
                        if let Some(previous) = summary.previous {
                            eprintln!(
                                "polylogue: previous bundle retained at {}",
                                previous.display()
                            );
                        }
                    }
                    Sink::Encoded {
                        begin,
                        mut items,
                        received,
                        ..
                    } => {
                        if complete.counts != received {
                            return Err(invalid(
                                "read completion counts disagree with its records",
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
                        } else {
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
                completed = true;
                return Ok(true);
            }
            _ => return Err(invalid("unknown read stream event")),
        }
        Ok(false)
    });
    if let Err(error) = result {
        let retained = if let Some(Sink::Directory(bundle)) = sink.as_mut() {
            Some(bundle.abort(error.to_string()).to_string())
        } else {
            None
        };
        return Err(match error {
            crate::client::stream::Error::Answer(mut answer)
                if matches!(
                    answer.as_ref(),
                    command_surface::Answer {
                        outcome: command_surface::Outcome::Error(_),
                        ..
                    }
                ) =>
            {
                if let (
                    Some(note),
                    command_surface::Answer {
                        outcome: command_surface::Outcome::Error(_),
                        envelope: command_surface::Envelope { notes, .. },
                        ..
                    },
                ) = (retained, answer.as_mut())
                {
                    notes.push(note);
                }
                DeliveryError::Answer(answer)
            }
            other => DeliveryError::Text(retained.unwrap_or_else(|| other.to_string())),
        });
    }
    if !completed {
        return Err("the read stream stopped before validated completion".into());
    }
    Ok(code)
}
fn written(value: serde_json::Value, code: u8) -> io::Result<()> {
    let answer = if code == 5 {
        command_surface::Answer {
            outcome: command_surface::Outcome::Degraded {
                value,
                missing: vec![
                    "the reading has incomplete coverage or noncurrent index currency".into(),
                ],
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
    serde_json::to_writer(io::stdout().lock(), &answer)?;
    writeln!(io::stdout().lock())
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
        polylogue_render::EncodeOptions::default(),
        output,
    )?;
    while let Some(mut item) = items.next_item().map_err(invalid)? {
        let mut raw = Vec::new();
        item.reader.read_to_end(&mut raw)?;
        let record: ReadRecord = serde_json::from_slice(&raw).map_err(invalid)?;
        for part in record.parts() {
            if let ReadPart::Item(item) = part {
                encoder.push(item)?;
            }
        }
    }
    encoder.finish(&complete.footer)
}

fn count(counts: &mut Counts, record: &polylogue_model::bundle::Record) -> io::Result<()> {
    use polylogue_model::bundle::Record;
    let value = match record {
        Record::Session(_) => &mut counts.sessions,
        Record::Message(_) => &mut counts.messages,
        Record::Document(_) => &mut counts.documents,
        Record::Event(_) => &mut counts.events,
        Record::Link(_) => &mut counts.links,
        Record::Authored(_) => &mut counts.authored,
        Record::Assertion(_) => &mut counts.assertions,
        Record::Variant(_) => &mut counts.variants,
        Record::VariantPart(_) => &mut counts.variant_parts,
        Record::TextFragment(_) => &mut counts.text_fragments,
        Record::Blob(_) => &mut counts.blobs,
        Record::Manifest(_) => return Err(invalid("a read cannot contain an embedded manifest")),
    };
    *value = value
        .checked_add(1)
        .ok_or_else(|| invalid("read count overflow"))?;
    Ok(())
}
