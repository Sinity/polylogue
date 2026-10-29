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
    time::Duration,
};

pub(crate) fn run(values: &Values, request: &Request, timeout: Option<Duration>) -> ExitCode {
    match deliver(values, request, timeout) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => fail(values.flag("json"), "read_output", &error, 1),
    }
}

fn invalid(message: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

fn deliver(values: &Values, request: &Request, timeout: Option<Duration>) -> Result<(), String> {
    let archive = Archive::at(&Archive::default_root());
    let mut request = request.clone();
    let Request::Read { delivery, .. } = &request else {
        return Err("a read request is required".into());
    };
    let destination = values.text("to").map(PathBuf::from);
    if destination.is_none() && *delivery == ReadDelivery::Page {
        let response =
            super::send_with_timeout(&archive, &request, timeout).map_err(|e| e.to_string())?;
        return match response {
            crate::daemon::Response::Ok { data, .. } => {
                let page: ReadPage = serde_json::from_value(data).map_err(|e| e.to_string())?;
                let Request::Read { format, .. } = &request else {
                    unreachable!()
                };
                let format =
                    polylogue_render::Format::parse(format).ok_or("unknown read format")?;
                polylogue_render::encode_document(
                    &page.document,
                    format,
                    polylogue_render::EncodeOptions::default(),
                    &mut io::stdout().lock(),
                )
                .map_err(|e| e.to_string())
            }
            other => Err(format!("{:?}", other.answer())),
        };
    }
    let Request::Read {
        target,
        delivery,
        format,
        with_blobs,
        ..
    } = &mut request
    else {
        unreachable!()
    };
    *delivery = ReadDelivery::Stream;
    let format = polylogue_render::Format::parse(format).ok_or("unknown read format")?;
    // A query writes a package. An exact reference writes one encoded document,
    // unless an existing directory explicitly names directory delivery.
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
    consume(
        destination.as_deref(),
        directory,
        format,
        package_format,
        with_blobs,
        |callback| crate::client::stream::run(&archive, &request, timeout, callback),
    )
}

enum Sink {
    Directory(bundle::Bundle),
    Encoded {
        begin: Box<ReadBegin>,
        items: ExportItems,
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
    run: impl FnOnce(
        &mut dyn FnMut(command_surface::SseEvent) -> io::Result<bool>,
    ) -> Result<(), crate::client::stream::Error>,
) -> Result<(), String> {
    let mut sink = None;
    let mut completed = false;
    let result = run(&mut |frame| {
        match frame.event.as_str() {
            "begin" => {
                if sink.is_some() {
                    return Err(invalid("the read repeated its header"));
                }
                let begin: ReadBegin = serde_json::from_str(&frame.data).map_err(invalid)?;
                begin.options.validate().map_err(invalid)?;
                sink = Some(if directory {
                    Sink::Directory(
                        bundle::Bundle::new(
                            destination.ok_or_else(|| invalid("directory destination missing"))?,
                            begin,
                            package_format.ok_or_else(|| invalid("directory format missing"))?,
                            with_blobs,
                        )
                        .map_err(invalid)?,
                    )
                } else {
                    Sink::Encoded {
                        begin: Box::new(begin),
                        items: ExportItems::new().map_err(invalid)?,
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
                let completed = sink
                    .take()
                    .ok_or_else(|| invalid("the read completed before its header"))?;
                match completed {
                    Sink::Directory(bundle) => {
                        let summary = bundle.finish(complete).map_err(invalid)?;
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
        return Err(match sink.as_mut() {
            Some(Sink::Directory(bundle)) => bundle.abort(error.to_string()).to_string(),
            _ => error.to_string(),
        });
    }
    if !completed {
        return Err("the read stream stopped before validated completion".into());
    }
    Ok(())
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
        Record::Variant(_) => &mut counts.variants,
        Record::VariantPart(_) => &mut counts.variant_parts,
        Record::Blob(_) => &mut counts.blobs,
        Record::Manifest(_) => return Err(invalid("a read cannot contain an embedded manifest")),
    };
    *value = value
        .checked_add(1)
        .ok_or_else(|| invalid("read count overflow"))?;
    Ok(())
}
