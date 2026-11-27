//! One-record view filtering and transfer over a caller-owned pinned selection.
use crate::Error;
use polylogue_model::bundle::{Counts, Record};
use polylogue_model::read::*;
use std::io::Write;

/// Original rows are kept separate from layout-only omissions. A mixed query
/// can report unsupported targets without exporting their hidden source bodies.
pub(super) fn project(
    selected: super::records::SelectedRecord,
    options: &ReadOptions,
    facts: &LayoutFacts,
    mixed: bool,
    footer: &mut ReadFooter,
) -> Result<Option<ReadRecord>, Error> {
    let record = match selected {
        super::records::SelectedRecord::Original(record) => *record,
        super::records::SelectedRecord::LinkIssue { session, issue } => {
            return Ok(Some(ReadRecord::Presentation {
                parts: vec![ReadPart::Item(ReadItem {
                    source: Source::record(format!("session:{session}")),
                    kind: ItemKind::LinkIssue { issue: *issue },
                })],
            }));
        }
    };
    let reference = record.reference().unwrap_or_default().to_owned();
    let target_kind = match &record {
        Record::Document(_) => Some(TargetKind::Document),
        Record::TextFragment(_) => Some(TargetKind::TextFragment),
        _ => None,
    };
    match project_record(record, options, facts) {
        Ok(Some(record)) => Ok(Some(ReadRecord::Original {
            record: Box::new(record),
            parts: Vec::new(),
        })),
        Ok(None) => Ok(None),
        Err(error) => view_error(reference, target_kind, error, mixed, footer),
    }
}

/// A page can report unsupported document presentation before hydrating bodies.
/// The same omission and error policy serves the full view.
pub(super) fn document_omission(
    reference: &str,
    options: &ReadOptions,
    mixed: bool,
    footer: &mut ReadFooter,
) -> Result<Option<ReadRecord>, Error> {
    let Err(error) = polylogue_model::read::document_view(options.view, reference) else {
        return Ok(None);
    };
    view_error(
        reference.into(),
        Some(TargetKind::Document),
        error,
        mixed,
        footer,
    )
}

fn view_error(
    reference: String,
    target_kind: Option<TargetKind>,
    error: ReadError,
    mixed: bool,
    footer: &mut ReadFooter,
) -> Result<Option<ReadRecord>, Error> {
    match error {
        ReadError::UnsupportedTarget {
            view,
            target,
            accepted_views,
        } if mixed && target_kind.is_some() => {
            let target_kind = target_kind.expect("checked target family");
            if let Some(coverage) = footer
                .coverage
                .iter_mut()
                .find(|coverage| coverage.target == target_kind && coverage.view == view)
            {
                coverage.omitted += 1;
            } else {
                footer.coverage.push(Coverage {
                    target: target_kind,
                    view,
                    omitted: 1,
                    accepted_views,
                });
            }
            Ok(Some(ReadRecord::Presentation {
                parts: vec![ReadPart::Item(ReadItem {
                    source: Source::record(reference),
                    kind: ItemKind::Omission {
                        count: ValueState::known(1),
                        unit: match target_kind {
                            TargetKind::Document => "document",
                            TargetKind::TextFragment => "text fragment",
                            _ => unreachable!("mixed shown family"),
                        }
                        .into(),
                        why: ReadError::UnsupportedTarget {
                            view,
                            target,
                            accepted_views: vec![],
                        }
                        .to_string(),
                        continue_call: None,
                    },
                })],
            }))
        }
        error => Err(Error::from(error)),
    }
}

pub(super) fn emit_record(
    reader: &polylogue_store::Reader<'_>,
    record: &ReadRecord,
    at: SelectionLocator,
    layout: &super::layout_facts::Layout<'_>,
    emit: &mut super::Emit<'_>,
) -> Result<bool, Error> {
    if record.record().is_some() && !emit(record, at, None)? {
        return Ok(false);
    }
    super::layout_facts::visit_parts(reader, record, layout, |item, ordinal| {
        emit(
            &ReadRecord::Presentation {
                parts: vec![ReadPart::Item(item)],
            },
            at,
            Some(ordinal),
        )
    })
}

/// Every emit checks the pinned archive scope, including the final completion.
/// The callback remains valid after a Reader has released a prepared spool.
pub(super) struct Stream<'a> {
    output: &'a mut dyn Write,
    check: &'a dyn Fn() -> Result<(), Error>,
    counts: Counts,
    shown: u64,
}
impl<'a> Stream<'a> {
    pub(super) fn begin(
        output: &'a mut dyn Write,
        check: &'a dyn Fn() -> Result<(), Error>,
        header: &ReadHeader,
    ) -> Result<Self, Error> {
        let mut stream = Self {
            output,
            check,
            counts: Counts::default(),
            shown: 0,
        };
        stream.emit("begin", header)?;
        Ok(stream)
    }

    fn emit(&mut self, name: &str, payload: &impl serde::Serialize) -> Result<(), Error> {
        (self.check)()?;
        let frame = command_surface::SseEvent {
            id: None,
            event: name.into(),
            data: serde_json::to_string(payload).map_err(Error::internal)?,
            retry: None,
        }
        .encode()
        .map_err(Error::Operation)?;
        self.output.write_all(frame.as_bytes()).map_err(Error::io)?;
        self.output.flush().map_err(Error::io)
    }

    pub(super) fn record(&mut self, record: &ReadRecord) -> Result<(), Error> {
        if matches!(record.record(), Some(Record::Manifest(_))) {
            return Err(Error::internal("a read cannot emit an embedded manifest"));
        }
        self.emit("record", record)?;
        if let Some(record) = record.record() {
            match record {
                Record::Session(_) => self.counts.sessions += 1,
                Record::Message(_) => self.counts.messages += 1,
                Record::Document(_) => self.counts.documents += 1,
                Record::Event(_) => self.counts.events += 1,
                Record::Link(_) => self.counts.links += 1,
                Record::Authored(_) => self.counts.authored += 1,
                Record::Annotation(_) => self.counts.annotations += 1,
                Record::Alternate(_) => self.counts.alternates += 1,
                Record::AlternatePart(_) => self.counts.alternate_parts += 1,
                Record::TextFragment(_) => self.counts.text_fragments += 1,
                Record::Blob(_) => self.counts.blobs += 1,
                Record::Manifest(_) => {
                    return Err(Error::internal("a read cannot emit an embedded manifest"));
                }
            }
        }
        self.shown += record
            .parts()
            .iter()
            .filter(|part| matches!(part, ReadPart::Item(_)))
            .count() as u64;
        Ok(())
    }

    pub(super) fn blob(&mut self, chunk: &ReadBlobChunk) -> Result<(), Error> {
        self.emit("blob", chunk)
    }

    pub(super) fn finish(mut self, mut footer: ReadFooter) -> Result<ReadFooter, Error> {
        footer.shown = self.shown;
        footer.count = ValueState::known(self.shown);
        footer.counts = self.counts;
        self.emit("complete", &footer)?;
        Ok(footer)
    }
}
